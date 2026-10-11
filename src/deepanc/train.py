"""Training entry point for a single deep-ANC model variant.

Usage:
  python -m deepanc.train --config configs/exp/crn_n.yaml [key=value ...]

Implements the training step of Section 3 / Fig. 2: for each batch, run
the reference signal through the CRN and the simulated acoustic pipeline
(anc.simulate), compute the loss (Eq. 4 or 6) on the resulting error
signal, and backprop. Supports checkpoint/resume for SLURM preemption
(Section "SLURM" of the plan): on SIGUSR1 (sent ~120s before a requeue
deadline) or at every `checkpoint_every` epochs, save a full training
state (model, optimizer, epoch) so a requeued job resumes exactly.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepanc.acoustics.paths import batched_fir_convolve, nl_secondary_path, volterra_primary
from deepanc.acoustics.rir import load_rir_sets
from deepanc.anc.losses import loss_n, loss_ns
from deepanc.anc.simulate import simulate
from deepanc.config import load_config
from deepanc.data.datasets import ANCNoiseDataset, ANCNoisySpeechDataset
from deepanc.metrics.nmse import nmse_db_batch
from deepanc.models.crn import CRN

_STOP_REQUESTED = False


def _handle_sigusr1(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def build_dataset(cfg, split: str):
    d = cfg.data
    if cfg.train.mode == "n":
        h5 = d.noise_train_h5 if split == "train" else d.noise_val_h5
        n_items = d.n_train_items if split == "train" else d.n_val_items
        return ANCNoiseDataset(
            h5, n_items=n_items, base_seed=d.base_seed + (0 if split == "train" else 10_000_000),
            resample_each_epoch=d.resample_each_epoch if split == "train" else False,
            n_rirs=_n_rirs(cfg),
        )
    elif cfg.train.mode == "ns":
        h5 = d.noise_train_h5 if split == "train" else d.noise_val_h5
        n_items = d.n_train_items if split == "train" else d.n_val_items
        return ANCNoisySpeechDataset(
            h5, d.timit_h5, n_items=n_items, base_seed=d.base_seed + (0 if split == "train" else 10_000_000),
            resample_each_epoch=d.resample_each_epoch if split == "train" else False,
            n_rirs=_n_rirs(cfg),
        )
    raise ValueError(cfg.train.mode)


def _n_rirs(cfg) -> int:
    rir_sets = load_rir_sets(cfg.data.rirs_npz)
    return rir_sets[cfg.train.rir_set].primary.shape[0]


def get_primary_fn(cfg):
    if cfg.train.primary_model == "volterra":
        return volterra_primary
    return None


def get_secondary_fn(cfg):
    if cfg.train.secondary_model == "nl":
        return nl_secondary_path
    return None


def run_step(model, batch, rir_primary, rir_secondary, cfg, device):
    x = batch["x"].to(device)
    eta2 = batch["eta2"].to(device)
    rir_idx = batch["rir_idx"]
    primary = rir_primary[rir_idx].to(device)
    secondary = rir_secondary[rir_idx].to(device)

    out = simulate(
        model, x, primary_rir=primary, secondary_rir=secondary,
        eta2=eta2, m_frames=cfg.train.m_frames,
        primary_fn=get_primary_fn(cfg), secondary_fn=get_secondary_fn(cfg),
    )

    if cfg.train.mode == "n":
        loss = loss_n(out.e)
    else:
        v = batch["speech"].to(device)
        p_conv_v = batched_fir_convolve(v, primary)
        loss = loss_ns(out.e, p_conv_v)

    if cfg.train.get("loss_norm", "none") == "d_power":
        # Same minimizer as Eq. 4/6 (a per-batch positive constant), but
        # rescaled to O(1): at the training level (RMS ~0.004) the raw MSE
        # is ~1e-5..1e-7 and the median gradient ~1e-10, below Adam's
        # eps=1e-8. A 600-step A/B didn't show a clear win, so it's opt-in.
        loss = loss / (out.d.detach() ** 2).mean().clamp_min(1e-12)

    return loss, out


def train(cfg):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    best_path = out_dir / "best.pt"

    model = CRN(
        n_freq=cfg.model.n_freq, channels=list(cfg.model.channels),
        lstm_layers=cfg.model.lstm_layers, lstm_groups=cfg.model.lstm_groups,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, amsgrad=cfg.train.amsgrad)

    rir_sets = load_rir_sets(cfg.data.rirs_npz)
    rset = rir_sets[cfg.train.rir_set]
    rir_primary = torch.from_numpy(rset.primary).float()
    rir_secondary = torch.from_numpy(rset.secondary).float()

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True, persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    start_epoch = 0
    best_val_nmse = float("inf")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_nmse = state.get("best_val_nmse", float("inf"))
        print(f"resumed from epoch {start_epoch}")

    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    def save_checkpoint(epoch, path):
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val_nmse": best_val_nmse},
            path,
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = run_step(model, batch, rir_primary, rir_secondary, cfg, device)
            loss.backward()
            if cfg.train.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

            if _STOP_REQUESTED:
                print("SIGUSR1 received, checkpointing and exiting for requeue")
                save_checkpoint(epoch - 1, ckpt_path)  # last fully-completed epoch
                writer.close()
                sys.exit(0)

        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_nmse_vals = []
        val_loss_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss, out = run_step(model, batch, rir_primary, rir_secondary, cfg, device)
                val_loss_total += loss.item()
                n_val_batches += 1
                val_nmse_vals.append(nmse_db_batch(out.e, out.d).cpu().numpy())
        val_loss = val_loss_total / max(n_val_batches, 1)
        val_nmse = float(np.mean(np.concatenate(val_nmse_vals))) if val_nmse_vals else float("nan")

        dt = time.time() - t0
        # loss = mean squared error on ~0.01-0.1 RMS audio, so it lives
        # around 1e-4 to 1e-8 — %.6f rounds most of that range to
        # "0.000000"/"0.000001" and makes real progress look flat; %.3e
        # keeps it legible. val_nmse_db is still the primary signal to
        # watch (it's what best.pt is selected on).
        print(f"epoch {epoch}: train_loss={train_loss:.3e} val_loss={val_loss:.3e} val_nmse_db={val_nmse:.3f} ({dt:.1f}s)")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("nmse_db/val", val_nmse, epoch)

        # CRN-n's target is e=0, so NMSE(e, d) is its objective. CRN-ns's
        # target is e = p*v, and NMSE(e, d) rewards also cancelling the
        # speech, so select on its actual validation loss (Eq. 6) instead.
        # best_val_nmse keeps its name for checkpoint compatibility.
        score = val_nmse if cfg.train.mode == "n" else val_loss
        if score < best_val_nmse:
            best_val_nmse = score
            save_checkpoint(epoch, best_path)

        if (epoch + 1) % cfg.train.checkpoint_every == 0:
            save_checkpoint(epoch, ckpt_path)

    writer.close()
    print(f"training complete; best val NMSE = {best_val_nmse:.3f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides=overrides)
    print(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
