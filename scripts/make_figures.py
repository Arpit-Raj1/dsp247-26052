#!/usr/bin/env python
"""Reproduce Figs. 7-10 (Section 7 of the plan) as PNGs under
results/figures/. Unlike make_tables.py (which only reads results/*.json
summaries), Figs. 7 and 10 need per-sample signal detail — a short-time
NMSE curve and a waveform/spectrogram — that evaluate.py's per-condition
summary JSON doesn't retain. This script recomputes those specific single
signals directly from a trained checkpoint (or a baseline run), reusing
the same simulate()/baseline machinery as evaluate.py and train.py.

Usage:
  python scripts/make_figures.py --results-dir results --testsets data/testsets.h5 \
      --rirs data/rirs.npz --configs-dir configs/exp --runs-dir runs --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepanc.acoustics.rir import load_rir_sets
from deepanc.anc.simulate import simulate
from deepanc.config import load_config
from deepanc.data.testsets import load_condition
from deepanc.evaluate import default_mu
from deepanc.metrics.spectrum import short_time_nmse_db, welch_psd_db
from deepanc.models.crn import CRN

BASELINE_RUNNERS = {}


def rev_terrain_colors(n: int = 256):
    """R's rev(terrain.colors(n)) as a matplotlib colormap. R builds
    terrain.colors from two HSV ramps (green->yellow, yellow->near-white);
    matplotlib's own "terrain" adds a blue band, so rebuild R's exactly."""
    import colorsys

    from matplotlib.colors import ListedColormap

    k = n // 2
    h, sat, v = (4 / 12, 2 / 12, 0 / 12), (1.0, 1.0, 0.0), (0.65, 0.9, 0.95)
    ramp = lambda a, b, m: np.linspace(a, b, m)
    first = zip(ramp(h[0], h[1], k), ramp(sat[0], sat[1], k), ramp(v[0], v[1], k))
    second = zip(ramp(h[1], h[2], n - k + 1)[1:], ramp(sat[1], sat[2], n - k + 1)[1:], ramp(v[1], v[2], n - k + 1)[1:])
    colors = [colorsys.hsv_to_rgb(*c) for c in list(first) + list(second)]
    return ListedColormap(colors[::-1], name="rev_terrain")


def spec_db(x, sr=16000):
    from scipy.signal import spectrogram

    f, t, S = spectrogram(x, fs=sr, nperseg=320, noverlap=160)
    return f, t, 10 * np.log10(S + 1e-20)

try:
    from deepanc.baselines.fxlms import run_fxlms
    from deepanc.baselines.sec_path_id import identify_secondary_path
    from deepanc.baselines.thf_fxlms import run_thf_fxlms
    from deepanc.acoustics.loudspeaker import sef

    BASELINE_RUNNERS = {"fxlms": run_fxlms, "thf_fxlms": run_thf_fxlms}
except ImportError:
    pass


def load_model_checkpoint(runs_dir: Path, configs_dir: Path, name: str, device):
    cfg = load_config(configs_dir / f"{name}.yaml")
    model = CRN(
        n_freq=cfg.model.n_freq, channels=list(cfg.model.channels),
        lstm_layers=cfg.model.lstm_layers, lstm_groups=cfg.model.lstm_groups,
    ).to(device)
    state = torch.load(runs_dir / name / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def signal_e_d(cond, idx: int, rset, source: str, runs_dir: Path, configs_dir: Path, device, mu=None):
    """Returns (e, d) numpy arrays for test signal `idx` of `cond`, either
    from a trained CRN checkpoint (source is a config/exp name) or a
    classical baseline (source in BASELINE_RUNNERS).
    """
    x = cond.x[idx]
    primary = rset.primary[cond.rir_idx[idx]]
    secondary = rset.secondary[cond.rir_idx[idx]]
    eta2 = float(cond.eta2[idx])

    if source in BASELINE_RUNNERS:
        d = np.convolve(x.astype(np.float64), primary.astype(np.float64))[: len(x)]
        loudspeaker_fn = (lambda y, eta2=eta2: sef(torch.from_numpy(y), eta2).numpy()) if eta2 != float("inf") else (lambda y: y)
        shat = identify_secondary_path(secondary, loudspeaker_fn, n_taps=min(512, len(secondary)))
        if mu is None:
            mu = default_mu(source, cond.meta.get("noise"))
        e, _ = BASELINE_RUNNERS[source](x.astype(np.float64), d, secondary_rir_true=secondary, secondary_rir_hat=shat, eta2=eta2, mu=mu, n_taps=len(shat))
        return e, d
    else:
        model, cfg = load_model_checkpoint(runs_dir, configs_dir, source, device)
        with torch.no_grad():
            xb = torch.from_numpy(x).float().unsqueeze(0).to(device)
            pr = torch.from_numpy(primary).float().unsqueeze(0).to(device)
            sr = torch.from_numpy(secondary).float().unsqueeze(0).to(device)
            eta2b = torch.tensor([eta2]).float().to(device)
            out = simulate(model, xb, primary_rir=pr, secondary_rir=sr, eta2=eta2b, m_frames=cfg.train.m_frames)
        return out.e[0].cpu().numpy(), out.d[0].cpu().numpy()


def fig7(testsets_h5, rirs_npz, runs_dir, configs_dir, out_dir, device):
    """(a) short-time NMSE vs time, (b) Welch PSD — engine noise, eta2=0.1,
    comparing CRN-n, FxLMS, THF-FxLMS on the same representative signal.
    """
    rir_sets = load_rir_sets(rirs_npz)
    with h5py.File(testsets_h5, "r") as tf:
        cond = load_condition(tf["table1"], "engine_eta0.1")
    rset = rir_sets[cond.meta.get("rir_set", "test")]
    idx = 0

    # FxLMS and THF-FxLMS give identical results at this signal level (the
    # tanh derivative is ~1), so FxLMS is drawn thick underneath and
    # THF-FxLMS dashed on top, or FxLMS would be hidden entirely.
    styles = {
        "crn_n": dict(color="tab:blue", lw=1.5, zorder=3),
        "fxlms": dict(color="tab:orange", lw=4.0, alpha=0.6, zorder=1),
        "thf_fxlms": dict(color="tab:green", lw=1.3, ls="--", zorder=2),
    }
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for source, label in [("crn_n", "CRN-n"), ("fxlms", "FxLMS"), ("thf_fxlms", "THF-FxLMS")]:
        try:
            e, d = signal_e_d(cond, idx, rset, source, runs_dir, configs_dir, device)
        except FileNotFoundError:
            print(f"skip {source}: checkpoint not found")
            continue
        st_nmse = short_time_nmse_db(e, d, window_s=0.1)
        t = np.arange(len(st_nmse)) * 0.1
        axes[0].plot(t, st_nmse, label=label, **styles[source])
        freqs, psd = welch_psd_db(e)
        axes[1].plot(freqs / 1000, psd, label=label, **styles[source])

    freqs, psd_d = welch_psd_db(d)
    axes[1].plot(freqs / 1000, psd_d, "k--", label="No ANC", alpha=0.6)

    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("NMSE (dB)"); axes[0].legend(); axes[0].set_title("(a) Short-time NMSE")
    axes[0].text(0.99, 0.02, "FxLMS and THF-FxLMS overlap (identical at this level)",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7, color="dimgray")
    axes[1].set_xlabel("Frequency (kHz)"); axes[1].set_ylabel("PSD (dB)"); axes[1].legend(); axes[1].set_title("(b) Power spectrum")
    fig.tight_layout()
    fig.savefig(out_dir / "fig7.png", dpi=150)
    plt.close(fig)
    print("wrote fig7.png")


def fig8(results_dir, rirs_npz, out_dir):
    path = results_dir / "crn_n_fig8.json"
    if not path.exists():
        print("skip fig8: crn_n_fig8.json not found"); return
    with open(path) as f:
        data = json.load(f)
    rir_sets = load_rir_sets(rirs_npz)
    t60s = rir_sets["test_untrained_t60"].t60
    keys = [f"t60_{i}" for i in range(len(t60s))]
    bars = {float(t): data[k]["nmse_db"] for t, k in zip(t60s, keys)}
    # The paper's Fig. 8 includes the trained T60 (0.2s, i.e. the Table 1
    # engine/eta2=0.1 condition) as a reference bar.
    ref_path = results_dir / "crn_n_table1.json"
    ref_t60 = float(rir_sets["test"].t60[0])
    if ref_path.exists():
        with open(ref_path) as f:
            bars[ref_t60] = json.load(f)["engine_eta0.1"]["nmse_db"]
    order = sorted(bars)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{t:.3f}" for t in order], [bars[t] for t in order],
           color=["gray" if abs(t - ref_t60) < 1e-9 else "black" for t in order])
    ax.set_xlabel("T60 (s)"); ax.set_ylabel("NMSE (dB)"); ax.set_title("CRN-n NMSE vs. untrained T60 (engine, eta2=0.1)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig8.png", dpi=150)
    plt.close(fig)
    print("wrote fig8.png")


def fig9(results_dir, testsets_h5, rirs_npz, out_dir):
    from make_tables import unprocessed_row  # reuse the same computation

    models = ["fxlms", "thf_fxlms", "crn_n", "crn_ns", "crn_ns_m1", "crn_ns_m2"]
    noises = ["factory", "babble", "ssn"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    width = 0.8 / len(models)
    x = np.arange(len(noises))

    for mi, m in enumerate(models):
        path = results_dir / f"{m}_fig9.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        dstoi, dpesq, nmse = [], [], []
        for n in noises:
            c = data.get(f"{n}_5db", {})
            u = unprocessed_row(testsets_h5, rirs_npz, "fig9", f"{n}_5db") or {}
            dstoi.append((c.get("stoi") or 0) - (u.get("stoi") or 0))
            dpesq.append((c.get("pesq_wb") or 0) - (u.get("pesq_wb") or 0))
            nmse.append(c.get("nmse_db") or 0)
        offset = (mi - len(models) / 2) * width
        axes[0].bar(x + offset, dstoi, width, label=m)
        axes[1].bar(x + offset, dpesq, width, label=m)
        axes[2].bar(x + offset, nmse, width, label=m)

    for ax, title in zip(axes, ["Delta STOI", "Delta PESQ", "NMSE (dB)"]):
        ax.set_xticks(x); ax.set_xticklabels(noises); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9.png", dpi=150)
    plt.close(fig)
    print("wrote fig9.png")


def _example(group, cname, runs_dir, configs_dir, device, testsets_h5, rirs_npz, sources):
    """(d, {source: e}) for signal 0 of a test condition."""
    rir_sets = load_rir_sets(rirs_npz)
    with h5py.File(testsets_h5, "r") as tf:
        cond = load_condition(tf[group], cname)
    rset = rir_sets[cond.meta.get("rir_set", "test")]
    es, d = {}, None
    for src in sources:
        try:
            e, d = signal_e_d(cond, 0, rset, src, runs_dir, configs_dir, device)
            es[src] = e
        except FileNotFoundError:
            print(f"skip {src}: checkpoint not found")
    return d, es


def fig10(testsets_h5, rirs_npz, runs_dir, configs_dir, out_dir, device):
    """Paper Fig. 10: CRN-ns, engine noise, eta2=0.1 — noise only (top) and
    noisy speech at 5dB (bottom). Waveforms of d (black) and e (gray)
    overlaid, then spectrograms of d and e on a shared dB scale, with R's
    rev(terrain.colors) palette."""
    rows = [("table2_noise", "engine", "Noise only"), ("table2_snr", "snr5", "Noisy speech, SNR = 5 dB")]
    cmap = rev_terrain_colors()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"width_ratios": [1.2, 1, 1]}, layout="constrained")
    for r, (group, cname, label) in enumerate(rows):
        d, es = _example(group, cname, runs_dir, configs_dir, device, testsets_h5, rirs_npz, ["crn_ns"])
        if "crn_ns" not in es:
            print("skip fig10: crn_ns checkpoint not found"); plt.close(fig); return
        e = es["crn_ns"]
        t = np.arange(len(d)) / 16000
        ax = axes[r, 0]
        ax.plot(t, d, color="black", lw=0.5, label="primary d(t)")
        ax.plot(t, e, color="gray", lw=0.5, alpha=0.9, label="error e(t)")
        ax.set_title(f"({'ad'[r]}) {label}: waveforms"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        f, tt, Sd = spec_db(d)
        _, _, Se = spec_db(e)
        vmax = np.percentile(Sd, 99.5); vmin = vmax - 80
        for c, (S, name) in enumerate([(Sd, "primary d(t)"), (Se, "error e(t)")], start=1):
            im = axes[r, c].pcolormesh(tt, f / 1000, S, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"({'bcef'[2 * r + c - 1]}) Spectrogram: {name}")
            axes[r, c].set_xlabel("Time (s)"); axes[r, c].set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=axes[r, 1:], label="dB", fraction=0.025)
    fig.suptitle("CRN-ns, engine noise, loudspeaker nonlinearity eta2 = 0.1")
    fig.savefig(out_dir / "fig10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig10.png")


def fig_waveforms(testsets_h5, rirs_npz, runs_dir, configs_dir, out_dir, device):
    """Primary d(t) vs residual e(t) for CRN-n and CRN-ns, noise only and
    noisy speech (engine, eta2=0.1, signal 0), with each panel's NMSE."""
    from deepanc.metrics.nmse import nmse_db

    rows = [("table2_noise", "engine", "Noise only"), ("table2_snr", "snr5", "Noisy speech, 5 dB")]
    cols = [("crn_n", "CRN-n", "tab:blue"), ("crn_ns", "CRN-ns", "tab:orange")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True)
    for r, (group, cname, rlabel) in enumerate(rows):
        d, es = _example(group, cname, runs_dir, configs_dir, device, testsets_h5, rirs_npz, [c[0] for c in cols])
        t = np.arange(len(d)) / 16000
        lim = 1.05 * np.abs(d).max()
        for c, (src, clabel, color) in enumerate(cols):
            ax = axes[r, c]
            if src not in es:
                ax.set_axis_off(); continue
            ax.plot(t, d, color="black", lw=0.5, label="primary d(t)")
            ax.plot(t, es[src], color=color, lw=0.5, alpha=0.85, label="residual e(t)")
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{clabel} — {rlabel} (NMSE {nmse_db(es[src], d):.1f} dB)")
            ax.set_ylabel("Amplitude"); ax.legend(loc="upper right", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Primary vs. residual waveforms (engine noise, eta2 = 0.1)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_waveforms.png", dpi=150)
    plt.close(fig)
    print("wrote fig_waveforms.png")


def _val_curves(run_dir: Path) -> dict:
    """{tag: {epoch: value}} from a run's tensorboard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for ev in sorted(run_dir.glob("tb/*")):
        ea = EventAccumulator(str(ev)); ea.Reload()
        for tag in ("nmse_db/val", "loss/val"):
            if tag in ea.Tags()["scalars"]:
                for s in ea.Scalars(tag):
                    out.setdefault(tag, {})[s.step] = s.value
    return out


def fig_training(runs_dir, out_dir, old_runs_dir=Path("runs_pre_fix")):
    """Validation NMSE vs. epoch for every retrained model (solid) against
    its pre-fix run (dashed); the epoch best.pt kept is starred. CRN-n and
    CRN-ns models get separate panels: CRN-ns keeps the speech, so its NMSE
    against d(t) sits near -0.4 dB by design and needs its own y-scale."""
    models = sorted(p.name for p in runs_dir.iterdir() if (p / "tb").is_dir())
    if not models:
        print("skip fig_training: no runs"); return
    groups = [("CRN-n models (target: cancel everything)", [m for m in models if not m.startswith("crn_ns")]),
              ("CRN-ns models (target: keep speech)", [m for m in models if m.startswith("crn_ns")])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), squeeze=False)
    for ax, (title, ms) in zip(axes[0], groups):
        for i, m in enumerate(ms):
            color = f"C{models.index(m)}"
            cur = _val_curves(runs_dir / m)
            new = cur.get("nmse_db/val", {})
            old = _val_curves(old_runs_dir / m).get("nmse_db/val", {}) if (old_runs_dir / m).is_dir() else {}
            if new:
                ep = sorted(new)
                ax.plot(ep, [new[k] for k in ep], "-", color=color, marker="o", ms=3, label=f"{m} (retrained)")
                # best.pt: lowest val NMSE for mode n, lowest val loss for mode ns
                crit = cur.get("loss/val", {}) if m.startswith("crn_ns") else new
                best = min(crit, key=crit.get)
                ax.plot([best], [new[best]], "*", color=color, ms=14, mec="black")
            if old:
                ep = sorted(old)
                ax.plot(ep, [old[k] for k in ep], "--", color=color, alpha=0.6, label=f"{m} (before fix)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation NMSE (dB)")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Training curves: validation NMSE vs. epoch (star = checkpoint kept)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training.png", dpi=150)
    plt.close(fig)
    print("wrote fig_training.png")


def fig_pesq(results_dir, out_dir, include=None, filename="fig_pesq.png"):
    """Grouped bars: narrowband raw PESQ (P.862, the paper's scale) at each
    SNR for Unprocessed, FxLMS, CRN-n and CRN-ns (engine, eta2=0.1)."""
    sources = [("unprocessed", "Unprocessed", "0.55"), ("fxlms", "FxLMS", "tab:orange"),
               ("crn_n", "CRN-n", "tab:blue"), ("crn_ns", "CRN-ns", "tab:green")]
    data = {}
    for key, _, _ in sources:
        path = results_dir / f"pesq_{key}.json"
        if path.exists():
            data[key] = json.loads(path.read_text())
    sources = [src for src in sources if src[0] in data and (include is None or src[0] in include)]
    if not sources:
        print("skip fig_pesq: no results/pesq_*.json"); return
    snrs = [5, 15, 20]
    x = np.arange(len(snrs))
    width = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label, color) in enumerate(sources):
        vals = [data[key].get(f"table2_snr/snr{k}", {}).get("pesq_nb_raw") or np.nan for k in snrs]
        bars = ax.bar(x + (i - (len(sources) - 1) / 2) * width, vals, width, label=label, color=color, edgecolor="black", lw=0.5)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([f"SNR = {k} dB" for k in snrs])
    ax.set_ylabel("PESQ (narrowband, P.862)")
    ax.set_ylim(1.0, 4.0)
    ax.set_title("PESQ vs. SNR (engine noise, eta2 = 0.1)")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(ncols=len(sources), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    print(f"wrote {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--testsets", default="data/testsets.h5")
    ap.add_argument("--rirs", default="data/rirs.npz")
    ap.add_argument("--configs-dir", default="configs/exp")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--figs", nargs="+", default=["7", "8", "9", "10", "waveforms", "training", "pesq"],
                    help="subset of figures to (re)generate")
    ap.add_argument("--pesq-sources", nargs="+", default=None,
                    help="bars in fig_pesq (default: all of unprocessed/fxlms/crn_n/crn_ns found)")
    ap.add_argument("--pesq-out", default="fig_pesq.png")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)
    configs_dir = Path(args.configs_dir)

    todo = set(args.figs)
    if "7" in todo:
        fig7(args.testsets, args.rirs, runs_dir, configs_dir, out_dir, device)
    if "8" in todo:
        fig8(results_dir, args.rirs, out_dir)
    if "9" in todo:
        fig9(results_dir, args.testsets, args.rirs, out_dir)
    if "10" in todo:
        fig10(args.testsets, args.rirs, runs_dir, configs_dir, out_dir, device)
    if "waveforms" in todo:
        fig_waveforms(args.testsets, args.rirs, runs_dir, configs_dir, out_dir, device)
    if "training" in todo:
        fig_training(runs_dir, out_dir)
    if "pesq" in todo:
        fig_pesq(results_dir, out_dir, include=args.pesq_sources, filename=args.pesq_out)


if __name__ == "__main__":
    main()
