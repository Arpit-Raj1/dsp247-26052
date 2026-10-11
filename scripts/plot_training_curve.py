#!/usr/bin/env python
"""Training and validation NMSE vs. epoch for one run, parsed from its
train.py log lines ("epoch N: train_loss=... val_loss=... val_nmse_db=...").

With train.loss_norm=d_power and mode n, the loss is mean(e^2)/mean(d^2)
per batch, i.e. the linear NMSE, so training NMSE (dB) = 10*log10(train_loss)
(an average over batches of the ratio, so it can differ slightly from the
per-signal dB average used for validation). Runs trained with raw MSE
(loss_norm=none) have no such mapping, so only their validation curve is
shown.

Usage:
  python scripts/plot_training_curve.py --log slurm_logs/retrain-crn_n-106285.out \
      --name crn_n --out results/figures/fig_training_crn_n.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(r"^epoch (\d+): train_loss=([\d.e+-]+) val_loss=([\d.e+-]+) val_nmse_db=([-\d.]+)")


def parse(log: Path):
    rows = [LINE.match(l) for l in log.read_text().splitlines()]
    rows = [m for m in rows if m]
    ep = np.array([int(m[1]) for m in rows])
    return ep, np.array([float(m[2]) for m in rows]), np.array([float(m[4]) for m in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", default="crn_n")
    ap.add_argument("--normalized-loss", action=argparse.BooleanOptionalAction, default=True,
                    help="train loss is mean(e^2)/mean(d^2) (loss_norm=d_power, mode n)")
    ap.add_argument("--val", action=argparse.BooleanOptionalAction, default=True,
                    help="also plot validation NMSE and the kept checkpoint (--no-val: training only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ep, train_loss, val_nmse = parse(Path(args.log))
    fig, ax = plt.subplots(figsize=(8, 5))
    if args.normalized_loss:
        ax.plot(ep, 10 * np.log10(train_loss), "-o", ms=3, color="tab:blue", label="training NMSE")
    if args.val:
        ax.plot(ep, val_nmse, "-s", ms=3, color="tab:orange", label="validation NMSE")
        best = int(ep[np.argmin(val_nmse)])
        ax.plot([best], [val_nmse.min()], "*", ms=15, color="tab:orange", mec="black",
                label=f"best checkpoint: epoch {best}, {val_nmse.min():.2f} dB")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"{args.name}: NMSE vs. epoch")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
