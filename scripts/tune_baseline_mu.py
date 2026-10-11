#!/usr/bin/env python
"""Pick per-noise step sizes for the adaptive baselines (evaluate.py's
DEFAULT_MU). The paper tunes its step sizes per noise type "heuristically
... to ensure stable updating and good noise attenuation" (Section 4.3);
its numbers are for an unnormalized FxLMS and don't carry over to this
repo's NLMS-normalized implementations, so we grid-search instead.

Tuning uses fresh noise cuts (seeds disjoint from build_testsets.py), not
the test signals themselves. fxlms/thf_fxlms are scored by mean NMSE over
the Table 1 conditions (eta2 in {inf, 0.5, 0.1}); vfxlms/fxmlp over the
Section 5.5 nonlinear setup.

Usage:
  python scripts/tune_baseline_mu.py --n-signals 8 --jobs 32
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np

from deepanc.acoustics.rir import load_rir_sets
from deepanc.data.noise import NoiseShard, write_noise_shard
from deepanc.data.testsets import build_noise_only_condition
from deepanc.evaluate import evaluate_baseline

NOISES = ["engine", "factory", "babble", "ssn"]
MU_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
SETUPS = {
    "fxlms": [0.1, 0.5, float("inf")],
    "thf_fxlms": [0.1, 0.5],  # THF-FxLMS is a nonlinear method; the paper reports no eta2=inf column
    "vfxlms": ["nl"],
    "fxmlp": ["nl"],
}
_CTX: dict = {}


def _init(noisex_h5, rirs_npz, n_signals):
    import os

    import torch

    torch.set_num_threads(1)
    rset = load_rir_sets(rirs_npz)["test"]
    shards = {}
    with h5py.File(noisex_h5, "r") as nf:
        for name in NOISES:
            path = f"/dev/shm/_tune_{name}_{os.getpid()}.h5"  # per worker: no concurrent writes
            write_noise_shard(path, [nf[name][:]])
            shards[name] = NoiseShard(path=path).open()
    _CTX.update(rset=rset, shards=shards, n_signals=n_signals)


def _score(job):
    baseline, noise, setup, mu = job
    eta2 = float("inf") if setup == "nl" else setup
    cond = build_noise_only_condition(
        "tune", _CTX["shards"][noise], eta2, 1, seed=90_000 + NOISES.index(noise), n_signals=_CTX["n_signals"]
    )
    if setup == "nl":
        cond.meta["setup"] = "nl"
    res = evaluate_baseline(baseline, cond, _CTX["rset"].primary, _CTX["rset"].secondary, mu=mu)
    nmse = [r["nmse_db"] for r in res]
    return job, float(np.mean(nmse)) if np.all(np.isfinite(nmse)) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisex-h5", default="data/noisex.h5")
    ap.add_argument("--rirs", default="data/rirs.npz")
    ap.add_argument("--n-signals", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--baselines", nargs="+", default=list(SETUPS))
    ap.add_argument("--out", default="results/baseline_mu_tuning.json")
    args = ap.parse_args()

    jobs = [(b, n, s, mu) for b in args.baselines for n, s, mu in itertools.product(NOISES, SETUPS[b], MU_GRID)]
    scores: dict = {}
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.noisex_h5, args.rirs, args.n_signals)) as ex:
        for (b, n, s, mu), v in ex.map(_score, jobs):
            scores.setdefault(b, {}).setdefault(n, {}).setdefault(str(mu), {})[str(s)] = v
            print(f"{b:10s} {n:8s} eta2/setup={s!s:4s} mu={mu:<6} NMSE={v:8.2f}", flush=True)

    best = {}
    for b, per_noise in scores.items():
        for n, per_mu in per_noise.items():
            mean = {mu: float(np.mean(list(v.values()))) for mu, v in per_mu.items()}
            mu_best = min(mean, key=mean.get)
            best.setdefault(b, {})[n] = float(mu_best)
            print(f"best {b} {n}: mu={mu_best} (mean NMSE {mean[mu_best]:.2f} dB)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"best": best, "scores": scores}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
