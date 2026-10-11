"""Evaluate a trained deep-ANC checkpoint (or a classical baseline) against
one or more fixed test conditions from data/testsets.h5, producing the
per-condition NMSE/STOI/PESQ numbers that feed scripts/make_tables.py and
scripts/make_figures.py (Section 7 of the plan).

Usage (deep model):
  python -m deepanc.evaluate --checkpoint runs/crn_n/best.pt --config configs/exp/crn_n.yaml \
      --testsets data/testsets.h5 --group table1 --out results/crn_n_table1.json

Usage (baseline):
  python -m deepanc.evaluate --baseline fxlms --testsets data/testsets.h5 --group table1 \
      --rirs data/rirs.npz --out results/fxlms_table1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from deepanc.acoustics.loudspeaker import sef, tanh_nl
from deepanc.acoustics.paths import batched_fir_convolve, nl_secondary_path, volterra_primary
from deepanc.acoustics.rir import load_rir_sets
from deepanc.anc.simulate import simulate
from deepanc.config import load_config
from deepanc.baselines.sec_path_id import identify_secondary_path
from deepanc.data.testsets import load_condition
from deepanc.metrics.nmse import nmse_db
from deepanc.metrics.stoi_pesq import pesq_score, stoi_score
from deepanc.models.crn import CRN

BASELINES = {}
try:
    from deepanc.baselines.fxlms import run_fxlms
    from deepanc.baselines.thf_fxlms import run_thf_fxlms
    from deepanc.baselines.vfxlms import run_vfxlms
    from deepanc.baselines.fxmlp import run_fxmlp

    BASELINES = {
        "fxlms": run_fxlms,
        "thf_fxlms": run_thf_fxlms,
        "vfxlms": run_vfxlms,
        "fxmlp": run_fxmlp,
    }
except ImportError:  # pragma: no cover - numba optional on some eval nodes
    pass


# Section 5.5's secondary FIR l(z) (Eq. 11), for conditions with setup="nl".
NL_SECONDARY_FIR = np.array([1.0, 0.2, 0.05])

# Per-noise step sizes. The paper tunes these per noise type ("chosen
# heuristically for different noises ... to ensure stable updating") but its
# values (0.05/0.4/0.3/0.4) are for its own, unnormalized FxLMS; ours are
# NLMS-normalized, so they were re-tuned — see REPRODUCTION.md.
# Values from scripts/tune_baseline_mu.py (results/baseline_mu_tuning.json).
DEFAULT_MU = {
    "fxlms": {"engine": 0.01, "factory": 0.03, "babble": 0.03, "ssn": 0.03},
    "thf_fxlms": {"engine": 0.01, "factory": 0.03, "babble": 0.03, "ssn": 0.03},
    "vfxlms": {"engine": 0.3, "factory": 0.3, "babble": 0.3, "ssn": 0.3},
    "fxmlp": {"engine": 0.1, "factory": 0.1, "babble": 0.1, "ssn": 0.1},
}
FALLBACK_MU = 0.01  # stable for every noise in the sweep


def default_mu(baseline: str, noise: str | None) -> float:
    return DEFAULT_MU.get(baseline, {}).get(noise, FALLBACK_MU)


def is_nl_setup(cond) -> bool:
    return cond.meta.get("setup") == "nl"


def evaluate_model(model, cfg, rir_primary, rir_secondary, cond, device):
    """Run a trained CRN over every signal in a TestCondition, return a
    list of per-signal metrics dicts.
    """
    results = []
    model.eval()
    batch = 8
    N = cond.x.shape[0]
    with torch.no_grad():
        for start in range(0, N, batch):
            xb = torch.from_numpy(cond.x[start : start + batch]).float().to(device)
            rir_idx = cond.rir_idx[start : start + batch]
            primary = rir_primary[rir_idx].to(device)
            secondary = rir_secondary[rir_idx].to(device)
            eta2b = torch.from_numpy(cond.eta2[start : start + batch]).float().to(device)
            nl = is_nl_setup(cond)
            out = simulate(
                model, xb, primary_rir=primary, secondary_rir=secondary, eta2=eta2b, m_frames=cfg.train.m_frames,
                primary_fn=volterra_primary if nl else None, secondary_fn=nl_secondary_path if nl else None,
            )
            e = out.e.cpu().numpy()
            d = out.d.cpu().numpy()
            for i in range(xb.shape[0]):
                idx = start + i
                m = {"nmse_db": nmse_db(e[i], d[i])}
                speech = cond.speech[idx]
                if np.any(speech != 0):
                    ref = np.convolve(speech, rir_primary[rir_idx[i]].numpy())[: len(speech)]
                    m["stoi"] = stoi_score(ref, e[i])
                    try:
                        m["pesq_wb"] = pesq_score(ref, e[i], mode="wb")
                    except Exception:
                        m["pesq_wb"] = None
                results.append(m)
    return results


def evaluate_baseline(name, cond, rir_sets_primary, rir_sets_secondary, mu, n_taps=512, **kwargs):
    fn = BASELINES[name]
    results = []
    N = cond.x.shape[0]
    # identify_secondary_path() does its own ~4s excitation simulation +
    # Wiener-Hopf solve — expensive, and its result depends only on
    # (secondary_rir, eta2), not on which signal we're scoring. A test
    # condition's 100 signals typically share a handful of RIR/eta2
    # combos (e.g. table1 has 1 RIR x 3 eta2 = 3 combos across 1200
    # signals total), so recomputing it per-signal was ~100x more work
    # than necessary and was silently timing out multi-condition groups
    # (table1, fig8, fig9, quiet_zone, table4) past their 6h walltime
    # with zero output. Cache it per unique (secondary_rir, eta2) pair
    # instead.
    shat_cache: dict[tuple[bytes, float], np.ndarray] = {}
    nl = is_nl_setup(cond)
    for i in range(N):
        eta2 = float(cond.eta2[i])
        x = cond.x[i].astype(np.float64)
        if nl:
            # Section 5.5: Volterra primary path, tanh + l(z) secondary path
            primary = None
            secondary = NL_SECONDARY_FIR
            d = volterra_primary(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            loudspeaker_fn = lambda y: tanh_nl(torch.from_numpy(y)).numpy()
        else:
            primary = rir_sets_primary[cond.rir_idx[i]]
            secondary = rir_sets_secondary[cond.rir_idx[i]]
            d = np.convolve(x, primary.astype(np.float64))[: len(x)]
            loudspeaker_fn = (lambda y, eta2=eta2: sef(torch.from_numpy(y), eta2).numpy()) if eta2 != float("inf") else (lambda y: y)

        cache_key = (secondary.tobytes(), eta2)
        shat = shat_cache.get(cache_key)
        if shat is None:
            shat = identify_secondary_path(secondary, loudspeaker_fn, n_taps=min(n_taps, len(secondary)))
            shat_cache[cache_key] = shat

        plant = {"plant": "nl"} if nl else {}
        if name in ("fxlms", "thf_fxlms"):
            if nl and name == "thf_fxlms":
                raise ValueError("thf_fxlms has no Section 5.5 (setup=nl) plant model")
            e, y = fn(x, d, secondary_rir_true=secondary, secondary_rir_hat=shat, eta2=eta2, mu=mu, n_taps=len(shat), **plant)
        else:  # vfxlms, fxmlp
            e, y = fn(x, d, secondary_rir_true=secondary, secondary_rir_hat=shat, eta2=eta2, mu=mu, **plant)

        m = {"nmse_db": nmse_db(e, d)}
        speech = cond.speech[i]
        if np.any(speech != 0):
            ref = np.convolve(speech, primary)[: len(speech)]
            m["stoi"] = stoi_score(ref, e)
            try:
                m["pesq_wb"] = pesq_score(ref, e, mode="wb")
            except Exception:
                m["pesq_wb"] = None
        results.append(m)
    return results


def summarize(results: list[dict]) -> dict:
    keys = results[0].keys() if results else []
    out = {}
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["n"] = len(results)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--config")
    ap.add_argument("--baseline", choices=list(BASELINES.keys()))
    ap.add_argument("--mu", type=float, default=None,
                    help="baseline step size; default: per-noise value from DEFAULT_MU")
    ap.add_argument("--testsets", default="data/testsets.h5")
    ap.add_argument("--rirs", default="data/rirs.npz")
    ap.add_argument("--rir-set", default="test")
    ap.add_argument("--group", required=True, help="top-level group in testsets.h5, e.g. table1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rir_sets = load_rir_sets(args.rirs)

    with h5py.File(args.testsets, "r") as tf:
        cond_names = list(tf[args.group].keys())
        all_summaries = {}
        for name in cond_names:
            cond = load_condition(tf[args.group], name)
            rir_set_name = cond.meta.get("rir_set", args.rir_set)
            rset = rir_sets[rir_set_name]

            if args.checkpoint:
                cfg = load_config(args.config)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = CRN(
                    n_freq=cfg.model.n_freq, channels=list(cfg.model.channels),
                    lstm_layers=cfg.model.lstm_layers, lstm_groups=cfg.model.lstm_groups,
                ).to(device)
                state = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(state["model"])
                rir_primary = torch.from_numpy(rset.primary).float()
                rir_secondary = torch.from_numpy(rset.secondary).float()
                results = evaluate_model(model, cfg, rir_primary, rir_secondary, cond, device)
            else:
                mu = args.mu if args.mu is not None else default_mu(args.baseline, cond.meta.get("noise"))
                results = evaluate_baseline(args.baseline, cond, rset.primary, rset.secondary, mu=mu)

            all_summaries[name] = summarize(results)
            print(name, all_summaries[name])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
