#!/usr/bin/env python
"""PESQ for trained models on every noisy-speech test condition, following
the conventions of Loizou's pesq.m (03_kaggle/pesq.m):

  pesq_nb_raw  - narrowband raw P.862 (8 kHz), range ~-0.5..4.5. The
                 paper's Table 2 values (e.g. Unprocessed 1.95 @ 5dB)
                 fall on this scale, so this is the one to compare.
  pesq_nb_mos  - narrowband MOS-LQO (P.862.1 mapping of the above)
  pesq_wb      - wideband MOS-LQO (P.862.2, 16 kHz) — what evaluate.py
                 has been reporting as `pesq_wb`

Reference = p*v (clean speech through the primary path), degraded = e(t),
as in evaluate.py; "unprocessed" scores d(t) itself (no ANC).

Sources can be trained models (runs/<name>/best.pt) or adaptive baselines
(fxlms, thf_fxlms, run with evaluate.py's per-noise DEFAULT_MU).

Usage:
  python scripts/eval_pesq.py --models crn_n crn_ns fxlms --jobs 32
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np
import torch

from deepanc.acoustics.rir import load_rir_sets
from deepanc.anc.simulate import simulate
from deepanc.config import load_config
from deepanc.data.testsets import load_condition
from deepanc.metrics.stoi_pesq import pesq_nb_scores, pesq_score
from deepanc.models.crn import CRN

# (group, condition) pairs that contain speech
CONDITIONS = [
    ("table2_snr", "snr5"), ("table2_snr", "snr15"), ("table2_snr", "snr20"),
    ("fig9", "factory_5db"), ("fig9", "babble_5db"), ("fig9", "ssn_5db"),
    ("table4", "snr_change_ns"), ("table4", "noise_change_ns"),
    ("table4", "noise_mixture_ns"), ("table4", "untrained_speaker"),
]
# Paper Table 2 PESQ (engine, eta2=0.1), for side-by-side reference
PAPER = {
    "unprocessed": {"snr5": 1.95, "snr15": 2.61, "snr20": 2.96},
    "crn_n": {"snr5": 1.71, "snr15": 2.02, "snr20": 2.10},
    "crn_ns": {"snr5": 2.26, "snr15": 3.00, "snr20": 3.32},
    "fxlms": {"snr5": 1.84, "snr15": 1.90, "snr20": 1.98},
    "thf_fxlms": {"snr5": 1.73, "snr15": 1.92, "snr20": 2.12},
}
SOURCE_ORDER = ["unprocessed", "fxlms", "thf_fxlms", "crn_n", "crn_ns"]


def _score(args):
    ref, deg = args
    out = {}
    try:
        out["pesq_nb_raw"], out["pesq_nb_mos"] = pesq_nb_scores(ref, deg)
    except Exception:
        out["pesq_nb_raw"] = out["pesq_nb_mos"] = None
    try:
        out["pesq_wb"] = pesq_score(ref, deg, mode="wb")
    except Exception:
        out["pesq_wb"] = None
    return out


def run_model(name, cond, rset, runs_dir, configs_dir):
    """Returns (refs, degs) for every signal of `cond`."""
    primary = rset.primary[cond.rir_idx]
    refs = np.stack([np.convolve(v, p)[: len(v)] for v, p in zip(cond.speech, primary)])
    if name == "unprocessed":
        degs = np.stack([np.convolve(x, p)[: len(x)] for x, p in zip(cond.x, primary)])
        return refs, degs
    if name in ("fxlms", "thf_fxlms"):
        from deepanc.evaluate import BASELINES, default_mu
        from deepanc.acoustics.loudspeaker import sef
        from deepanc.baselines.sec_path_id import identify_secondary_path

        mu = default_mu(name, cond.meta.get("noise"))
        degs, shat_cache = [], {}
        for i in range(len(cond.x)):
            x = cond.x[i].astype(np.float64)
            eta2 = float(cond.eta2[i])
            sec = rset.secondary[cond.rir_idx[i]]
            d = np.convolve(x, primary[i].astype(np.float64))[: len(x)]
            key = (sec.tobytes(), eta2)
            if key not in shat_cache:
                ls = (lambda y, eta2=eta2: sef(torch.from_numpy(y), eta2).numpy()) if eta2 != float("inf") else (lambda y: y)
                shat_cache[key] = identify_secondary_path(sec, ls, n_taps=min(512, len(sec)))
            shat = shat_cache[key]
            e, _ = BASELINES[name](x, d, secondary_rir_true=sec, secondary_rir_hat=shat, eta2=eta2, mu=mu, n_taps=len(shat))
            degs.append(e)
        return refs, np.stack(degs)
    cfg = load_config(configs_dir / f"{name}.yaml")
    model = CRN(n_freq=cfg.model.n_freq, channels=list(cfg.model.channels),
                lstm_layers=cfg.model.lstm_layers, lstm_groups=cfg.model.lstm_groups)
    model.load_state_dict(torch.load(runs_dir / name / "best.pt", map_location="cpu")["model"])
    model.eval()
    degs = []
    with torch.no_grad():
        for s in range(0, len(cond.x), 8):
            sl = slice(s, s + 8)
            out = simulate(
                model, torch.from_numpy(cond.x[sl]).float(),
                primary_rir=torch.from_numpy(primary[sl]).float(),
                secondary_rir=torch.from_numpy(rset.secondary[cond.rir_idx[sl]]).float(),
                eta2=torch.from_numpy(cond.eta2[sl]).float(), m_frames=cfg.train.m_frames,
            )
            degs.append(out.e.numpy())
    return refs, np.concatenate(degs)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["crn_n"])
    ap.add_argument("--testsets", default="data/testsets.h5")
    ap.add_argument("--rirs", default="data/rirs.npz")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--configs-dir", default="configs/exp")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    # sanity check: identical signals must score at the top of each scale
    with h5py.File(args.testsets, "r") as tf:
        v = load_condition(tf["table2_snr"], "snr5").speech[0]
    print("sanity (ref vs itself):", _score((v, v)), flush=True)

    rir_sets = load_rir_sets(args.rirs)
    sources = args.models if (Path(args.out_dir) / "pesq_unprocessed.json").exists() else ["unprocessed"] + args.models
    results = {s: {} for s in sources}
    with ProcessPoolExecutor(args.jobs) as ex, h5py.File(args.testsets, "r") as tf:
        for group, cname in CONDITIONS:
            cond = load_condition(tf[group], cname)
            rset = rir_sets[cond.meta.get("rir_set", "test")]
            for src in sources:
                refs, degs = run_model(src, cond, rset, Path(args.runs_dir), Path(args.configs_dir))
                per = list(ex.map(_score, zip(refs, degs), chunksize=4))
                summ = {k: mean([p[k] for p in per]) for k in ("pesq_nb_raw", "pesq_nb_mos", "pesq_wb")}
                summ["n"] = len(per)
                summ["n_failed_nb"] = sum(p["pesq_nb_raw"] is None for p in per)
                results[src][f"{group}/{cname}"] = summ
                print(f"{src:12s} {group}/{cname:18s} " + " ".join(
                    f"{k}={summ[k]:.3f}" if summ[k] is not None else f"{k}=NA"
                    for k in ("pesq_nb_raw", "pesq_nb_mos", "pesq_wb")), flush=True)

    out_dir = Path(args.out_dir)
    for src in sources:
        with open(out_dir / f"pesq_{src}.json", "w") as f:
            json.dump(results[src], f, indent=2)

    # rebuild the summary table from every pesq_<source>.json on disk, so
    # running one new source doesn't drop the others
    alls = {}
    for path in out_dir.glob("pesq_*.json"):
        alls[path.stem[len("pesq_"):]] = json.loads(path.read_text())
    order = [x for x in SOURCE_ORDER if x in alls] + sorted(x for x in alls if x not in SOURCE_ORDER)
    lines = ["# PESQ (conventions of pesq.m: NB raw P.862 / NB MOS-LQO P.862.1 / WB P.862.2)", "",
             "| Condition | Source | NB raw | NB MOS-LQO | WB MOS-LQO | Paper (NB raw) |", "|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.2f}"
    for group, cname in CONDITIONS:
        for src in order:
            r = alls[src].get(f"{group}/{cname}")
            if r is None:
                continue
            paper = PAPER.get(src, {}).get(cname) if group == "table2_snr" else None
            lines.append(f"| {group}/{cname} | {src} | {f(r['pesq_nb_raw'])} | {f(r['pesq_nb_mos'])} | {f(r['pesq_wb'])} | {f(paper)} |")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables" / "pesq.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
