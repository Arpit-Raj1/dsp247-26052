#!/usr/bin/env python
"""Assemble Tables 1-5 (Section 7 of the plan) from results/*.json, the
per-condition summaries written by `deepanc.evaluate`.

Each results file is named `<model_or_baseline>_<group>.json` and contains
`{condition_name: {"nmse_db":..., "stoi":..., "pesq_wb":..., "n":...}}`.
This script does not run any model — it only reads what evaluate.py (run
locally or via slurm/eval_array.sbatch + baselines_array.sbatch) already
wrote to `results/`.

"Unprocessed" rows (Table 2, Fig. 9's reference row) aren't produced by
evaluate.py — there's no controller to evaluate — so they're computed here
directly from data/testsets.h5 + data/rirs.npz: d(t) = primary_path(x),
NMSE=0 by definition, and STOI/PESQ compare d against the speech reference
p*v (i.e. how intelligible the *un-cancelled* signal already is).

Usage:
  python scripts/make_tables.py --results-dir results --testsets data/testsets.h5 \
      --rirs data/rirs.npz --out-dir results/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from deepanc.acoustics.rir import load_rir_sets
from deepanc.data.testsets import load_condition
from deepanc.metrics.stoi_pesq import pesq_score, stoi_score

MODEL_LABELS = {
    "crn_n": "CRN-n", "crn_n_m1": "CRN-n(-1)", "crn_n_m2": "CRN-n(-2)",
    "crn_ns": "CRN-ns", "crn_ns_m1": "CRN-ns(-1)", "crn_ns_m2": "CRN-ns(-2)",
    "crn_n_qz": "CRN-n (quiet zone)", "crn_ns_qz": "CRN-ns (quiet zone)",
    "crn_n_nl": "CRN-n (nonlinear)",
    "fxlms": "FxLMS", "thf_fxlms": "THF-FxLMS", "vfxlms": "VFxLMS", "fxmlp": "FxMLP",
}


def load_results(results_dir: Path) -> dict:
    """results[model][group] = {condition: summary}"""
    out = {}
    for path in sorted(results_dir.glob("*.json")):
        # filenames are "<model>_<group>.json"; groups are known, so match
        # the longest known group suffix.
        stem = path.stem
        for group in ["table1", "table2_noise", "table2_snr", "fig8", "fig9", "quiet_zone", "table4", "table5"]:
            suffix = f"_{group}"
            if stem.endswith(suffix):
                model = stem[: -len(suffix)]
                with open(path) as f:
                    out.setdefault(model, {})[group] = json.load(f)
                break
    return out


def unprocessed_row(testsets_h5: str, rirs_npz: str, group: str, condition: str) -> dict | None:
    import h5py

    rir_sets = load_rir_sets(rirs_npz)
    with h5py.File(testsets_h5, "r") as tf:
        cond = load_condition(tf[group], condition)
    rset = rir_sets[cond.meta.get("rir_set", "test")]

    nmses, stois, pesqs = [], [], []
    for i in range(cond.x.shape[0]):
        primary = rset.primary[cond.rir_idx[i]]
        d = np.convolve(cond.x[i], primary)[: len(cond.x[i])]
        nmses.append(0.0)  # e == d for "no ANC", so NMSE = 10*log10(sum(d^2)/sum(d^2)) = 0
        speech = cond.speech[i]
        if np.any(speech != 0):
            ref = np.convolve(speech, primary)[: len(speech)]
            stois.append(stoi_score(ref, d))
            try:
                pesqs.append(pesq_score(ref, d, mode="wb"))
            except Exception:
                pass
    out = {"nmse_db": float(np.mean(nmses)), "n": len(nmses)}
    if stois:
        out["stoi"] = float(np.mean(stois))
    if pesqs:
        out["pesq_wb"] = float(np.mean(pesqs))
    return out


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def table1(results: dict, out_dir: Path):
    noises = ["engine", "factory", "babble", "ssn"]
    etas = ["etainf", "eta0.5", "eta0.1"]
    eta_labels = ["inf", "0.5", "0.1"]
    models = ["fxlms", "thf_fxlms", "crn_n", "crn_n_m1", "crn_n_m2"]

    headers = ["Model"] + [f"{n} ({e})" for n in noises for e in eta_labels]
    rows = []
    for m in models:
        row = [MODEL_LABELS.get(m, m)]
        for n in noises:
            for e in etas:
                v = results.get(m, {}).get("table1", {}).get(f"{n}_{e}", {}).get("nmse_db")
                row.append(fmt(v))
        rows.append(row)

    text = "# Table 1 — Average NMSE (dB)\n\n" + md_table(headers, rows) + "\n"
    (out_dir / "table1.md").write_text(text)
    print(text)


def table2(results: dict, testsets_h5: str, rirs_npz: str, out_dir: Path):
    models = ["fxlms", "thf_fxlms", "crn_n", "crn_ns", "crn_ns_m1", "crn_ns_m2"]
    headers = ["Model", "NoiseOnly NMSE", "SNR5 STOI", "SNR5 PESQ", "SNR5 NMSE",
               "SNR15 STOI", "SNR15 PESQ", "SNR15 NMSE", "SNR20 STOI", "SNR20 PESQ", "SNR20 NMSE"]
    rows = []

    unproc_noise = unprocessed_row(testsets_h5, rirs_npz, "table2_noise", "engine")
    unproc_row = ["Unprocessed", fmt(unproc_noise["nmse_db"]) if unproc_noise else "-"]
    for snr in [5, 15, 20]:
        u = unprocessed_row(testsets_h5, rirs_npz, "table2_snr", f"snr{snr}")
        unproc_row += [fmt(u.get("stoi")), fmt(u.get("pesq_wb")), fmt(u.get("nmse_db"))]
    rows.append(unproc_row)

    for m in models:
        r2n = results.get(m, {}).get("table2_noise", {}).get("engine", {})
        row = [MODEL_LABELS.get(m, m), fmt(r2n.get("nmse_db"))]
        for snr in [5, 15, 20]:
            c = results.get(m, {}).get("table2_snr", {}).get(f"snr{snr}", {})
            row += [fmt(c.get("stoi")), fmt(c.get("pesq_wb")), fmt(c.get("nmse_db"))]
        rows.append(row)

    text = "# Table 2 — NMSE/STOI/PESQ, engine noise, eta2=0.1\n\n" + md_table(headers, rows) + "\n"
    (out_dir / "table2.md").write_text(text)
    print(text)


def table3_quiet_zone(results: dict, testsets_h5: str, rirs_npz: str, out_dir: Path):
    """Paper layout: rows = radius, CRN-n (quiet-zone model) NMSE on the
    4 noises, CRN-ns (quiet-zone model) STOI/PESQ on engine noisy speech."""
    noises = ["engine", "factory", "babble", "ssn"]
    snrs = [5, 15, 20]
    headers = (["Radius"] + [f"CRN-n {n} NMSE" for n in noises]
               + [f"CRN-ns SNR{k} {m}" for k in snrs for m in ("STOI", "PESQ")])

    unproc = ["Unprocessed"] + ["0.00"] * len(noises)
    for k in snrs:
        u = unprocessed_row(testsets_h5, rirs_npz, "quiet_zone", f"r0_snr{k}") or {}
        unproc += [fmt(u.get("stoi")), fmt(u.get("pesq_wb"))]
    rows = [unproc]

    qn = results.get("crn_n_qz", {}).get("quiet_zone", {})
    qs = results.get("crn_ns_qz", {}).get("quiet_zone", {})
    for r, label in [("r0", "r=0"), ("r2", "r=2cm"), ("r5", "0<=r<=5cm")]:
        row = [label] + [fmt(qn.get(f"{r}_{n}", {}).get("nmse_db")) for n in noises]
        for k in snrs:
            c = qs.get(f"{r}_snr{k}", {})
            row += [fmt(c.get("stoi")), fmt(c.get("pesq_wb"))]
        rows.append(row)
    text = "# Table 3 — Quiet zone (5cm radius), eta2=0.1\n\n" + md_table(headers, rows) + "\n"
    (out_dir / "table3.md").write_text(text)
    print(text)


def table4_robustness(results: dict, testsets_h5: str, rirs_npz: str, out_dir: Path):
    """Paper layout: per case, Noise-only NMSE | Noisy-speech STOI, PESQ, NMSE."""
    cases = [("snr_change", "SNR change"), ("noise_change", "Noise change"),
             ("noise_mixture", "Noise mixture"), ("untrained_speaker", "Untrained speakers")]
    headers = ["Case", "Model", "Noise only NMSE", "Noisy speech STOI", "Noisy speech PESQ", "Noisy speech NMSE"]
    rows = []
    for case, label in cases:
        ns_cond = "untrained_speaker" if case == "untrained_speaker" else f"{case}_ns"
        noise_cond = None if case == "untrained_speaker" else case
        u = unprocessed_row(testsets_h5, rirs_npz, "table4", ns_cond) or {}
        rows.append([label, "Unprocessed", "-" if noise_cond is None else "0.00",
                     fmt(u.get("stoi")), fmt(u.get("pesq_wb")), "0.00"])
        for m in ["crn_n", "crn_ns"]:
            t4 = results.get(m, {}).get("table4", {})
            n = t4.get(noise_cond, {}) if noise_cond else {}
            c = t4.get(ns_cond, {})
            rows.append(["", MODEL_LABELS[m], fmt(n.get("nmse_db")) if noise_cond else "-",
                         fmt(c.get("stoi")), fmt(c.get("pesq_wb")), fmt(c.get("nmse_db"))])
    text = "# Table 4 — Robustness, eta2=0.1\n\n" + md_table(headers, rows) + "\n"
    (out_dir / "table4.md").write_text(text)
    print(text)


def table5_nonlinear(results: dict, out_dir: Path):
    models = ["vfxlms", "fxmlp", "crn_n_nl"]
    headers = ["Model", "Engine NMSE", "Factory NMSE", "Babble NMSE", "SSN NMSE"]
    rows = []
    for m in models:
        row = [MODEL_LABELS.get(m, m)]
        t5 = results.get(m, {}).get("table5", {})
        for n in ["engine", "factory", "babble", "ssn"]:
            row.append(fmt(t5.get(n, {}).get("nmse_db")))
        rows.append(row)
    text = "# Table 5 — Nonlinear ANC setup (Volterra primary + tanh/FIR secondary)\n\n" + md_table(headers, rows) + "\n"
    (out_dir / "table5.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--testsets", default="data/testsets.h5")
    ap.add_argument("--rirs", default="data/rirs.npz")
    ap.add_argument("--out-dir", default="results/tables")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    print(f"loaded results for models/baselines: {sorted(results.keys())}")

    table1(results, out_dir)
    table2(results, args.testsets, args.rirs, out_dir)
    table3_quiet_zone(results, args.testsets, args.rirs, out_dir)
    table4_robustness(results, args.testsets, args.rirs, out_dir)
    table5_nonlinear(results, out_dir)

    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
