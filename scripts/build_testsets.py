#!/usr/bin/env python
"""Build every fixed test set used by the tables/figures (Section 7 of the
plan) into a single data/testsets.h5, using the noise/speech/RIR data
already prepared by prepare_timit.py / prepare_noise.py / prepare_noisex.py
/ gen_rirs.py.

Groups written (see data/testsets.py for the TestCondition schema):
  table1/<noise>_<eta2tag>            - engine/factory/babble/ssn x {inf,0.5,0.1}
  table2_noise/<noise>                - engine, eta2=0.1, noise-only
  table2_snr/<snr>                    - engine, eta2=0.1, noisy speech @ 5/15/20dB
  fig7/<noise>_<eta2tag>               - engine, eta2=0.1 (reuses table1 engine_eta0.1)
  fig8/t60_<idx>                       - engine, eta2=0.1, each untrained T60
  fig9/<noise>_5db                     - factory/babble/ssn @ 5dB, eta2=0.1
  quiet_zone/<r>_<noise>, <r>_snr<k>  - r in {r0,r2,r5}: 4 noises noise-only,
                                          and engine noisy speech @ 5/15/20dB
  table4/<case>, <case>_ns             - snr_change, noise_change, noise_mixture
                                          (noise-only / noisy speech), plus
                                          untrained_speaker (noisy speech)
  table5/<noise>                       - 4 noises, Section 5.5 nonlinear setup

Every noise-only reference is scaled to the training level distribution
(see data/testsets.py _level_like_training). Each condition carries a
`noise` attr so evaluate.py can pick per-noise baseline step sizes.

Usage:
  python scripts/build_testsets.py --out data/testsets.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np

from deepanc.acoustics.rir import load_rir_sets
from deepanc.data.noise import NoiseShard
from deepanc.data.testsets import (
    SIG_LEN,
    _write_condition,
    build_noise_only_condition,
    build_noisy_speech_condition,
    build_robustness_condition,
    build_untrained_t60_condition,
)

ETA2_TAGS = {"inf": float("inf"), "0.5": 0.5, "0.1": 0.1}


def _seed(*parts: str) -> int:
    """Stable per-condition seed. (Python's hash() of a str is salted per
    process, so the old hash((noise, tag)) seeds changed on every rebuild.)"""
    import zlib

    return zlib.crc32("/".join(parts).encode()) % (2**31)


def load_speech_signals(timit_h5: str, n: int, untrained: bool = False) -> list:
    with h5py.File(timit_h5, "r") as hf:
        group = hf["untrained"] if untrained else hf["train"]
        speakers = sorted(group.keys())
        sigs = []
        rng = np.random.default_rng(0 if not untrained else 1)
        for _ in range(n):
            spk = speakers[int(rng.integers(0, len(speakers)))]
            if untrained:
                utts = list(group[spk].keys())
                k = utts[int(rng.integers(0, len(utts)))]
                sigs.append(group[spk][k][:])
            else:
                utts = list(group[spk]["test"].keys())
                k = utts[int(rng.integers(0, len(utts)))]
                sigs.append(group[spk]["test"][k][:])
        return sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/testsets.h5")
    ap.add_argument("--noisex-h5", default="data/noisex.h5")
    ap.add_argument("--timit-h5", default="data/timit.h5")
    ap.add_argument("--rirs-npz", default="data/rirs.npz")
    ap.add_argument("--n-signals", type=int, default=100)
    args = ap.parse_args()

    rir_sets = load_rir_sets(args.rirs_npz)
    n_test_rirs = rir_sets["test"].primary.shape[0]  # always 1 (T60=0.2s)
    n_qz_rirs = rir_sets["quiet_zone_train"].primary.shape[0]

    with h5py.File(args.noisex_h5, "r") as nf:
        noise_names = ["engine", "factory", "babble", "ssn"]
        shards = {}
        for name in noise_names:
            arr = nf[name][:]
            tmp_path = f"/dev/shm/_noisex_{name}.h5"
            from deepanc.data.noise import write_noise_shard

            # wrap the single long track as one giant "clip" so NoiseShard's
            # random_cut can slice 6s windows from it
            write_noise_shard(tmp_path, [arr])
            shards[name] = NoiseShard(path=tmp_path).open()

    speech_signals = load_speech_signals(args.timit_h5, args.n_signals, untrained=False)
    untrained_speech_signals = load_speech_signals(args.timit_h5, args.n_signals, untrained=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(group, cond, rir_set="test", **meta):
        cond.meta["rir_set"] = rir_set
        cond.meta.update(meta)
        _write_condition(group, cond)

    with h5py.File(out_path, "w") as out:
        g1 = out.create_group("table1")
        for noise in noise_names:
            for tag, eta2 in ETA2_TAGS.items():
                cond = build_noise_only_condition(
                    f"{noise}_eta{tag}", shards[noise], eta2, n_test_rirs, seed=_seed(noise, tag),
                    n_signals=args.n_signals,
                )
                write(g1, cond, noise=noise)
        print("table1 done")

        g2n = out.create_group("table2_noise")
        cond = build_noise_only_condition("engine", shards["engine"], 0.1, n_test_rirs, seed=100, n_signals=args.n_signals)
        write(g2n, cond, noise="engine")

        g2s = out.create_group("table2_snr")
        for snr in [5, 15, 20]:
            cond = build_noisy_speech_condition(
                f"snr{snr}", shards["engine"], speech_signals, snr, 0.1, n_test_rirs, seed=200 + snr,
                n_signals=args.n_signals,
            )
            write(g2s, cond, noise="engine", snr_db=snr)
        print("table2 done")

        # Fig. 8: rir_idx for each condition indexes the untrained-T60 RIR
        # set directly (one condition per T60), not the default "test" set.
        g8 = out.create_group("fig8")
        n_untrained_t60 = rir_sets["test_untrained_t60"].primary.shape[0]
        for idx in range(n_untrained_t60):
            cond = build_untrained_t60_condition(f"t60_{idx}", shards["engine"], 0.1, idx, seed=300 + idx, n_signals=args.n_signals)
            write(g8, cond, rir_set="test_untrained_t60", noise="engine")
        print("fig8 done")

        g9 = out.create_group("fig9")
        for noise in ["factory", "babble", "ssn"]:
            cond = build_noisy_speech_condition(
                f"{noise}_5db", shards[noise], speech_signals, 5.0, 0.1, n_test_rirs, seed=_seed(noise, "fig9"),
                n_signals=args.n_signals,
            )
            write(g9, cond, noise=noise, snr_db=5.0)
        print("fig9 done")

        # Quiet zone (Table 3): each radius uses its own dedicated RIR set
        # (built by gen_rirs.py at r=0, r=2cm, and 10 points over
        # 0<=r<=5cm) rather than the 100-point *training* sphere sample.
        # CRN-n columns: 4 noises, noise only; CRN-ns columns: engine
        # noisy speech at 5/15/20dB.
        gqz = out.create_group("quiet_zone")
        for ri, radius in enumerate(["r0", "r2", "r5"]):
            rset_name = f"quiet_zone_test_{radius}"
            n_r = rir_sets[rset_name].primary.shape[0]
            for ni, noise in enumerate(noise_names):
                cond = build_noise_only_condition(
                    f"{radius}_{noise}", shards[noise], 0.1, n_r, seed=500 + 10 * ri + ni, n_signals=args.n_signals
                )
                write(gqz, cond, rir_set=rset_name, noise=noise)
            for snr in [5, 15, 20]:
                cond = build_noisy_speech_condition(
                    f"{radius}_snr{snr}", shards["engine"], speech_signals, snr, 0.1, n_r,
                    seed=550 + 10 * ri + snr, n_signals=args.n_signals,
                )
                write(gqz, cond, rir_set=rset_name, noise="engine", snr_db=snr)
        print("quiet_zone done")

        # Table 4: each case has a noise-only and a noisy-speech column
        # ("<case>" / "<case>_ns"); untrained_speaker is noisy speech only.
        g4 = out.create_group("table4")
        for ci, case in enumerate(["snr_change", "noise_change", "noise_mixture"]):
            cond = build_robustness_condition(
                case, case, shards, None, 0.1, n_test_rirs, seed=600 + ci, n_signals=args.n_signals
            )
            write(g4, cond, noise="engine")
            cond = build_robustness_condition(
                f"{case}_ns", case, shards, speech_signals, 0.1, n_test_rirs, seed=610 + ci,
                n_signals=args.n_signals, with_speech=True,
            )
            write(g4, cond, noise="engine")
        cond = build_robustness_condition(
            "untrained_speaker", "untrained_speaker", shards, untrained_speech_signals, 0.1, n_test_rirs,
            seed=603, n_signals=args.n_signals,
        )
        write(g4, cond, noise="engine")
        print("table4 done")

        # Table 5 (Section 5.5): the nonlinear ANC setup — Volterra primary
        # path (Eq. 9) and tanh + 3-tap FIR secondary path (Eqs. 10-11)
        # replace the RIRs and SEF loudspeaker entirely; evaluate.py
        # switches paths on setup="nl". eta2/rir_idx are unused there.
        g5 = out.create_group("table5")
        for ni, noise in enumerate(noise_names):
            cond = build_noise_only_condition(noise, shards[noise], float("inf"), 1, seed=700 + ni, n_signals=args.n_signals)
            write(g5, cond, noise=noise, setup="nl")
        print("table5 done")

    for shard in shards.values():
        shard.close()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
