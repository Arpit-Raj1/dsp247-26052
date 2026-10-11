#!/usr/bin/env python
"""Pack the TIMIT corpus into a single HDF5 file for fast cluster I/O.

Builds the 200-speaker training manifest and 20-speaker untrained-speaker
manifest (Section 4.2 / Table 4), reads every selected utterance via the
SPHERE reader, and writes:

  /train/<speaker_id>/train/<idx>   float32 waveform, attrs: gender
  /train/<speaker_id>/test/<idx>    float32 waveform
  /untrained/<speaker_id>/<idx>     float32 waveform, attrs: gender

Usage:
  python scripts/prepare_timit.py --timit-root TIMIT --out data/timit.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
from tqdm import tqdm

from deepanc.audio.io import read_sphere_wav
from deepanc.data.timit import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timit-root", default="TIMIT")
    ap.add_argument("--out", default="data/timit.h5")
    ap.add_argument("--n-train-speakers", type=int, default=200)
    ap.add_argument("--n-test-utts-per-speaker", type=int, default=3)
    ap.add_argument("--n-untrained-speakers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_speakers, untrained_speakers = build_manifest(
        args.timit_root,
        n_train_speakers=args.n_train_speakers,
        n_test_utts_per_speaker=args.n_test_utts_per_speaker,
        n_untrained_speakers=args.n_untrained_speakers,
        seed=args.seed,
    )

    with h5py.File(out_path, "w") as hf:
        hf.attrs["sample_rate"] = 16000
        hf.attrs["n_train_speakers"] = len(train_speakers)
        hf.attrs["n_untrained_speakers"] = len(untrained_speakers)

        g_train = hf.create_group("train")
        for spk in tqdm(train_speakers, desc="train speakers"):
            spk_g = g_train.create_group(spk.speaker_id)
            spk_g.attrs["gender"] = spk.gender
            tr_g = spk_g.create_group("train")
            for i, utt_path in enumerate(spk.train_utts):
                audio, sr = read_sphere_wav(utt_path)
                assert sr == 16000, (utt_path, sr)
                tr_g.create_dataset(str(i), data=audio, compression="gzip", compression_opts=4)
            te_g = spk_g.create_group("test")
            for i, utt_path in enumerate(spk.test_utts):
                audio, sr = read_sphere_wav(utt_path)
                te_g.create_dataset(str(i), data=audio, compression="gzip", compression_opts=4)

        g_untrained = hf.create_group("untrained")
        for spk in tqdm(untrained_speakers, desc="untrained speakers"):
            spk_g = g_untrained.create_group(spk.speaker_id)
            spk_g.attrs["gender"] = spk.gender
            for i, utt_path in enumerate(spk.test_utts):
                audio, sr = read_sphere_wav(utt_path)
                spk_g.create_dataset(str(i), data=audio, compression="gzip", compression_opts=4)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
