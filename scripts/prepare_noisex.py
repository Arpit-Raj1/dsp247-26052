#!/usr/bin/env python
"""Download and prepare NOISEX-92 test noises + SSN into noisex.h5.

Source: the SPIB (UFSC) mirror of NOISEX-92, which serves each noise track
as a .mat file of int16 samples at 19.98 kHz. We resample to 16 kHz to
match the pipeline's working rate.

The paper's four test noise types (Table 1/2) are: engine, factory,
babble, SSN. We map:
  engine  -> destroyerengine.mat
  factory -> factory1.mat
  babble  -> babble.mat
  SSN     -> synthesized from the TIMIT test-speaker LTAS (see data/ssn.py;
             the free NOISEX mirror has no SSN track, see module docstring)

We also fetch leopard, m109, and machinegun for later SIH26052 use
(impulsive/vehicle defence noises), not used by the paper reproduction
itself.

Usage:
  python scripts/prepare_noisex.py --out data/noisex.h5 --timit-h5 data/timit.h5
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np
import scipy.io as sio

from deepanc.audio.io import resample
from deepanc.data.ssn import compute_ltas, generate_ssn

SPIB_BASE = "https://spib.linse.ufsc.br/data/noise/"
SPIB_SR = 19980  # per SPIB documentation

NOISE_MAP = {
    "engine": "destroyerengine.mat",
    "factory": "factory1.mat",
    "babble": "babble.mat",
}
EXTRA_FILES = {
    "leopard": "leopard.mat",
    "m109": "m109.mat",
    "machinegun": "machinegun.mat",
}


def fetch_mat(name: str, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / name
    if not local.exists():
        url = SPIB_BASE + name
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, local)
    mat = sio.loadmat(local)
    key = [k for k in mat if not k.startswith("__")][0]
    raw = mat[key].reshape(-1)
    arr = raw.astype(np.float32)
    # Most SPIB tracks are int16, but babble.mat is float64 already in
    # [-1, 1]; dividing that by 32768 left babble ~90dB too quiet.
    if np.issubdtype(raw.dtype, np.integer):
        arr = arr / 32768.0  # int16 -> [-1, 1]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noisex.h5")
    ap.add_argument("--timit-h5", default="data/timit.h5")
    ap.add_argument("--cache-dir", default="data/_cache/noisex")
    ap.add_argument("--skip-extra", action="store_true", help="skip leopard/m109/machinegun")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = {}
    for name, fname in NOISE_MAP.items():
        arr = fetch_mat(fname, cache_dir)
        tracks[name] = resample(arr, SPIB_SR, 16000)

    if not args.skip_extra:
        for name, fname in EXTRA_FILES.items():
            arr = fetch_mat(fname, cache_dir)
            tracks[name] = resample(arr, SPIB_SR, 16000)

    # SSN: LTAS from TIMIT test-speaker utterances
    ssn_len_s = 60
    with h5py.File(args.timit_h5, "r") as hf:
        speech_sigs = []
        for spk in list(hf["train"].keys())[:50]:
            for k in hf["train"][spk]["test"]:
                speech_sigs.append(hf["train"][spk]["test"][k][:])
    ltas = compute_ltas(speech_sigs, n_fft=1024)
    tracks["ssn"] = generate_ssn(ltas, 16000 * ssn_len_s, seed=0)

    with h5py.File(out_path, "w") as hf:
        hf.attrs["sample_rate"] = 16000
        for name, arr in tracks.items():
            hf.create_dataset(name, data=arr.astype(np.float32), compression="gzip", compression_opts=4)
            print(f"{name}: {len(arr)/16000:.1f}s")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
