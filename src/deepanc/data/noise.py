"""Training noise shard access: a registry of HDF5-backed noise sources
plus a uniform random-cut interface used by the training datasets.

The paper trains on 10,000 non-speech clips from a commercial library
(sound-ideas.com); we substitute freely-available noise (DNS-Challenge
noise set by default, MUSAN as a fallback — see scripts/prepare_noise.py).
This module doesn't care which source built the shard file, only that it
exposes a flat list of 1-D float32 waveforms under group "clips".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class NoiseShard:
    path: Path
    _file: h5py.File | None = None
    _keys: list[str] | None = None

    def open(self) -> "NoiseShard":
        self._file = h5py.File(self.path, "r")
        self._keys = sorted(self._file["clips"].keys(), key=int)
        return self

    def __len__(self) -> int:
        assert self._keys is not None, "call .open() first"
        return len(self._keys)

    def get(self, idx: int) -> np.ndarray:
        assert self._file is not None, "call .open() first"
        return self._file["clips"][self._keys[idx]][:]

    def random_cut(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """Return a `length`-sample cut from a random clip. If the chosen
        clip is shorter than `length`, it is looped (tiled) to fill it —
        this only happens for unusually short source clips.
        """
        idx = int(rng.integers(0, len(self)))
        clip = self.get(idx)
        if len(clip) < length:
            reps = int(np.ceil(length / len(clip)))
            clip = np.tile(clip, reps)
        start = int(rng.integers(0, len(clip) - length + 1))
        return clip[start : start + length].astype(np.float32)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def write_noise_shard(path: str | Path, clips: list[np.ndarray], sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.attrs["sample_rate"] = sample_rate
        hf.attrs["n_clips"] = len(clips)
        grp = hf.create_group("clips")
        for i, clip in enumerate(clips):
            grp.create_dataset(str(i), data=clip.astype(np.float32), compression="gzip", compression_opts=4)
