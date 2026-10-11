"""Fixed-size training datasets, Section 4.2: "20,000 training signals and
100 test signals are generated". Each item is deterministically generated
from `seed = base_seed + index`, so the training set is identical across
epochs (matching the paper's "fixed" 20k signal set) unless
`resample_each_epoch=True` (a config knob for later SIH work, where fresh
on-the-fly mixing may be preferred).

Two dataset classes, mirroring Fig. 4 / Eqs. (4)-(6):
  ANCNoiseDataset:       x(t) = n(t)                         (CRN-n)
  ANCNoisySpeechDataset: x(t) = v(t) + n(t), at a random SNR  (CRN-ns)

Each item returns a dict of numpy arrays (collated by the default
DataLoader): x (reference), speech (v, zeros for noise-only), rir_idx
(index into the RirSet used by the training step), eta2 (loudspeaker
nonlinearity strength for this item).

Random RIR selection and eta2 selection are done here (cheap, CPU-only);
everything depending on the network's output y (primary/secondary path
convolution, loudspeaker nonlinearity, loss) happens in the training step
on GPU (anc.simulate), per the plan's CPU/GPU split.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from deepanc.data.mixing import mix_at_snr, scale_to_rms
from deepanc.data.noise import NoiseShard

SIGNAL_SECONDS = 6
SR = 16000
SIGNAL_LEN = SIGNAL_SECONDS * SR

TRAIN_SNRS_DB = (5, 10, 15, 20)
ETA2_CHOICES = (0.1, 1.0, 10.0, float("inf"))

# Reference speech RMS, measured over TIMIT test utterances (see
# scripts/prepare_timit.py output); used only to calibrate the level of
# noise-only training items so their amplitude matches the noise
# component's amplitude in noisy-speech mixtures (Section 3.4 level rule,
# not specified by the paper — see CLAUDE.md "Known paper ambiguities").
REFERENCE_SPEECH_RMS = 0.0125


@dataclass
class NoiseSourceConfig:
    path: str
    group: str = "clips"


class _NoiseIndex:
    """Thin wrapper around NoiseShard that opens its HDF5 handle lazily on
    first access, since h5py.File isn't fork/picklable-safe across
    DataLoader workers — each worker process must open its own handle.
    """

    def __init__(self, path: str):
        self._shard = NoiseShard(path=path)
        self._opened = False

    def _ensure_open(self):
        if not self._opened:
            self._shard.open()
            self._opened = True

    def random_cut(self, length: int, rng: np.random.Generator) -> np.ndarray:
        self._ensure_open()
        return self._shard.random_cut(length, rng)


class _SpeechIndex:
    """Lazy per-worker HDF5 handle over the TIMIT training-speaker pool
    (data/timit.h5's "/train/<speaker>/train/<idx>" datasets).
    """

    def __init__(self, path: str):
        self.path = path
        self._file: h5py.File | None = None
        self._speakers: list[str] | None = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._speakers = sorted(self._file["train"].keys())

    def random_signal(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """Concatenate randomly chosen utterances from a randomly chosen
        training speaker until `length` samples are filled (Section 3.2
        assumption: a 6s signal may span several utterances/speech
        segments, as in Fig. 10).
        """
        self._ensure_open()
        spk = self._speakers[int(rng.integers(0, len(self._speakers)))]
        utt_group = self._file["train"][spk]["train"]
        utt_keys = list(utt_group.keys())
        out = np.zeros(0, dtype=np.float32)
        # small silence gap between concatenated utterances
        gap = np.zeros(int(0.2 * SR), dtype=np.float32)
        while len(out) < length:
            k = utt_keys[int(rng.integers(0, len(utt_keys)))]
            utt = utt_group[k][:]
            out = np.concatenate([out, utt, gap]) if len(out) else utt.copy()
        return out[:length].astype(np.float32)


class ANCNoiseDataset(Dataset):
    """CRN-n training data: reference x(t) = noise n(t). Target error is
    e(t) = 0 (Eq. 4); the loss itself is computed in the training step.
    """

    def __init__(
        self,
        noise_path: str,
        n_items: int = 20000,
        base_seed: int = 0,
        resample_each_epoch: bool = False,
        eta2_choices=ETA2_CHOICES,
        n_rirs: int = 5,
    ):
        self.noise = _NoiseIndex(noise_path)
        self.n_items = n_items
        self.base_seed = base_seed
        self.resample_each_epoch = resample_each_epoch
        self.eta2_choices = eta2_choices
        self.n_rirs = n_rirs
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n_items

    def _seed_for(self, idx: int) -> int:
        if self.resample_each_epoch:
            return self.base_seed + idx * 100003 + self._epoch
        return self.base_seed + idx

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self._seed_for(idx))
        x = self.noise.random_cut(SIGNAL_LEN, rng)
        # Match the level distribution of the noise component in noisy-
        # speech mixtures: draw an SNR the same way ANCNoisySpeechDataset
        # does, and scale this noise-only reference to the RMS that noise
        # would have at that SNR against REFERENCE_SPEECH_RMS.
        snr_db = float(rng.choice(TRAIN_SNRS_DB))
        target_rms = REFERENCE_SPEECH_RMS / (10 ** (snr_db / 20.0))
        x = scale_to_rms(x, target_rms)
        eta2 = float(rng.choice(self.eta2_choices))
        rir_idx = int(rng.integers(0, self.n_rirs))
        return {
            "x": torch.from_numpy(x),
            "speech": torch.zeros(SIGNAL_LEN, dtype=torch.float32),
            "eta2": torch.tensor(eta2, dtype=torch.float32),
            "rir_idx": torch.tensor(rir_idx, dtype=torch.long),
        }


class ANCNoisySpeechDataset(Dataset):
    """CRN-ns training data: reference x(t) = v(t) + n(t), a mixture of
    TIMIT speech and noise at a random SNR (Eq. 5-6). Also returns the
    clean speech v(t) so the training step can form the target
    p(t) * v(t) after convolving with the primary path.
    """

    def __init__(
        self,
        noise_path: str,
        timit_path: str,
        n_items: int = 20000,
        base_seed: int = 0,
        resample_each_epoch: bool = False,
        snr_choices_db=TRAIN_SNRS_DB,
        eta2_choices=ETA2_CHOICES,
        n_rirs: int = 5,
    ):
        self.noise = _NoiseIndex(noise_path)
        self.speech = _SpeechIndex(timit_path)
        self.n_items = n_items
        self.base_seed = base_seed
        self.resample_each_epoch = resample_each_epoch
        self.snr_choices_db = snr_choices_db
        self.eta2_choices = eta2_choices
        self.n_rirs = n_rirs
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n_items

    def _seed_for(self, idx: int) -> int:
        if self.resample_each_epoch:
            return self.base_seed + idx * 100003 + self._epoch
        return self.base_seed + idx

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self._seed_for(idx))
        v = self.speech.random_signal(SIGNAL_LEN, rng)
        n = self.noise.random_cut(SIGNAL_LEN, rng)
        snr_db = float(rng.choice(self.snr_choices_db))
        x, n_scaled = mix_at_snr(v, n, snr_db)
        eta2 = float(rng.choice(self.eta2_choices))
        rir_idx = int(rng.integers(0, self.n_rirs))
        return {
            "x": torch.from_numpy(x),
            "speech": torch.from_numpy(v),
            "eta2": torch.tensor(eta2, dtype=torch.float32),
            "rir_idx": torch.tensor(rir_idx, dtype=torch.long),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }
