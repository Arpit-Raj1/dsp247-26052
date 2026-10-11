"""Fixed test-set construction for every table/figure in Section 7 of the
plan. Each condition is written once to disk as a set of 100 6-second
signals (Section 4.2: "100 test signals"), so every model and baseline is
evaluated on byte-identical inputs.

A "condition" bundles: the reference/primary signal x(t) plus everything
needed to score it — primary path RIR(s), secondary path RIR, loudspeaker
eta2, and (for noisy-speech conditions) the clean speech v(t) so p*v can
be formed as the STOI/PESQ reference.

Robustness conditions (Table 4: SNR change / noise change / noise mixture
/ untrained speakers mid-signal) store a *time-varying* eta2/noise
description instead of a single fixed value; see build_robustness_set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from deepanc.data.datasets import REFERENCE_SPEECH_RMS, TRAIN_SNRS_DB
from deepanc.data.mixing import mix_at_snr, scale_to_rms
from deepanc.data.noise import NoiseShard

SR = 16000
SIG_LEN = 6 * SR
N_TEST_SIGNALS = 100


@dataclass
class TestCondition:
    name: str
    x: np.ndarray  # [N, L] reference signals
    speech: np.ndarray  # [N, L] clean speech component (zeros if noise-only)
    rir_idx: np.ndarray  # [N] index into the associated RirSet
    eta2: np.ndarray  # [N] loudspeaker nonlinearity strength (inf allowed)
    meta: dict = field(default_factory=dict)


def _write_condition(h5group: h5py.Group, cond: TestCondition) -> None:
    grp = h5group.create_group(cond.name)
    grp.create_dataset("x", data=cond.x.astype(np.float32), compression="gzip")
    grp.create_dataset("speech", data=cond.speech.astype(np.float32), compression="gzip")
    grp.create_dataset("rir_idx", data=cond.rir_idx.astype(np.int64))
    eta2_finite = np.where(np.isinf(cond.eta2), -1.0, cond.eta2).astype(np.float32)
    grp.create_dataset("eta2", data=eta2_finite)  # -1 sentinel for inf
    for k, v in cond.meta.items():
        grp.attrs[k] = v


def load_condition(h5group: h5py.Group, name: str) -> TestCondition:
    grp = h5group[name]
    eta2 = grp["eta2"][:].astype(np.float64)
    eta2 = np.where(eta2 < 0, np.inf, eta2)
    return TestCondition(
        name=name, x=grp["x"][:], speech=grp["speech"][:],
        rir_idx=grp["rir_idx"][:], eta2=eta2, meta=dict(grp.attrs),
    )


def noise_only_rms(snr_db: float) -> float:
    """RMS of a noise-only reference at the level its noise would have in a
    noisy-speech mixture at `snr_db` (ANCNoiseDataset's level rule)."""
    return REFERENCE_SPEECH_RMS / (10 ** (snr_db / 20.0))


def _level_like_training(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale a noise-only test reference to the same level distribution
    ANCNoiseDataset trains on. Raw NOISEX tracks sit 25-80x above that
    (and babble.mat at a different scale again), which put every
    noise-only test far outside the training distribution and made the
    SEF loudspeaker saturate in a way training never exercised.
    """
    return scale_to_rms(x, noise_only_rms(float(rng.choice(TRAIN_SNRS_DB))))


def build_noise_only_condition(
    name: str, noise_shard: NoiseShard, eta2: float, n_rirs: int, seed: int, n_signals: int = N_TEST_SIGNALS
) -> TestCondition:
    rng = np.random.default_rng(seed)
    x = np.stack([_level_like_training(noise_shard.random_cut(SIG_LEN, rng), rng) for _ in range(n_signals)])
    rir_idx = rng.integers(0, n_rirs, size=n_signals)
    eta2_arr = np.full(n_signals, eta2, dtype=np.float64)
    return TestCondition(name=name, x=x, speech=np.zeros_like(x), rir_idx=rir_idx, eta2=eta2_arr)


def build_noisy_speech_condition(
    name: str,
    noise_shard: NoiseShard,
    speech_signals: list[np.ndarray],
    snr_db: float,
    eta2: float,
    n_rirs: int,
    seed: int,
    n_signals: int = N_TEST_SIGNALS,
) -> TestCondition:
    rng = np.random.default_rng(seed)
    xs, vs = [], []
    for i in range(n_signals):
        v = speech_signals[i % len(speech_signals)]
        n = noise_shard.random_cut(SIG_LEN, rng)
        x, _ = mix_at_snr(v[:SIG_LEN] if len(v) >= SIG_LEN else np.pad(v, (0, SIG_LEN - len(v))), n, snr_db)
        xs.append(x)
        vs.append(v[:SIG_LEN] if len(v) >= SIG_LEN else np.pad(v, (0, SIG_LEN - len(v))))
    x = np.stack(xs)
    speech = np.stack(vs)
    rir_idx = rng.integers(0, n_rirs, size=n_signals)
    eta2_arr = np.full(n_signals, eta2, dtype=np.float64)
    return TestCondition(name=name, x=x, speech=speech, rir_idx=rir_idx, eta2=eta2_arr)


def build_untrained_t60_condition(
    name: str, noise_shard: NoiseShard, eta2: float, t60_index: int, seed: int, n_signals: int = N_TEST_SIGNALS
) -> TestCondition:
    """Fig. 8: same noise-only construction but rir_idx always points at a
    single (untrained) T60's RIR pair (t60_index into the
    test_untrained_t60 RirSet).
    """
    rng = np.random.default_rng(seed)
    x = np.stack([_level_like_training(noise_shard.random_cut(SIG_LEN, rng), rng) for _ in range(n_signals)])
    rir_idx = np.full(n_signals, t60_index, dtype=np.int64)
    eta2_arr = np.full(n_signals, eta2, dtype=np.float64)
    return TestCondition(name=name, x=x, speech=np.zeros_like(x), rir_idx=rir_idx, eta2=eta2_arr)


def build_robustness_condition(
    name: str,
    kind: str,
    noise_shards: dict[str, NoiseShard],
    speech_signals: list[np.ndarray] | None,
    eta2: float,
    n_rirs: int,
    seed: int,
    n_signals: int = N_TEST_SIGNALS,
    change_at_s: float = 3.0,
    with_speech: bool = False,
    snr_db: float = 5.0,
) -> TestCondition:
    """Table 4 robustness conditions. `kind` in:
      "snr_change"      - engine noise, level 5dB->20dB (SNR-equivalent)
                           at change_at_s
      "noise_change"     - engine->factory at change_at_s
      "noise_mixture"    - engine+factory simultaneously
      "untrained_speaker"- engine + speech from untrained speakers at
                            `snr_db` (always noisy speech)
    with_speech=False builds the noise-only column of Table 4, with levels
    following ANCNoiseDataset's rule (noise RMS = REFERENCE_SPEECH_RMS at
    the given SNR). with_speech=True builds the noisy-speech column: the
    same noise construction mixed with `speech_signals` at `snr_db` (for
    snr_change, the SNR itself steps from 5dB to 20dB).
    noise_shards: dict with keys among {"engine", "factory"} as needed.
    """
    rng = np.random.default_rng(seed)
    change_at = int(change_at_s * SR)
    xs, vs = [], []

    def fit(v: np.ndarray) -> np.ndarray:
        return v[:SIG_LEN] if len(v) >= SIG_LEN else np.pad(v, (0, SIG_LEN - len(v)))

    for i in range(n_signals):
        v = fit(speech_signals[i % len(speech_signals)]) if (with_speech or kind == "untrained_speaker") else None
        if kind == "untrained_speaker":
            n = noise_shards["engine"].random_cut(SIG_LEN, rng)
            x, _ = mix_at_snr(v, n, snr_db)
        elif kind == "snr_change":
            n = noise_shards["engine"].random_cut(SIG_LEN, rng)
            if v is None:
                x = n.copy()
                x[:change_at] = scale_to_rms(n[:change_at], noise_only_rms(5.0))
                x[change_at:] = scale_to_rms(n[change_at:], noise_only_rms(20.0))
            else:
                _, n5 = mix_at_snr(v[:change_at], n[:change_at], 5.0)
                _, n20 = mix_at_snr(v[change_at:], n[change_at:], 20.0)
                x = v + np.concatenate([n5, n20])
        elif kind in ("noise_change", "noise_mixture"):
            n1 = noise_shards["engine"].random_cut(SIG_LEN, rng)
            n2 = noise_shards["factory"].random_cut(SIG_LEN, rng)
            # equalize the two noises' RMS so neither dominates just
            # because of its NOISEX recording level
            n2 = scale_to_rms(n2, float(np.sqrt(np.mean(n1.astype(np.float64) ** 2))))
            if kind == "noise_change":
                n = n1.copy()
                n[change_at:] = n2[change_at:]
            else:
                n = n1 + n2
            if v is None:
                x = _level_like_training(n, rng)
            else:
                x, _ = mix_at_snr(v, n, snr_db)
        else:
            raise ValueError(kind)
        xs.append(np.asarray(x, dtype=np.float32))
        vs.append(v if v is not None else np.zeros(SIG_LEN, dtype=np.float32))

    x = np.stack(xs)
    speech = np.stack(vs).astype(np.float32)
    rir_idx = rng.integers(0, n_rirs, size=n_signals)
    eta2_arr = np.full(n_signals, eta2, dtype=np.float64)
    return TestCondition(name=name, x=x, speech=speech, rir_idx=rir_idx, eta2=eta2_arr, meta={"kind": kind})
