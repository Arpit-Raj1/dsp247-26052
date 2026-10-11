import os

import numpy as np
import pytest

from deepanc.data.noise import NoiseShard
from deepanc.data.testsets import (
    SIG_LEN,
    build_noise_only_condition,
    build_noisy_speech_condition,
    build_robustness_condition,
    build_untrained_t60_condition,
)

NOISE_H5 = "data/noise_train.h5"
skip_if_missing = pytest.mark.skipif(not os.path.exists(NOISE_H5), reason="requires data/noise_train.h5")


@pytest.fixture
def shard():
    s = NoiseShard(path=NOISE_H5).open()
    yield s
    s.close()


@skip_if_missing
def test_noise_only_condition_shape(shard):
    cond = build_noise_only_condition("test", shard, eta2=0.1, n_rirs=5, seed=0, n_signals=10)
    assert cond.x.shape == (10, SIG_LEN)
    assert cond.speech.shape == (10, SIG_LEN)
    assert np.all(cond.speech == 0)
    assert cond.rir_idx.max() < 5
    assert np.all(cond.eta2 == 0.1)


@skip_if_missing
def test_noise_only_condition_deterministic(shard):
    c1 = build_noise_only_condition("test", shard, eta2=0.1, n_rirs=5, seed=42, n_signals=5)
    c2 = build_noise_only_condition("test", shard, eta2=0.1, n_rirs=5, seed=42, n_signals=5)
    assert np.allclose(c1.x, c2.x)
    assert np.array_equal(c1.rir_idx, c2.rir_idx)


@skip_if_missing
def test_noisy_speech_condition_snr(shard):
    from deepanc.data.mixing import measured_snr_db

    rng = np.random.default_rng(0)
    speech = [rng.standard_normal(SIG_LEN).astype(np.float32) * 0.1 for _ in range(3)]
    cond = build_noisy_speech_condition("test", shard, speech, snr_db=10.0, eta2=1.0, n_rirs=1, seed=0, n_signals=3)
    assert cond.x.shape == (3, SIG_LEN)
    assert cond.speech.shape == (3, SIG_LEN)
    for i in range(3):
        implied_noise = cond.x[i] - cond.speech[i]
        got_snr = measured_snr_db(cond.speech[i], implied_noise)
        assert abs(got_snr - 10.0) < 0.5


@skip_if_missing
def test_untrained_t60_condition_fixed_rir(shard):
    cond = build_untrained_t60_condition("test", shard, eta2=0.1, t60_index=3, seed=0, n_signals=5)
    assert np.all(cond.rir_idx == 3)


@skip_if_missing
def test_robustness_snr_change_shape(shard):
    cond = build_robustness_condition(
        "snr_change", "snr_change", {"engine": shard}, None, eta2=0.1, n_rirs=1, seed=0, n_signals=4
    )
    assert cond.x.shape == (4, SIG_LEN)
    assert cond.meta["kind"] == "snr_change"


@skip_if_missing
def test_robustness_noise_change_uses_both_shards(shard):
    cond = build_robustness_condition(
        "noise_change", "noise_change", {"engine": shard, "factory": shard}, None,
        eta2=0.1, n_rirs=1, seed=0, n_signals=2, change_at_s=1.0,
    )
    assert cond.x.shape == (2, SIG_LEN)


@skip_if_missing
def test_robustness_untrained_speaker(shard):
    rng = np.random.default_rng(0)
    speech = [rng.standard_normal(SIG_LEN).astype(np.float32) * 0.1 for _ in range(3)]
    cond = build_robustness_condition(
        "untrained_speaker", "untrained_speaker", {"engine": shard}, speech,
        eta2=0.1, n_rirs=1, seed=0, n_signals=3,
    )
    assert cond.x.shape == (3, SIG_LEN)
    assert not np.all(cond.speech == 0)


@skip_if_missing
def test_noise_only_condition_matches_training_levels(shard):
    from deepanc.data.mixing import rms
    from deepanc.data.testsets import noise_only_rms
    from deepanc.data.datasets import TRAIN_SNRS_DB

    allowed = [noise_only_rms(s) for s in TRAIN_SNRS_DB]
    cond = build_noise_only_condition("test", shard, eta2=0.1, n_rirs=1, seed=0, n_signals=6)
    for x in cond.x:
        assert min(abs(rms(x) - a) / a for a in allowed) < 1e-3


@skip_if_missing
def test_robustness_noisy_speech_variant(shard):
    rng = np.random.default_rng(0)
    speech = [rng.standard_normal(SIG_LEN).astype(np.float32) * 0.1 for _ in range(2)]
    cond = build_robustness_condition(
        "noise_mixture_ns", "noise_mixture", {"engine": shard, "factory": shard}, speech,
        eta2=0.1, n_rirs=1, seed=0, n_signals=2, with_speech=True,
    )
    assert not np.all(cond.speech == 0)
    from deepanc.data.mixing import measured_snr_db

    assert abs(measured_snr_db(cond.speech[0], cond.x[0] - cond.speech[0]) - 5.0) < 0.5
