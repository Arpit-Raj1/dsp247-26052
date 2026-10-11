import numpy as np
import torch

from deepanc.metrics.nmse import nmse_db, nmse_db_batch
from deepanc.metrics.spectrum import short_time_nmse_db, welch_psd_db
from deepanc.metrics.stoi_pesq import pesq_score, stoi_score


def test_nmse_zero_error():
    d = np.random.randn(1000).astype(np.float32)
    e = np.zeros_like(d)
    val = nmse_db(e, d)
    assert val < -100  # essentially -inf, clamped by epsilon


def test_nmse_equal_signals():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(1000).astype(np.float32)
    val = nmse_db(d, d)
    assert abs(val - 0.0) < 1e-6


def test_nmse_batch_matches_scalar():
    rng = np.random.default_rng(0)
    e = torch.from_numpy(rng.standard_normal((4, 500)).astype(np.float32))
    d = torch.from_numpy(rng.standard_normal((4, 500)).astype(np.float32))
    batch_vals = nmse_db_batch(e, d).numpy()
    for i in range(4):
        assert abs(batch_vals[i] - nmse_db(e[i], d[i])) < 1e-4


def test_short_time_nmse_shape():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(16000 * 2).astype(np.float32)
    d = rng.standard_normal(16000 * 2).astype(np.float32)
    vals = short_time_nmse_db(e, d, window_s=0.1)
    assert vals.shape == (20,)


def test_welch_psd_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    freqs, psd_db = welch_psd_db(x, nperseg=1024)
    assert freqs.shape == psd_db.shape
    assert freqs[-1] <= 8000


def test_stoi_identical_signals_near_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float64)
    score = stoi_score(x, x)
    assert score > 0.99


def test_pesq_identical_signals_near_max():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 2) * 0.1).astype(np.float64)
    score = pesq_score(x, x, mode="wb")
    assert score > 4.0
