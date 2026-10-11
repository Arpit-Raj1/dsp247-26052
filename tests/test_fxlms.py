import numpy as np

from deepanc.baselines.fxlms import run_fxlms
from deepanc.baselines.sec_path_id import identify_secondary_path
from deepanc.acoustics.loudspeaker import sef
from deepanc.metrics.nmse import nmse_db


def test_fxlms_converges_linear_tonal_plant():
    fs = 16000
    t = np.arange(fs * 4) / fs
    x = np.sin(2 * np.pi * 300 * t).astype(np.float64)
    rir = np.zeros(64)
    rir[5] = 1.0
    d = np.convolve(x, rir)[: len(x)]

    e, _ = run_fxlms(x, d, secondary_rir_true=rir, secondary_rir_hat=rir, eta2=float("inf"), mu=0.05, n_taps=64)
    steady_state_nmse = nmse_db(e[-fs:], d[-fs:])
    assert steady_state_nmse < -20.0, steady_state_nmse


def test_fxlms_reduces_broadband_noise():
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs * 3).astype(np.float64) * 0.1
    rir = np.zeros(32)
    rir[3] = 1.0
    d = np.convolve(x, rir)[: len(x)]
    e, _ = run_fxlms(x, d, secondary_rir_true=rir, secondary_rir_hat=rir, eta2=float("inf"), mu=0.1, n_taps=32)
    assert nmse_db(e[-fs:], d[-fs:]) < -10.0


def test_secondary_path_identification_linear():
    rng = np.random.default_rng(0)
    true_rir = rng.standard_normal(64).astype(np.float32) * 0.1
    true_rir[0] = 1.0  # ensure a dominant tap
    identity = lambda y: y  # linear loudspeaker (eta2=inf)
    shat = identify_secondary_path(true_rir, identity, n_taps=64, excitation_len=16000 * 8, seed=1)
    err = np.max(np.abs(shat - true_rir))
    assert err < 0.05, err
