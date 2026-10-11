import numpy as np

from deepanc.baselines.fxmlp import run_fxmlp
from deepanc.baselines.thf_fxlms import fit_beta, run_thf_fxlms
from deepanc.baselines.vfxlms import n_volterra_weights, run_vfxlms
from deepanc.metrics.nmse import nmse_db


def test_n_volterra_weights_memory_10():
    assert n_volterra_weights(10) == 10 + 55


def test_vfxlms_converges_linear():
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs * 3).astype(np.float64) * 0.1
    rir = np.zeros(32)
    rir[3] = 1.0
    d = np.convolve(x, rir)[: len(x)]
    e, _ = run_vfxlms(x, d, secondary_rir_true=rir, secondary_rir_hat=rir, eta2=float("inf"), mu=0.05, memory=10)
    assert nmse_db(e[-fs:], d[-fs:]) < -20.0


def test_fit_beta_reasonable():
    b1 = fit_beta(0.1)
    b10 = fit_beta(10.0)
    assert 0 < b1 < b10  # larger eta2 (weaker saturation) -> larger beta


def test_thf_fxlms_converges_nonlinear():
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs * 4).astype(np.float64) * 0.1
    rir = np.zeros(64)
    rir[5] = 1.0
    d = np.convolve(x, rir)[: len(x)]
    e, _ = run_thf_fxlms(x, d, secondary_rir_true=rir, secondary_rir_hat=rir, eta2=0.1, mu=0.05, n_taps=64)
    assert nmse_db(e[-fs:], d[-fs:]) < -15.0


def test_fxmlp_reduces_error():
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs * 6).astype(np.float64) * 0.1
    rir = np.zeros(32)
    rir[3] = 1.0
    d = np.convolve(x, rir)[: len(x)]
    e, _ = run_fxmlp(x, d, secondary_rir_true=rir, secondary_rir_hat=rir, eta2=float("inf"), mu=0.03)
    assert nmse_db(e[-fs:], d[-fs:]) < -10.0


def test_vfxlms_nl_plant_uses_tanh_loudspeaker():
    # Section 5.5 plant: a = l(z) * 3.3*tanh(0.3*y). With a zero-weight
    # controller the error is just d, whatever the plant; after adapting,
    # the tanh plant must still be cancellable at small amplitude.
    rng = np.random.default_rng(0)
    fs = 16000
    x = rng.standard_normal(fs * 3) * 0.01
    l = np.array([1.0, 0.2, 0.05])
    d = np.convolve(x, [0.99, 0.2, 0.05])[: len(x)]
    e, _ = run_vfxlms(x, d, secondary_rir_true=l, secondary_rir_hat=l * 0.99, eta2=float("inf"), mu=0.1, plant="nl")
    assert nmse_db(e[-fs:], d[-fs:]) < -15.0
