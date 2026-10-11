"""Offline secondary-path identification, Ŝ(z): the step every FxLMS-family
baseline needs before it can run (Section 5.1: "The secondary path is
usually estimated offline during an initial stage in ANC applications").

We excite the true plant (loudspeaker nonlinearity f_LS, then the true
secondary-path RIR s(t)) with white noise at an amplitude matched to
typical canceling-signal levels, and fit a linear FIR Ŝ(z) by least
squares. Under nonlinear loudspeaker saturation this recovers only a
linear approximation of the true (nonlinear) plant — which is exactly the
failure mode Section 5.1 describes for FxLMS under nonlinearity, and why
THF-FxLMS (a nonlinear plant model) does better there.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def identify_secondary_path(
    secondary_rir: np.ndarray,
    loudspeaker_fn,
    n_taps: int = 512,
    excitation_len: int = 16000 * 4,
    excitation_amplitude: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Returns an estimated FIR Ŝ(z) of length n_taps.

    secondary_rir: the true secondary path h_s(t) (loudspeaker -> error mic).
    loudspeaker_fn: callable(np.ndarray) -> np.ndarray, the true loudspeaker
        nonlinearity (e.g. functools.partial(sef, eta2=...)).
    """
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal(excitation_len).astype(np.float64) * excitation_amplitude
    probe_nl = np.asarray(loudspeaker_fn(probe))
    plant_out = lfilter(secondary_rir.astype(np.float64), [1.0], probe_nl)

    # Least-squares FIR fit: build a Toeplitz-style design matrix via
    # sliding windows of the (nonlinear) probe and solve for taps that
    # best predict plant_out (Wiener-Hopf via normal equations, using
    # scipy's fast correlation instead of an explicit dense matrix).
    from scipy.linalg import solve_toeplitz

    autocorr = np.correlate(probe_nl, probe_nl, mode="full")
    mid = len(autocorr) // 2
    r = autocorr[mid : mid + n_taps]
    crosscorr = np.correlate(plant_out, probe_nl, mode="full")
    mid_c = len(crosscorr) // 2
    p = crosscorr[mid_c : mid_c + n_taps]

    taps = solve_toeplitz(r, p)
    return taps.astype(np.float32)
