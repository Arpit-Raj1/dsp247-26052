"""FxLMS baseline (Section 5.1): a linear adaptive filter-x LMS controller.

  y(n) = w^T * x_buf(n)                      (controller output)
  a(n) = s(t) * f_LS{y(n)}                   (true plant: loudspeaker + secondary path)
  e(n) = d(n) - a(n)                          (error)
  x'(n) = shat(t) * x(n)                      (filtered reference, using the
                                                *estimated* secondary path)
  w += mu * e(n) * x'_buf(n) / (||x'_buf||^2 + eps)   (normalized LMS update)

Implemented sample-by-sample with numba for speed (this can't be
vectorized across time since w is updated every sample).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
_SQRT_2 = math.sqrt(2.0)


@njit(cache=True)
def _fxlms_core(x, d, s_true, shat, w_len, mu, eps, loudspeaker_kind, eta):
    N = len(x)
    K_s = len(s_true)
    K_shat = len(shat)

    w = np.zeros(w_len, dtype=np.float64)
    x_buf = np.zeros(w_len, dtype=np.float64)  # x(n), x(n-1), ...
    xf_buf = np.zeros(w_len, dtype=np.float64)  # filtered-x buffer
    s_hist = np.zeros(K_s, dtype=np.float64)  # loudspeaker-output history for secondary path
    xf_hist = np.zeros(K_shat, dtype=np.float64)  # x history for computing filtered-x via shat

    e = np.zeros(N, dtype=np.float64)
    y_out = np.zeros(N, dtype=np.float64)

    for n in range(N):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        y = 0.0
        for k in range(w_len):
            y += w[k] * x_buf[k]
        y_out[n] = y

        # true loudspeaker nonlinearity: exact SEF (Eq. 8), via numba's
        # native math.erf support.
        if loudspeaker_kind == 0:
            y_nl = eta * _SQRT_PI_OVER_2 * math.erf(y / (eta * _SQRT_2)) if eta > 0 else y
        elif loudspeaker_kind == 2:  # Sec. 5.5 N[y] = 3.3*tanh(0.3y), Eq. 10
            y_nl = 3.3 * math.tanh(0.3 * y)
        else:  # linear
            y_nl = y

        s_hist[1:] = s_hist[:-1]
        s_hist[0] = y_nl
        a = 0.0
        for k in range(K_s):
            a += s_true[k] * s_hist[k]

        err = d[n] - a
        e[n] = err

        xf_hist[1:] = xf_hist[:-1]
        xf_hist[0] = x[n]
        xf = 0.0
        for k in range(K_shat):
            xf += shat[k] * xf_hist[k]

        xf_buf[1:] = xf_buf[:-1]
        xf_buf[0] = xf

        norm = eps
        for k in range(w_len):
            norm += xf_buf[k] * xf_buf[k]
        step = mu * err / norm
        for k in range(w_len):
            w[k] += step * xf_buf[k]

    return e, y_out


def run_fxlms(
    x: np.ndarray,
    d: np.ndarray,
    secondary_rir_true: np.ndarray,
    secondary_rir_hat: np.ndarray,
    eta2: float,
    mu: float,
    n_taps: int = 512,
    eps: float = 1e-6,
    plant: str = "sef",
) -> tuple[np.ndarray, np.ndarray]:
    """Run FxLMS over the full signal. Returns (e, y).

    eta2=float('inf') (or None) selects the linear loudspeaker case.
    """
    is_linear = eta2 == float("inf") or eta2 is None
    eta = 0.0 if is_linear else float(np.sqrt(eta2))
    kind = 1 if is_linear else 0
    if plant == "nl":
        kind = 2  # Section 5.5 tanh loudspeaker; secondary_rir_true is l(z)
    e, y = _fxlms_core(
        x.astype(np.float64), d.astype(np.float64),
        secondary_rir_true.astype(np.float64), secondary_rir_hat.astype(np.float64),
        n_taps, mu, eps, kind, eta,
    )
    return e, y
