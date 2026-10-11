"""THF-FxLMS baseline (Ghasemi, Kamil & Marhaban, 2016; Section 5.1):
tangential-hyperbolic-function FxLMS for nonlinear ANC with loudspeaker
saturation.

Where plain FxLMS filters the raw reference x(n) through an estimated
*linear* secondary path Ŝ(z), THF-FxLMS instead models the loudspeaker's
saturation with a fitted tanh function

  ghat(y) = beta * tanh(y / beta)

and filters the reference through the *derivative* of this estimated
nonlinearity evaluated at the controller's own output, before the linear
secondary-path filtering step (the standard "nonlinear FxLMS" filtered-
reference construction: x'(n) = shat * [ghat'(y(n)) * x(n) ]). This lets
the LMS update account for the local slope of the saturation instead of
treating the plant as purely linear.

beta is fit offline so that ghat approximates the true SEF loudspeaker
nonlinearity (Eq. 8) over the operating amplitude range, by least-squares
matching ghat(y) to f_SEF(y) on a grid of y values.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
_SQRT_2 = math.sqrt(2.0)


def fit_beta(eta2: float, y_range: float = 5.0, n_points: int = 2000) -> float:
    """Fit beta in ghat(y) = beta*tanh(y/beta) to minimize squared error
    against the true SEF f_LS(y) over y in [-y_range, y_range].
    """
    eta = math.sqrt(eta2) if eta2 != float("inf") else None
    y = np.linspace(-y_range, y_range, n_points)
    if eta is None:
        target = y
    else:
        from scipy.special import erf

        target = eta * _SQRT_PI_OVER_2 * erf(y / (eta * _SQRT_2))

    def loss(beta):
        beta = max(beta, 1e-6)
        pred = beta * np.tanh(y / beta)
        return float(np.sum((pred - target) ** 2))

    res = minimize_scalar(loss, bounds=(1e-3, 100.0), method="bounded")
    return float(res.x)


@njit(cache=True)
def _thf_fxlms_core(x, d, s_true, shat, w_len, mu, eps, loudspeaker_kind, eta, beta):
    N = len(x)
    K_s = len(s_true)
    K_shat = len(shat)

    w = np.zeros(w_len, dtype=np.float64)
    x_buf = np.zeros(w_len, dtype=np.float64)
    xf_buf = np.zeros(w_len, dtype=np.float64)
    s_hist = np.zeros(K_s, dtype=np.float64)
    filtref_hist = np.zeros(K_shat, dtype=np.float64)  # g'(y(n))*x(n) history

    e = np.zeros(N, dtype=np.float64)
    y_out = np.zeros(N, dtype=np.float64)

    for n in range(N):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        y = 0.0
        for k in range(w_len):
            y += w[k] * x_buf[k]
        y_out[n] = y

        if loudspeaker_kind == 0:
            y_nl = eta * 1.2533141373155003 * math.erf(y / (eta * 1.4142135623730951)) if eta > 0 else y
        else:
            y_nl = y

        s_hist[1:] = s_hist[:-1]
        s_hist[0] = y_nl
        a = 0.0
        for k in range(K_s):
            a += s_true[k] * s_hist[k]

        err = d[n] - a
        e[n] = err

        # g'(y) = 1 - tanh(y/beta)^2  (derivative of beta*tanh(y/beta))
        th = math.tanh(y / beta)
        g_prime = 1.0 - th * th

        filtref_hist[1:] = filtref_hist[:-1]
        filtref_hist[0] = g_prime * x[n]
        xf = 0.0
        for k in range(K_shat):
            xf += shat[k] * filtref_hist[k]

        xf_buf[1:] = xf_buf[:-1]
        xf_buf[0] = xf

        norm = eps
        for k in range(w_len):
            norm += xf_buf[k] * xf_buf[k]
        step = mu * err / norm
        for k in range(w_len):
            w[k] += step * xf_buf[k]

    return e, y_out


def run_thf_fxlms(
    x: np.ndarray,
    d: np.ndarray,
    secondary_rir_true: np.ndarray,
    secondary_rir_hat: np.ndarray,
    eta2: float,
    mu: float,
    n_taps: int = 512,
    eps: float = 1e-6,
    beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run THF-FxLMS over the full signal. Returns (e, y).

    beta: fitted tanh-saturation parameter; if None, fit_beta(eta2) is
    called automatically.
    """
    is_linear = eta2 == float("inf") or eta2 is None
    eta = 0.0 if is_linear else float(np.sqrt(eta2))
    kind = 1 if is_linear else 0
    if beta is None:
        beta = fit_beta(eta2)
    e, y = _thf_fxlms_core(
        x.astype(np.float64), d.astype(np.float64),
        secondary_rir_true.astype(np.float64), secondary_rir_hat.astype(np.float64),
        n_taps, mu, eps, kind, eta, beta,
    )
    return e, y
