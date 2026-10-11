"""VFxLMS baseline (Section 5.5): second-order Volterra adaptive filter-x
LMS, following Guo et al. (2018) / Tan & Jiang (2001)'s memory-M Volterra
controller and secondary-path model.

Controller output:
  y(n) = sum_i w1[i]*x(n-i) + sum_{i<=j} w2[i,j]*x(n-i)*x(n-j)
for i,j in [0, M), giving M linear taps and M*(M+1)/2 quadratic taps
(memory M=10 -> 10 linear + 55 quadratic = 65 total weights, matching the
plan's "memory 10 (10 linear + 55 quadratic terms)").

The secondary path is also modeled as a (fixed, pre-identified) second-
order Volterra system in the filtered-reference construction, analogous to
linear FxLMS's Ŝ(z) but with linear+quadratic filtered-reference terms.
For simplicity and stability (matching common VFxLMS implementations) we
filter each Volterra *basis term* of the reference through a single linear
Ŝ(z) (the dominant, linear part of the identified secondary path) rather
than a full second-order filtered-reference expansion; this is a standard
simplification noted in Guo et al. as acceptable when secondary-path
nonlinearity is weak relative to its linear component.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
_SQRT_2 = math.sqrt(2.0)


def n_volterra_weights(memory: int) -> int:
    return memory + memory * (memory + 1) // 2


@njit(cache=True)
def _vfxlms_core(x, d, s_true, shat, memory, n_w, mu, eps, loudspeaker_kind, eta):
    N = len(x)
    K_s = len(s_true)
    K_shat = len(shat)

    w = np.zeros(n_w, dtype=np.float64)
    x_buf = np.zeros(memory, dtype=np.float64)
    basis = np.zeros(n_w, dtype=np.float64)
    filt_basis = np.zeros(n_w, dtype=np.float64)  # each basis term after Shat filtering
    filt_basis_hist = np.zeros((n_w, K_shat), dtype=np.float64)  # per-basis-term filtered-x history
    s_hist = np.zeros(K_s, dtype=np.float64)

    e = np.zeros(N, dtype=np.float64)
    y_out = np.zeros(N, dtype=np.float64)

    for n in range(N):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        # build Volterra basis: [x(n),...,x(n-M+1), x(n)^2, x(n)x(n-1),...]
        idx = 0
        for i in range(memory):
            basis[idx] = x_buf[i]
            idx += 1
        for i in range(memory):
            for j in range(i, memory):
                basis[idx] = x_buf[i] * x_buf[j]
                idx += 1

        y = 0.0
        for k in range(n_w):
            y += w[k] * basis[k]
        y_out[n] = y

        if loudspeaker_kind == 0:
            y_nl = eta * 1.2533141373155003 * math.erf(y / (eta * 1.4142135623730951)) if eta > 0 else y

        elif loudspeaker_kind == 2:  # Sec. 5.5 N[y] = 3.3*tanh(0.3y), Eq. 10

            y_nl = 3.3 * math.tanh(0.3 * y)
        else:
            y_nl = y

        s_hist[1:] = s_hist[:-1]
        s_hist[0] = y_nl
        a = 0.0
        for k in range(K_s):
            a += s_true[k] * s_hist[k]

        err = d[n] - a
        e[n] = err

        norm = eps
        for k in range(n_w):
            filt_basis_hist[k, 1:] = filt_basis_hist[k, :-1]
            filt_basis_hist[k, 0] = basis[k]
            xf = 0.0
            for t in range(K_shat):
                xf += shat[t] * filt_basis_hist[k, t]
            filt_basis[k] = xf
            norm += xf * xf

        for k in range(n_w):
            w[k] += mu * err * filt_basis[k] / norm

    return e, y_out


def run_vfxlms(
    x: np.ndarray,
    d: np.ndarray,
    secondary_rir_true: np.ndarray,
    secondary_rir_hat: np.ndarray,
    eta2: float,
    mu: float,
    memory: int = 10,
    eps: float = 1e-6,
    plant: str = "sef",
) -> tuple[np.ndarray, np.ndarray]:
    """Run VFxLMS over the full signal. Returns (e, y)."""
    is_linear = eta2 == float("inf") or eta2 is None
    eta = 0.0 if is_linear else float(np.sqrt(eta2))
    kind = 1 if is_linear else 0
    if plant == "nl":
        kind = 2  # Section 5.5 tanh loudspeaker; secondary_rir_true is l(z)
    n_w = n_volterra_weights(memory)
    e, y = _vfxlms_core(
        x.astype(np.float64), d.astype(np.float64),
        secondary_rir_true.astype(np.float64), secondary_rir_hat.astype(np.float64),
        memory, n_w, mu, eps, kind, eta,
    )
    return e, y
