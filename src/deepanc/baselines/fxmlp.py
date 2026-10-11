"""FxMLP baseline (Section 5.5): a small MLP-based nonlinear ANC controller
(Chang & Luoh, 2007; Zhou, Zhang, Li & Gan, 2005), extending FxLMS-style
online gradient training to a nonlinear controller.

Architecture (as specified in the plan / Section 5.5): 6 tapped-delay
inputs -> 2 hidden layers of 12 sigmoid units each -> 1 linear output.
Trained online by backprop through the estimated secondary path Ŝ(z):
at each sample n, compute y(n) via a forward pass, run it through the true
plant to get e(n), then backprop e(n)^2 through Ŝ and the MLP to update
the MLP's weights by one gradient step (a "filtered-error" nonlinear
extension of FxLMS, implemented in plain numpy since the network is tiny
and this must run causally, sample-by-sample).

Note: FxMLP's forward pass is not causal-parallelizable (the same
sample-by-sample constraint as FxLMS/VFxLMS), but is small enough that a
non-numba, per-sample numpy loop is fast enough for our test-set sizes; a
numba core is used for the sample loop's dot products/updates.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

N_INPUTS = 6
N_HIDDEN = 12


def init_weights(seed: int = 0):
    rng = np.random.default_rng(seed)
    scale1 = 1.0 / math.sqrt(N_INPUTS)
    scale2 = 1.0 / math.sqrt(N_HIDDEN)
    W1 = rng.uniform(-scale1, scale1, size=(N_HIDDEN, N_INPUTS))
    b1 = np.zeros(N_HIDDEN)
    W2 = rng.uniform(-scale2, scale2, size=(N_HIDDEN, N_HIDDEN))
    b2 = np.zeros(N_HIDDEN)
    w3 = rng.uniform(-scale2, scale2, size=N_HIDDEN)
    b3 = 0.0
    return W1, b1, W2, b2, w3, b3


@njit(cache=True)
def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


@njit(cache=True)
def _fxmlp_core(x, d, s_true, shat, W1, b1, W2, b2, w3, b3, mu, loudspeaker_kind, eta, delay, gain):
    """delay, gain: the secondary path Shat is approximated for gradient
    purposes by a single dominant tap (index `delay`, value `gain`) rather
    than a full FIR filtered-gradient (which would require filtering the
    network's per-weight sensitivity, not just a scalar, through the whole
    Shat impulse response). Backprop for sample n therefore uses the
    forward-pass activations from sample n-delay (buffered below), paired
    with the current error e(n) and scaled by `gain` — i.e. a delay-
    compensated filtered-error update. This is the same simplification
    VFxLMS avoids by filtering full basis terms; here it's needed because
    the "basis" is the whole network, which is too costly to filter
    term-by-term through Shat sample-by-sample.
    """
    N = len(x)
    K_s = len(s_true)

    x_buf = np.zeros(N_INPUTS, dtype=np.float64)
    s_hist = np.zeros(K_s, dtype=np.float64)

    # ring buffers of forward-pass activations, so backprop at sample n can
    # use the state from sample n-delay once e(n) becomes available.
    buf_len = delay + 1
    x_ring = np.zeros((buf_len, N_INPUTS), dtype=np.float64)
    h1_ring = np.zeros((buf_len, N_HIDDEN), dtype=np.float64)
    h2_ring = np.zeros((buf_len, N_HIDDEN), dtype=np.float64)

    e = np.zeros(N, dtype=np.float64)
    y_out = np.zeros(N, dtype=np.float64)

    for n in range(N):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = x[n]

        z1 = np.zeros(N_HIDDEN, dtype=np.float64)
        for i in range(N_HIDDEN):
            s = b1[i]
            for j in range(N_INPUTS):
                s += W1[i, j] * x_buf[j]
            z1[i] = s
        h1 = _sigmoid(z1)

        z2 = np.zeros(N_HIDDEN, dtype=np.float64)
        for i in range(N_HIDDEN):
            s = b2[i]
            for j in range(N_HIDDEN):
                s += W2[i, j] * h1[j]
            z2[i] = s
        h2 = _sigmoid(z2)

        y = b3
        for j in range(N_HIDDEN):
            y += w3[j] * h2[j]
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

        # record this sample's forward-pass activations into the ring
        # buffer (slot n % buf_len), then read back the activations from
        # `delay` samples ago to pair with today's error for backprop
        # (delay-compensated filtered-error update; see docstring above).
        slot = n % buf_len
        x_ring[slot, :] = x_buf
        h1_ring[slot, :] = h1
        h2_ring[slot, :] = h2

        if n >= delay:
            read_slot = (n - delay) % buf_len
            x_d = x_ring[read_slot]
            h1_d = h1_ring[read_slot]
            h2_d = h2_ring[read_slot]

            dE_dy = -2.0 * err * gain

            dw3 = dE_dy * h2_d
            db3 = dE_dy
            dh2 = dE_dy * w3

            dz2 = dh2 * h2_d * (1.0 - h2_d)
            dW2 = np.zeros((N_HIDDEN, N_HIDDEN), dtype=np.float64)
            for i in range(N_HIDDEN):
                for j in range(N_HIDDEN):
                    dW2[i, j] = dz2[i] * h1_d[j]
            db2 = dz2

            dh1 = np.zeros(N_HIDDEN, dtype=np.float64)
            for j in range(N_HIDDEN):
                s = 0.0
                for i in range(N_HIDDEN):
                    s += dz2[i] * W2[i, j]
                dh1[j] = s
            dz1 = dh1 * h1_d * (1.0 - h1_d)
            dW1 = np.zeros((N_HIDDEN, N_INPUTS), dtype=np.float64)
            for i in range(N_HIDDEN):
                for j in range(N_INPUTS):
                    dW1[i, j] = dz1[i] * x_d[j]
            db1 = dz1

            w3 -= mu * dw3
            b3 -= mu * db3
            W2 -= mu * dW2
            b2 -= mu * db2
            W1 -= mu * dW1
            b1 -= mu * db1

    return e, y_out, W1, b1, W2, b2, w3, b3


def run_fxmlp(
    x: np.ndarray,
    d: np.ndarray,
    secondary_rir_true: np.ndarray,
    secondary_rir_hat: np.ndarray,
    eta2: float,
    mu: float,
    seed: int = 0,
    plant: str = "sef",
) -> tuple[np.ndarray, np.ndarray]:
    """Run FxMLP over the full signal. Returns (e, y)."""
    is_linear = eta2 == float("inf") or eta2 is None
    eta = 0.0 if is_linear else float(np.sqrt(eta2))
    kind = 1 if is_linear else 0
    if plant == "nl":
        kind = 2  # Section 5.5 tanh loudspeaker; secondary_rir_true is l(z)
    W1, b1, W2, b2, w3, b3 = init_weights(seed)

    delay = int(np.argmax(np.abs(secondary_rir_hat)))
    gain = float(secondary_rir_hat[delay])

    e, y, *_ = _fxmlp_core(
        x.astype(np.float64), d.astype(np.float64),
        secondary_rir_true.astype(np.float64), secondary_rir_hat.astype(np.float64),
        W1, b1, W2, b2, w3, b3, mu, kind, eta, delay, gain,
    )
    return e, y
