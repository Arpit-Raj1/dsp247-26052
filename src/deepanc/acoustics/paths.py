"""Batched per-sample FIR path convolution and Volterra/NL path models.

Each item in a batch is convolved with its own filter (different RIR per
example). This is implemented with a single grouped 1-D convolution rather
than a Python loop over the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batched_fir_convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal 'full' convolution, cropped to len(x), per batch item.

    x: [B, L] signals.
    h: [B, K] filters (h[b] applied to x[b]).
    returns: [B, L] where out[b, t] = sum_k h[b, k] * x[b, t - k] (x padded
    with K-1 zeros on the left, i.e. standard causal FIR filtering).

    Implemented as grouped conv1d with groups=B: reshape x to [1, B, L],
    weight to [B, 1, K] (flipped, since conv1d cross-correlates).
    """
    B, L = x.shape
    Bh, K = h.shape
    assert B == Bh, f"batch mismatch: x has {B}, h has {Bh}"

    x_pad = F.pad(x, (K - 1, 0))  # causal left-pad
    x_in = x_pad.unsqueeze(0)  # [1, B, L+K-1]
    weight = h.flip(-1).unsqueeze(1)  # [B, 1, K] (flip for true convolution)
    out = F.conv1d(x_in, weight, groups=B)  # [1, B, L]
    return out.squeeze(0)


def volterra_primary(x: torch.Tensor) -> torch.Tensor:
    """Primary path Volterra model, Eq. (9):

    d(t) = x(t) + 0.8x(t-1) + 0.3x(t-2) + 0.4x(t-3)
           - 0.8x(t)x(t-1) + 0.9x(t)x(t-2) + 0.7x(t)x(t-3)

    x: [B, L] -> d: [B, L]. x is treated as causally zero-padded before t=0.
    """
    B, L = x.shape
    xp = F.pad(x, (3, 0))  # xp[:, 3:] == x ; xp[:, 3-k] == x(t-k)

    def shift(k: int) -> torch.Tensor:
        return xp[:, 3 - k : 3 - k + L]

    x0, x1, x2, x3 = shift(0), shift(1), shift(2), shift(3)
    linear = x0 + 0.8 * x1 + 0.3 * x2 + 0.4 * x3
    quad = -0.8 * x0 * x1 + 0.9 * x0 * x2 + 0.7 * x0 * x3
    return linear + quad


def nl_secondary_path(y: torch.Tensor) -> torch.Tensor:
    """Nonlinear secondary path for Sec. 5.5: N[y(t)] = 3.3*tanh(0.3*y(t)),
    then l(z) = 1 + 0.2*z^-1 + 0.05*z^-2 applied to N[y] (Eqs. 10-11).

    y: [B, L] -> a: [B, L].
    """
    from .loudspeaker import tanh_nl

    n = tanh_nl(y)
    B, L = n.shape
    h = torch.tensor([1.0, 0.2, 0.05], device=y.device, dtype=y.dtype)
    h = h.unsqueeze(0).expand(B, -1)
    return batched_fir_convolve(n, h)
