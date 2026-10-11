"""Training losses, Eqs. (4) and (6) of the paper."""

from __future__ import annotations

import torch


def loss_n(e: torch.Tensor) -> torch.Tensor:
    """Deep ANC trained with noise: L_n = mean_t e(t)^2 (Eq. 4)."""
    return (e**2).mean()


def loss_ns(e: torch.Tensor, p_conv_v: torch.Tensor) -> torch.Tensor:
    """Deep ANC trained with noisy speech: L_ns = mean_t (e(t) - p*v(t))^2
    (Eq. 6), where p_conv_v is the speech component of the primary signal,
    p(t) * v(t) (the target ideal error signal: noise cancelled, speech
    passed through).
    """
    return ((e - p_conv_v) ** 2).mean()
