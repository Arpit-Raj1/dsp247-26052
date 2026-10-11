"""Loudspeaker nonlinearity models.

Scaled error function (SEF), Eq. (8) of the paper:

    f_SEF(y) = integral_0^y exp(-z^2 / (2*eta^2)) dz
             = eta * sqrt(pi/2) * erf(y / (sqrt(2)*eta))

eta^2 controls the strength of the saturation nonlinearity: eta^2 -> 0 is a
hard limiter, eta^2 -> inf is linear (identity). We treat eta2 == float('inf')
specially and return y unchanged (the closed form above is a 0/0 limit).

Also implements the tanh nonlinearity N[y] = 3.3*tanh(0.3*y) used for the
nonlinear ANC setup in Section 5.5 (Eq. 10).
"""

from __future__ import annotations

import math

import torch


def sef(y: torch.Tensor, eta2: float) -> torch.Tensor:
    """Scaled error function nonlinearity, Eq. (8)."""
    if eta2 == float("inf"):
        return y
    eta = math.sqrt(eta2)
    return eta * math.sqrt(math.pi / 2.0) * torch.erf(y / (eta * math.sqrt(2.0)))


def tanh_nl(y: torch.Tensor, a: float = 3.3, b: float = 0.3) -> torch.Tensor:
    """N[y] = a * tanh(b * y), Eq. (10)."""
    return a * torch.tanh(b * y)
