"""NMSE metric, Eq. (7): NMSE = 10*log10( sum(e^2) / sum(d^2) )."""

from __future__ import annotations

import numpy as np
import torch


def nmse_db(e, d) -> float:
    """e, d: 1-D arrays/tensors of equal length (error and primary-path
    signals). Returns a python float in dB.
    """
    if isinstance(e, torch.Tensor):
        e = e.detach().cpu().numpy()
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu().numpy()
    e = np.asarray(e, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    num = np.sum(e**2)
    den = np.sum(d**2) + 1e-12
    return float(10.0 * np.log10((num + 1e-12) / den))


def nmse_db_batch(e: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Batched version: e, d: [B, L] -> [B] NMSE values in dB."""
    num = (e.double() ** 2).sum(dim=-1)
    den = (d.double() ** 2).sum(dim=-1) + 1e-12
    return 10.0 * torch.log10((num + 1e-12) / den)
