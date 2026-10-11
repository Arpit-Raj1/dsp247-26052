"""The full deep-ANC forward simulation: x -> (delay pad) -> STFT -> CRN ->
iSTFT -> loudspeaker nonlinearity -> secondary path -> a(t); and
d(t) = primary_path(x); e(t) = d(t) - a(t). See Fig. 2 / Eq. (1) of the paper.

This module wires together audio.stft, acoustics.loudspeaker,
acoustics.paths and a CRN model into one differentiable forward pass usable
both for training (loss on e) and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepanc.audio.stft import HOP, N_FFT, istft, stft
from deepanc.acoustics.loudspeaker import sef
from deepanc.acoustics.paths import batched_fir_convolve


@dataclass
class SimulateOutput:
    y: torch.Tensor  # [B, L] canceling signal (CRN output waveform)
    a: torch.Tensor  # [B, L] anti-noise (after loudspeaker + secondary path)
    d: torch.Tensor  # [B, L] primary-path signal
    e: torch.Tensor  # [B, L] error signal = d - a


def apply_delay_compensation(spec: torch.Tensor, m_frames: int) -> torch.Tensor:
    """Pad m_frames zero frames at the start along the time axis and drop
    the last m_frames, so the model at frame t sees input up to frame
    t - m_frames removed, i.e. is asked to predict m_frames ahead
    (Section 3.4 / Fig. 5). spec: [..., T, F] complex or real.
    """
    if m_frames <= 0:
        return spec
    *lead, T, F = spec.shape
    pad_shape = (*lead, m_frames, F)
    zeros = torch.zeros(pad_shape, dtype=spec.dtype, device=spec.device)
    shifted = torch.cat([zeros, spec[..., : T - m_frames, :]], dim=-2)
    return shifted


def crn_forward_waveform(model: torch.nn.Module, x: torch.Tensor, m_frames: int = 0) -> torch.Tensor:
    """x: [B, L] -> y: [B, L], running the CRN in the complex-STFT domain.

    The reference STFT is optionally delay-shifted (m_frames) before being
    fed to the model, so the model's output at frame t corresponds to a
    prediction m_frames*hop samples ahead in time relative to x's own
    frame t (see apply_delay_compensation / Fig. 5).
    """
    L = x.shape[-1]
    X = stft(x)  # [B, T, F] complex
    Xin = apply_delay_compensation(X, m_frames)
    Xr = torch.stack([Xin.real, Xin.imag], dim=1)  # [B, 2, T, F]
    Yr = model(Xr)  # [B, 2, T, F]
    Y = torch.complex(Yr[:, 0], Yr[:, 1])
    y = istft(Y, length=L)
    return y


def simulate(
    model: torch.nn.Module,
    x: torch.Tensor,
    primary_rir: torch.Tensor,
    secondary_rir: torch.Tensor,
    eta2: torch.Tensor | float,
    m_frames: int = 0,
    primary_fn=None,
    secondary_fn=None,
) -> SimulateOutput:
    """Run the full pipeline for a batch.

    x: [B, L] reference signal.
    primary_rir, secondary_rir: [B, K] per-item FIR filters (ignored if
        primary_fn/secondary_fn are given).
    eta2: python float, or [B] tensor of per-item SEF strengths.
    m_frames: delay-compensation frames (0, 1, or 2 in the paper).
    primary_fn: optional callable(x) -> d, overriding primary_rir (used for
        the nonlinear/Volterra primary path setup, Eq. 9).
    secondary_fn: optional callable(y_after_loudspeaker) -> a, overriding
        secondary_rir (used for the nonlinear secondary path, Eqs. 10-11,
        which replaces the SEF-loudspeaker-then-RIR stage entirely with its
        own tanh nonlinearity + 3-tap FIR).
    """
    y = crn_forward_waveform(model, x, m_frames=m_frames)

    if primary_fn is not None:
        d = primary_fn(x)
    else:
        d = batched_fir_convolve(x, primary_rir)

    if secondary_fn is not None:
        a = secondary_fn(y)
    else:
        if isinstance(eta2, torch.Tensor) and eta2.dim() > 0:
            y_nl = _sef_per_item(y, eta2)
        else:
            y_nl = sef(y, eta2 if not isinstance(eta2, torch.Tensor) else float(eta2.item()))
        a = batched_fir_convolve(y_nl, secondary_rir)

    e = d - a
    return SimulateOutput(y=y, a=a, d=d, e=e)


def _sef_per_item(y: torch.Tensor, eta2: torch.Tensor) -> torch.Tensor:
    """Vectorized SEF with a different eta2 per batch item.

    eta2: [B] tensor; entries equal to float('inf') (or <=0 sentinel handled
    by caller) mean identity for that item.
    """
    import math

    eta2 = eta2.to(y.device, y.dtype)
    is_inf = torch.isinf(eta2)
    eta2_safe = torch.where(is_inf, torch.ones_like(eta2), eta2)
    eta = torch.sqrt(eta2_safe).view(-1, *([1] * (y.dim() - 1)))
    out_nl = eta * math.sqrt(math.pi / 2.0) * torch.erf(y / (eta * math.sqrt(2.0)))
    is_inf_b = is_inf.view(-1, *([1] * (y.dim() - 1))).expand_as(y)
    return torch.where(is_inf_b, y, out_nl)
