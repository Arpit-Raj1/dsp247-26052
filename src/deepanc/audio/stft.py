"""Causal STFT/iSTFT matching the paper's 320-point, 20ms/10ms analysis.

Frame m of the STFT uses only samples t <= m*hop + n_fft - 1 (no centering),
so a stack of these frames is causal in time. Analysis/synthesis use a
periodic sqrt-Hann window so that, with hop = n_fft/2, overlap-add of the
squared windows sums to a constant and perfect reconstruction holds exactly
in the interior of the signal (edges are handled by dividing by the actual,
clamped overlap envelope).
"""

from __future__ import annotations

import torch

N_FFT = 320
HOP = 160
N_FREQ = N_FFT // 2 + 1


def sqrt_hann_window(n_fft: int = N_FFT, device=None, dtype=None) -> torch.Tensor:
    # periodic=True Hann, then sqrt, so that analysis*synthesis is a valid
    # overlap-add window pair (COLA) at hop = n_fft/2.
    w = torch.hann_window(n_fft, periodic=True, device=device, dtype=dtype)
    return torch.sqrt(w)


def stft(x: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP) -> torch.Tensor:
    """x: [..., L] real waveform -> [..., T, F] complex, causal framing.

    T = floor((L - n_fft) / hop) + 1. Frame m spans samples
    [m*hop, m*hop + n_fft).
    """
    *lead, L = x.shape
    assert L >= n_fft, f"signal length {L} shorter than n_fft {n_fft}"
    x2 = x.reshape(-1, L)
    window = sqrt_hann_window(n_fft, device=x.device, dtype=x.dtype)
    # unfold into frames: [B, T, n_fft]
    frames = x2.unfold(-1, n_fft, hop)  # [B, T, n_fft]
    frames = frames * window
    spec = torch.fft.rfft(frames, n=n_fft, dim=-1)  # [B, T, F] complex
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft(spec: torch.Tensor, length: int, n_fft: int = N_FFT, hop: int = HOP) -> torch.Tensor:
    """spec: [..., T, F] complex -> [..., length] real waveform.

    Overlap-add with the sqrt-Hann synthesis window, normalised by the
    (clamped) window-overlap envelope so reconstruction is exact wherever
    the envelope is fully built up (i.e. away from the very first/last
    n_fft - hop samples).
    """
    *lead, T, F = spec.shape
    device, dtype = spec.device, spec.real.dtype
    window = sqrt_hann_window(n_fft, device=device, dtype=dtype)

    spec2 = spec.reshape(-1, T, F)
    B = spec2.shape[0]
    frames = torch.fft.irfft(spec2, n=n_fft, dim=-1)  # [B, T, n_fft]
    frames = frames * window

    out_len = (T - 1) * hop + n_fft
    # overlap-add via fold: treat frames as [B, n_fft, T] "columns"
    frames_t = frames.transpose(1, 2)  # [B, n_fft, T]
    ola = torch.nn.functional.fold(
        frames_t, output_size=(1, out_len), kernel_size=(1, n_fft), stride=(1, hop)
    ).reshape(B, out_len)

    win_sq = (window * window).unsqueeze(0).unsqueeze(-1)  # [1, n_fft, 1]
    win_sq = win_sq.expand(1, n_fft, T)
    envelope = torch.nn.functional.fold(
        win_sq, output_size=(1, out_len), kernel_size=(1, n_fft), stride=(1, hop)
    ).reshape(out_len)
    envelope = envelope.clamp_min(1e-8)

    ola = ola / envelope.unsqueeze(0)

    if out_len >= length:
        ola = ola[:, :length]
    else:
        ola = torch.nn.functional.pad(ola, (0, length - out_len))
    return ola.reshape(*lead, length)


def n_frames(length: int, n_fft: int = N_FFT, hop: int = HOP) -> int:
    return (length - n_fft) // hop + 1
