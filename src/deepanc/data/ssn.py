"""Speech-shaped noise (SSN) generation.

SSN is white noise filtered to match the long-term average spectrum (LTAS)
of real speech, used as one of the four test noise types (Table 1/2, "SSN")
alongside the NOISEX-92 recordings. The paper takes SSN from NOISEX-92
directly; since our free NOISEX mirror does not include an SSN track, we
synthesize it from the TIMIT test-speaker LTAS (Section 4.1 test setup: SSN
"denoted 'SSN'" is used only for testing, so it must be speech-representative
but need not literally be the original NOISEX-92 SSN file).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def compute_ltas(speech_signals: list[np.ndarray], n_fft: int = 1024) -> np.ndarray:
    """Average power spectrum (LTAS) across a list of speech waveforms."""
    psum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    n = 0
    for sig in speech_signals:
        hop = n_fft // 2
        win = np.hanning(n_fft)
        for start in range(0, max(1, len(sig) - n_fft), hop):
            frame = sig[start : start + n_fft]
            if len(frame) < n_fft:
                break
            spec = np.fft.rfft(frame * win)
            psum += np.abs(spec) ** 2
            n += 1
    if n == 0:
        return np.ones(n_fft // 2 + 1)
    return psum / n


def generate_ssn(ltas: np.ndarray, length: int, seed: int = 0, n_fft: int = 1024) -> np.ndarray:
    """Synthesize `length` samples of noise whose long-term spectrum
    matches `ltas`, via FIR filtering of white noise (frequency-sampling
    design: filter magnitude = sqrt(ltas), linear-phase, applied with
    overlap-add through scipy's `lfilter` on a time-domain FIR).
    """
    rng = np.random.default_rng(seed)
    mag = np.sqrt(np.maximum(ltas, 0))
    mag = mag / (mag.max() + 1e-12)
    # design a linear-phase FIR from the magnitude spectrum via inverse FFT
    fir = np.fft.irfft(mag, n=n_fft)
    fir = np.fft.fftshift(fir)  # center the (symmetric) impulse response
    white = rng.standard_normal(length + n_fft).astype(np.float64)
    filtered = lfilter(fir, [1.0], white)
    out = filtered[n_fft // 2 : n_fft // 2 + length]
    out = out / (np.std(out) + 1e-12)
    return out.astype(np.float32)
