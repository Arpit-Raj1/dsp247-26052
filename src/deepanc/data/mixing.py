"""SNR mixing and level normalization for training-example synthesis.

Not specified by the paper (see CLAUDE.md "Known paper ambiguities"): we
fix concrete, documented rules here so every generated example is
reproducible from its seed.
"""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale `noise` so that mixing it with `speech` (unchanged) yields the
    given SNR, then return (mixture, scaled_noise). Both must be the same
    length.

    SNR_dB = 10*log10( rms(speech)^2 / rms(scaled_noise)^2 )
    => scaled_noise = noise * rms(speech) / (rms(noise) * 10^(SNR_dB/20))
    """
    assert speech.shape == noise.shape, (speech.shape, noise.shape)
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-12)
    scaled_noise = noise * scale
    mixture = speech + scaled_noise
    return mixture.astype(np.float32), scaled_noise.astype(np.float32)


def measured_snr_db(speech: np.ndarray, noise: np.ndarray) -> float:
    return 20.0 * float(np.log10(rms(speech) / (rms(noise) + 1e-12)))


def scale_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    cur = rms(x)
    if cur < 1e-12:
        return x.astype(np.float32)
    return (x * (target_rms / cur)).astype(np.float32)
