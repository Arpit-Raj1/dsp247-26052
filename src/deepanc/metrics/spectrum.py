"""Short-time NMSE (Fig. 7a) and Welch power-spectrum (Fig. 7b) helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

SR = 16000


def short_time_nmse_db(e: np.ndarray, d: np.ndarray, window_s: float = 0.1, sr: int = SR) -> np.ndarray:
    """NMSE computed in non-overlapping `window_s`-second windows, for
    plotting NMSE vs. time (Fig. 7a). Returns one value per window.
    """
    win = int(window_s * sr)
    n_windows = len(e) // win
    out = np.zeros(n_windows, dtype=np.float64)
    for i in range(n_windows):
        seg_e = e[i * win : (i + 1) * win]
        seg_d = d[i * win : (i + 1) * win]
        num = np.sum(seg_e**2)
        den = np.sum(seg_d**2) + 1e-12
        out[i] = 10.0 * np.log10((num + 1e-12) / den)
    return out


def welch_psd_db(x: np.ndarray, sr: int = SR, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Returns (freqs, psd_db) via Welch's method."""
    freqs, psd = welch(x, fs=sr, nperseg=nperseg)
    psd_db = 10.0 * np.log10(psd + 1e-20)
    return freqs, psd_db
