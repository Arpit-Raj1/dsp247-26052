"""STOI and PESQ metrics (Section 4.1): measure intelligibility/quality of
the error signal e(t) against the speech component of the primary signal,
p(t) * v(t) — i.e. how well the ideal denoised speech is preserved.
"""

from __future__ import annotations

import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

SR = 16000


def stoi_score(reference: np.ndarray, degraded: np.ndarray, sr: int = SR) -> float:
    return float(_stoi(reference, degraded, sr, extended=False))


def pesq_score(reference: np.ndarray, degraded: np.ndarray, sr: int = SR, mode: str = "wb") -> float:
    """mode: 'wb' (wideband, ITU-T P.862.2) needs sr=16000; 'nb'
    (narrowband, P.862) needs sr=8000 audio — resample both signals to
    8kHz before calling with mode='nb' (evaluate.py does this so both
    wideband and narrowband scores can be logged per Section 4.1).
    """
    return float(_pesq(sr, reference, degraded, mode))


# P.862.1 mapping (raw P.862 -> MOS-LQO), constants as in Loizou's pesq.m:
#   mos_lqo = 0.999 + (4.999-0.999) / (1 + exp(-1.4945*raw + 4.6607))
_NB_A, _NB_B = 1.4945, 4.6607


def nb_mos_to_raw(mos_lqo: float) -> float:
    """Invert P.862.1 to recover the raw P.862 score (range ~-0.5..4.5, the
    scale the paper's PESQ numbers appear to use)."""
    return float((_NB_B - np.log(4.0 / (mos_lqo - 0.999) - 1.0)) / _NB_A)


def pesq_nb_scores(reference: np.ndarray, degraded: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Narrowband PESQ, as pesq.m does for 8 kHz input: returns
    (raw P.862, MOS-LQO P.862.1). 16 kHz input is resampled to 8 kHz."""
    from scipy.signal import resample_poly

    if sr != 8000:
        reference = resample_poly(reference, 8000, sr)
        degraded = resample_poly(degraded, 8000, sr)
    mos = float(_pesq(8000, reference, degraded, "nb"))  # python-pesq returns the P.862.1-mapped score
    return nb_mos_to_raw(mos), mos
