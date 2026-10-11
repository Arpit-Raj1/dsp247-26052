"""Audio I/O helpers: NIST SPHERE (TIMIT) reading, generic WAV reading, and
resampling to the pipeline's working rate (16 kHz).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000


def read_sphere_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a NIST SPHERE-headered WAV file (TIMIT's native format).

    TIMIT .WAV files are NIST SPHERE, not RIFF WAV, so `soundfile` can't
    read them directly. The header is a fixed 1024-byte ASCII block
    (`NIST_1A` + key-value fields) followed by raw samples. We parse the
    header for sample_rate/sample_count/sample_n_bytes/channel_count, then
    read the raw PCM directly. TIMIT audio is 16-bit signed, mono,
    little-endian ("01" encoding per the corpus docs).
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:7] == b"NIST_1A", f"not a SPHERE file: {path}"
    header_len = int(raw[8:16].split()[0])
    header_text = raw[:header_len].decode("ascii", errors="ignore")

    fields = {}
    for line in header_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].startswith("-"):
            fields[parts[0]] = parts[2]

    sample_rate = int(fields.get("sample_rate", TARGET_SR))
    n_channels = int(fields.get("channel_count", 1))
    sample_n_bytes = int(fields.get("sample_n_bytes", 2))
    sample_count = int(fields.get("sample_count", 0))

    data = raw[header_len:]
    if sample_n_bytes == 2:
        dtype = np.dtype("<i2")  # little-endian int16
    else:
        raise ValueError(f"unsupported sample_n_bytes={sample_n_bytes} in {path}")

    audio = np.frombuffer(data, dtype=dtype)
    if sample_count and len(audio) > sample_count * n_channels:
        audio = audio[: sample_count * n_channels]
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    audio_f = audio.astype(np.float32) / 32768.0
    return audio_f, sample_rate


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a regular RIFF WAV (or any soundfile-supported) file as float32
    mono, falling back to read_sphere_wav if the file is SPHERE-headered.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic[:7] == b"NIST_1A":
        return read_sphere_wav(path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def resample(audio: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    return soxr.resample(audio.astype(np.float32), sr_in, sr_out).astype(np.float32)


def load_resampled(path: str | Path, sr_out: int = TARGET_SR) -> np.ndarray:
    audio, sr = read_wav(path)
    return resample(audio, sr, sr_out)
