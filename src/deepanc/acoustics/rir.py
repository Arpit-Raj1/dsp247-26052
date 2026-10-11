"""Room impulse response generation via the image method (Allen & Berkley, 1979).

Matches the paper's setup (Section 4.2):
- room 3m x 4m x 2m (width x length x height)
- reference mic at (1.5, 1, 1)
- canceling loudspeaker at (1.5, 2.5, 1)
- error mic at (1.5, 3, 1)
- fs = 16000, RIR length = 512
- primary path: source colocated with the reference mic (its RIR is the
  "primary path" p(t) from wherever the primary noise/speech originates,
  modeled here as originating at the reference mic position) -> error mic
- secondary path: loudspeaker -> error mic
- training T60s: {0.15, 0.175, 0.2, 0.225, 0.25} s
- test T60 (default): 0.2 s; untrained T60s for Fig. 8:
  {0.16, 0.17, 0.18, 0.19, 0.21, 0.22} s
- quiet zone (Section 5.3): 100 error-mic points sampled uniformly in a 5cm
  radius sphere around the nominal error-mic position, T60 = 0.2s.

Uses the `rir_generator` package (Python bindings for the classic image
method), c = 343 m/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS = 16000
ROOM = (3.0, 4.0, 2.0)  # width, length, height (m)
REF_MIC = (1.5, 1.0, 1.0)
LOUDSPEAKER = (1.5, 2.5, 1.0)
ERROR_MIC = (1.5, 3.0, 1.0)
NSAMPLE = 512
SOUND_SPEED = 343.0

TRAIN_T60S = (0.15, 0.175, 0.2, 0.225, 0.25)
TEST_T60 = 0.2
UNTRAINED_T60S = (0.16, 0.17, 0.18, 0.19, 0.21, 0.22)

QUIET_ZONE_RADIUS = 0.05  # 5 cm
QUIET_ZONE_N_POINTS = 100


def _generate_rir(source_pos, mic_pos, t60: float, nsample: int = NSAMPLE) -> np.ndarray:
    import rir_generator as rg

    h = rg.generate(
        c=SOUND_SPEED,
        fs=FS,
        r=[list(mic_pos)],
        s=list(source_pos),
        L=list(ROOM),
        reverberation_time=t60,
        nsample=nsample,
    )
    return np.asarray(h).reshape(-1).astype(np.float32)


def generate_pair(t60: float, error_mic_pos=ERROR_MIC, nsample: int = NSAMPLE):
    """Return (primary_rir, secondary_rir) for a given T60 and error-mic position.

    primary_rir: REF_MIC -> error_mic_pos
    secondary_rir: LOUDSPEAKER -> error_mic_pos
    """
    p = _generate_rir(REF_MIC, error_mic_pos, t60, nsample)
    s = _generate_rir(LOUDSPEAKER, error_mic_pos, t60, nsample)
    return p, s


def sample_sphere_points(n_points: int, radius: float, center=ERROR_MIC, seed: int = 0) -> np.ndarray:
    """Uniformly sample n_points inside a sphere of given radius around center.

    Uses rejection-free sampling: draw a random direction (uniform on the
    sphere) and a random radius r = radius * u^(1/3) so points are uniform
    in volume, not just on the surface.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    u = rng.uniform(0.0, 1.0, size=n_points)
    radii = radius * u ** (1.0 / 3.0)
    points = center[None, :] + directions * radii[:, None]
    return points.astype(np.float32)


@dataclass
class RirSet:
    """Container for a batch of (primary, secondary) RIR pairs with metadata."""

    primary: np.ndarray  # [N, nsample]
    secondary: np.ndarray  # [N, nsample]
    t60: np.ndarray  # [N]
    error_mic_pos: np.ndarray  # [N, 3]
    tag: str = ""


def build_training_set(t60s=TRAIN_T60S, nsample: int = NSAMPLE) -> RirSet:
    primaries, secondaries, t60_list = [], [], []
    for t60 in t60s:
        p, s = generate_pair(t60, ERROR_MIC, nsample)
        primaries.append(p)
        secondaries.append(s)
        t60_list.append(t60)
    return RirSet(
        primary=np.stack(primaries),
        secondary=np.stack(secondaries),
        t60=np.array(t60_list, dtype=np.float32),
        error_mic_pos=np.tile(np.asarray(ERROR_MIC, dtype=np.float32), (len(t60s), 1)),
        tag="train",
    )


def build_test_set(t60s, nsample: int = NSAMPLE, tag: str = "test") -> RirSet:
    primaries, secondaries, t60_list = [], [], []
    for t60 in t60s:
        p, s = generate_pair(t60, ERROR_MIC, nsample)
        primaries.append(p)
        secondaries.append(s)
        t60_list.append(t60)
    return RirSet(
        primary=np.stack(primaries),
        secondary=np.stack(secondaries),
        t60=np.array(t60_list, dtype=np.float32),
        error_mic_pos=np.tile(np.asarray(ERROR_MIC, dtype=np.float32), (len(t60s), 1)),
        tag=tag,
    )


def build_quiet_zone_set(
    n_points: int = QUIET_ZONE_N_POINTS,
    radius: float = QUIET_ZONE_RADIUS,
    t60: float = TEST_T60,
    seed: int = 0,
    nsample: int = NSAMPLE,
) -> RirSet:
    points = sample_sphere_points(n_points, radius, ERROR_MIC, seed=seed)
    primaries, secondaries = [], []
    for pt in points:
        p, s = generate_pair(t60, pt, nsample)
        primaries.append(p)
        secondaries.append(s)
    return RirSet(
        primary=np.stack(primaries),
        secondary=np.stack(secondaries),
        t60=np.full(n_points, t60, dtype=np.float32),
        error_mic_pos=points,
        tag="quiet_zone",
    )


def save_rir_sets(path: str, sets: dict[str, RirSet]) -> None:
    payload = {}
    for name, rset in sets.items():
        payload[f"{name}.primary"] = rset.primary
        payload[f"{name}.secondary"] = rset.secondary
        payload[f"{name}.t60"] = rset.t60
        payload[f"{name}.error_mic_pos"] = rset.error_mic_pos
    np.savez_compressed(path, **payload)


def load_rir_sets(path: str) -> dict[str, RirSet]:
    data = np.load(path)
    names = sorted({k.split(".")[0] for k in data.files})
    out = {}
    for name in names:
        out[name] = RirSet(
            primary=data[f"{name}.primary"],
            secondary=data[f"{name}.secondary"],
            t60=data[f"{name}.t60"],
            error_mic_pos=data[f"{name}.error_mic_pos"],
            tag=name,
        )
    return out
