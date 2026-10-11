import numpy as np

from deepanc.acoustics.rir import (
    ERROR_MIC,
    LOUDSPEAKER,
    REF_MIC,
    SOUND_SPEED,
    build_quiet_zone_set,
    build_training_set,
    generate_pair,
    sample_sphere_points,
)


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def test_direct_path_delay_primary():
    # Use an anechoic (T60=0) RIR to isolate the direct-path arrival: with
    # reverberation, early reflections in this small room can exceed the
    # direct sound in amplitude, and the band-limited (sinc-interpolated)
    # impulse also spreads/pre-rings a few samples around its true peak, so
    # neither argmax nor a relative-threshold onset is a reliable delay
    # marker once reflections are present.
    p, _ = generate_pair(0.0)
    expected_delay = _dist(REF_MIC, ERROR_MIC) / SOUND_SPEED * 16000
    peak = int(np.argmax(np.abs(p)))
    assert abs(peak - expected_delay) <= 2, (peak, expected_delay)


def test_direct_path_delay_secondary():
    _, s = generate_pair(0.0)
    expected_delay = _dist(LOUDSPEAKER, ERROR_MIC) / SOUND_SPEED * 16000
    peak = int(np.argmax(np.abs(s)))
    assert abs(peak - expected_delay) <= 2, (peak, expected_delay)


def test_training_set_shapes():
    rset = build_training_set()
    assert rset.primary.shape[0] == 5
    assert rset.primary.shape[1] == 512


def test_sphere_sampling_within_radius():
    pts = sample_sphere_points(200, 0.05, center=ERROR_MIC, seed=1)
    center = np.array(ERROR_MIC)
    dists = np.linalg.norm(pts - center, axis=1)
    assert dists.max() <= 0.05 + 1e-6
    assert dists.min() >= 0.0


def test_quiet_zone_set_shape():
    rset = build_quiet_zone_set(n_points=5, seed=2)
    assert rset.primary.shape == (5, 512)
    assert rset.secondary.shape == (5, 512)
