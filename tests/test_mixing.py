import numpy as np

from deepanc.data.mixing import measured_snr_db, mix_at_snr, rms, scale_to_rms


def test_mix_at_snr_accuracy():
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(16000).astype(np.float32) * 0.1
    noise = rng.standard_normal(16000).astype(np.float32) * 0.3
    for target_snr in [0, 5, 10, 15, 20]:
        mixture, scaled_noise = mix_at_snr(speech, noise, target_snr)
        got_snr = measured_snr_db(speech, scaled_noise)
        assert abs(got_snr - target_snr) < 0.01, (target_snr, got_snr)


def test_scale_to_rms():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000).astype(np.float32)
    y = scale_to_rms(x, 0.05)
    assert abs(rms(y) - 0.05) < 1e-4


def test_scale_to_rms_silence_safe():
    x = np.zeros(100, dtype=np.float32)
    y = scale_to_rms(x, 0.1)
    assert np.all(y == 0)
