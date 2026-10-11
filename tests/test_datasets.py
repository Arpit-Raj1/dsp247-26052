import numpy as np
import pytest

from deepanc.data.datasets import SIGNAL_LEN, ANCNoiseDataset, ANCNoisySpeechDataset

pytest.importorskip("h5py")

NOISE_H5 = "data/noise_train.h5"
TIMIT_H5 = "data/timit.h5"

import os

skip_if_missing = pytest.mark.skipif(
    not (os.path.exists(NOISE_H5) and os.path.exists(TIMIT_H5)),
    reason="requires prepared data/noise_train.h5 and data/timit.h5 (run scripts/prepare_*.py first)",
)


@skip_if_missing
def test_noise_dataset_shapes_and_determinism():
    ds = ANCNoiseDataset(NOISE_H5, n_items=10)
    item = ds[3]
    assert item["x"].shape == (SIGNAL_LEN,)
    assert item["speech"].shape == (SIGNAL_LEN,)
    item2 = ds[3]
    assert np.allclose(item["x"].numpy(), item2["x"].numpy())


@skip_if_missing
def test_noise_dataset_items_differ():
    ds = ANCNoiseDataset(NOISE_H5, n_items=10)
    a = ds[0]["x"].numpy()
    b = ds[1]["x"].numpy()
    assert not np.allclose(a, b)


@skip_if_missing
def test_noisy_speech_dataset_shapes():
    ds = ANCNoisySpeechDataset(NOISE_H5, TIMIT_H5, n_items=10)
    item = ds[0]
    assert item["x"].shape == (SIGNAL_LEN,)
    assert item["speech"].shape == (SIGNAL_LEN,)
    assert item["snr_db"].item() in (5.0, 10.0, 15.0, 20.0)


@skip_if_missing
def test_noisy_speech_dataset_determinism():
    ds = ANCNoisySpeechDataset(NOISE_H5, TIMIT_H5, n_items=10)
    a = ds[5]["x"].numpy()
    b = ds[5]["x"].numpy()
    assert np.allclose(a, b)
