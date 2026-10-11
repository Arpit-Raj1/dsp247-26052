import numpy as np
import torch

from deepanc.acoustics.paths import batched_fir_convolve, nl_secondary_path, volterra_primary


def test_batched_conv_matches_numpy():
    torch.manual_seed(0)
    B, L, K = 5, 200, 17
    x = torch.randn(B, L)
    h = torch.randn(B, K)
    out = batched_fir_convolve(x, h)
    for b in range(B):
        expected = np.convolve(x[b].numpy(), h[b].numpy())[:L]
        assert np.allclose(out[b].numpy(), expected, atol=1e-4)


def test_batched_conv_identity_filter():
    x = torch.randn(3, 50)
    h = torch.zeros(3, 5)
    h[:, 0] = 1.0
    out = batched_fir_convolve(x, h)
    assert torch.allclose(out, x, atol=1e-5)


def test_volterra_primary_shape_and_causality():
    x = torch.randn(2, 100)
    d = volterra_primary(x)
    assert d.shape == x.shape
    # causality: perturbing future samples must not affect earlier outputs
    x2 = x.clone()
    x2[:, 50:] = torch.randn(2, 50)
    d1 = volterra_primary(x)
    d2 = volterra_primary(x2)
    assert torch.allclose(d1[:, :50], d2[:, :50], atol=1e-5)


def test_nl_secondary_path_shape():
    y = torch.randn(2, 100)
    a = nl_secondary_path(y)
    assert a.shape == y.shape
