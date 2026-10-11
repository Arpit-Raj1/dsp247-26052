import torch

from deepanc.models.crn import CRN


def test_output_shape_matches_input():
    model = CRN(n_freq=161)
    x = torch.randn(2, 2, 20, 161)
    y = model(x)
    assert y.shape == x.shape


def test_param_count_reasonable():
    model = CRN(n_freq=161)
    n = model.num_parameters()
    # paper reports the CRN is parameter-efficient but gives no exact count;
    # Tan & Wang (2019a)-style CRNs with a ~1500-dim grouped-LSTM bottleneck
    # typically land in the 10-25M range (dominated by the LSTM weights).
    assert 1_000_000 < n < 25_000_000, n


def test_causality_perturb_future():
    model = CRN(n_freq=161)
    model.eval()
    torch.manual_seed(0)
    x1 = torch.randn(1, 2, 15, 161)
    x2 = x1.clone()
    x2[:, :, 8:, :] = torch.randn(1, 2, 7, 161)  # perturb frames >= 8
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    # frames < 8 must be unaffected by perturbing frames >= 8
    assert torch.allclose(y1[:, :, :8, :], y2[:, :, :8, :], atol=1e-5)


def test_variable_time_length():
    model = CRN(n_freq=161)
    model.eval()
    with torch.no_grad():
        for T in [3, 7, 16, 33]:
            x = torch.randn(1, 2, T, 161)
            y = model(x)
            assert y.shape == (1, 2, T, 161), T


def test_gradients_flow():
    model = CRN(n_freq=161)
    x = torch.randn(2, 2, 10, 161, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
