import math

import torch
from scipy import integrate

from deepanc.acoustics.loudspeaker import sef, tanh_nl


def test_sef_identity_at_infinity():
    y = torch.linspace(-5, 5, 101)
    out = sef(y, float("inf"))
    assert torch.allclose(out, y)


def test_sef_matches_quadrature():
    eta2 = 1.0
    for yval in [-3.0, -0.5, 0.0, 0.7, 4.2]:
        expected, _ = integrate.quad(lambda z: math.exp(-(z**2) / (2 * eta2)), 0, yval)
        got = sef(torch.tensor(yval), eta2).item()
        assert abs(got - expected) < 1e-5, (yval, got, expected)


def test_sef_saturation_limit():
    eta2 = 0.1
    eta = math.sqrt(eta2)
    limit = eta * math.sqrt(math.pi / 2.0)
    y = torch.tensor([100.0])
    out = sef(y, eta2).item()
    assert abs(out - limit) < 1e-4


def test_sef_gradients_finite():
    y = torch.linspace(-10, 10, 21, requires_grad=True)
    out = sef(y, 0.1).sum()
    out.backward()
    assert torch.isfinite(y.grad).all()


def test_tanh_nl_bounds():
    y = torch.tensor([-100.0, 0.0, 100.0])
    out = tanh_nl(y)
    assert out[1].item() == 0.0
    assert out[0].item() < 0 < out[2].item()
    assert out.abs().max().item() <= 3.3 + 1e-4
