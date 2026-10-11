import torch

from deepanc.anc.simulate import apply_delay_compensation, crn_forward_waveform, simulate
from deepanc.audio.stft import N_FFT


class IdentityModel(torch.nn.Module):
    """Passes real/imag spectrogram through unchanged (oracle: y == x)."""

    def forward(self, spec):
        return spec


def test_delay_compensation_shift():
    T, F = 10, 4
    spec = torch.arange(T * F, dtype=torch.float32).reshape(1, T, F)
    shifted = apply_delay_compensation(spec, 2)
    assert torch.all(shifted[:, :2] == 0)
    assert torch.allclose(shifted[:, 2:], spec[:, : T - 2])


def test_delay_compensation_zero_is_noop():
    spec = torch.randn(1, 10, 4)
    assert torch.equal(apply_delay_compensation(spec, 0), spec)


def test_oracle_loss_path_linear_case():
    # If s(t) is a pure delay-1 impulse and eta2 = inf (linear loudspeaker),
    # and the primary path equals the secondary path, then an oracle y = x
    # (identity model) should drive e close to zero.
    torch.manual_seed(0)
    B, L = 2, N_FFT * 8
    x = torch.randn(B, L) * 0.1
    K = 16
    rir = torch.zeros(B, K)
    rir[:, 3] = 1.0  # identical primary/secondary path: pure delay of 3
    model = IdentityModel()
    out = simulate(model, x, primary_rir=rir, secondary_rir=rir, eta2=float("inf"), m_frames=0)
    # exclude STFT/iSTFT edge regions (envelope not fully built up) and the
    # RIR's own onset, where reconstruction is not exact by construction.
    margin = N_FFT
    err = out.e[:, margin:-margin].abs().max().item()
    assert err < 0.01, f"oracle error too large: {err}"


def test_crn_forward_waveform_shape():
    model = IdentityModel()
    x = torch.randn(3, N_FFT * 5)
    y = crn_forward_waveform(model, x, m_frames=1)
    assert y.shape == x.shape


def test_simulate_secondary_fn_override():
    B, L = 2, N_FFT * 4
    x = torch.randn(B, L) * 0.1
    rir = torch.zeros(B, 8)
    rir[:, 0] = 1.0
    model = IdentityModel()

    calls = []

    def secondary_fn(y):
        calls.append(y.shape)
        return y * 0.5  # trivial stand-in secondary path

    out = simulate(model, x, primary_rir=rir, secondary_rir=rir, eta2=float("inf"), secondary_fn=secondary_fn)
    assert calls and calls[0] == (B, L)
    assert torch.allclose(out.a, out.y * 0.5)
