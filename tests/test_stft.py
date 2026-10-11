import torch

from deepanc.audio.stft import HOP, N_FFT, istft, n_frames, stft


def test_perfect_reconstruction_interior():
    torch.manual_seed(0)
    L = N_FFT * 20
    x = torch.randn(L)
    X = stft(x)
    y = istft(X, length=L)
    # interior samples (away from first/last n_fft - hop) should reconstruct
    # near-exactly; edges are attenuated by the clamped envelope.
    margin = N_FFT - HOP
    err = (x[margin:-margin] - y[margin:-margin]).abs().max().item()
    assert err < 1e-4, f"max interior reconstruction error {err}"


def test_batched_shapes():
    x = torch.randn(4, 3, 3200)
    X = stft(x)
    assert X.shape[:-1] == (4, 3, n_frames(3200))
    assert X.shape[-1] == N_FFT // 2 + 1
    y = istft(X, length=3200)
    assert y.shape == (4, 3, 3200)


def test_causal_framing():
    # frame 0 should only depend on samples [0, n_fft)
    torch.manual_seed(1)
    L = N_FFT * 10
    x1 = torch.randn(L)
    x2 = x1.clone()
    x2[N_FFT:] = torch.randn(L - N_FFT)  # perturb everything after frame 0
    X1 = stft(x1)
    X2 = stft(x2)
    assert torch.allclose(X1[0], X2[0], atol=1e-5)
