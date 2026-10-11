"""Convolutional Recurrent Network (CRN) for complex spectral mapping,
following the encoder-decoder-with-skip-connections architecture of Fig. 3
and the causal CRN of Tan & Wang (2019a), with a grouped LSTM bottleneck
(Gao et al., 2018, groups=2) as specified in Section 3.3.

Input/output: [B, 2, T, F] real/imaginary spectrogram, F = 161 (n_fft=320).

Encoder: 5 causal Conv2d layers, frequency stride 2, channels
  2 -> 16 -> 32 -> 64 -> 128 -> 256
  frequency dim: 161 -> 80 -> 39 -> 19 -> 9 -> 4 (kernel_size=(2,3))
Bottleneck: reshape to [B, T, 256*4=1024] -> 2-layer grouped LSTM (groups=2,
  512 units per group) -> reshape back to [B, 256, T, 4]
Decoder: 5 causal ConvTranspose2d layers, each fed cat(prev_output, skip)
  so input channels double; output channels 256->128->64->32->16->2
  frequency dim: 4 -> 9 -> 19 -> 39 -> 80 -> 161
Skip connections: encoder layer i's output feeds decoder layer (5-i)'s input
  (concatenated with the previous decoder layer's output), per Fig. 3.
All convolutions/deconvolutions are causal in time: they only look at the
current and past time frames, never future ones, so the network can run
streaming/online (Section 3.3: "all convolutions and deconvolutions are
causal... thus suited for real-time implementation").
"""

from __future__ import annotations

import torch
import torch.nn as nn

ENCODER_CHANNELS = [2, 16, 32, 64, 128, 256]
FREQ_KERNEL = 3
TIME_KERNEL = 2
FREQ_STRIDE = 2
# For F=161 the encoder's frequency progression (kernel=3, stride=2, pad=1)
# is 161 -> 81 -> 41 -> 21 -> 11 -> 6. The decoder's output_padding per
# layer is computed in CRN.__init__ from the actual encoder freqs so this
# always exactly inverts the encoder's downsampling, for any n_freq.


class CausalConv2d(nn.Module):
    """Conv2d causal in time (dim -2): pads (kernel_time - 1) frames on the
    past side only, and pads/strides normally in frequency (dim -1).
    """

    def __init__(self, in_ch, out_ch, time_kernel=TIME_KERNEL, freq_kernel=FREQ_KERNEL, freq_stride=FREQ_STRIDE):
        super().__init__()
        self.time_pad = time_kernel - 1
        # frequency padding chosen so that with stride 2 and kernel 3,
        # out_freq = ceil(in_freq / 2); pad=1 on both sides of freq then trim.
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(time_kernel, freq_kernel),
            stride=(1, freq_stride),
            padding=(0, 1),
        )

    def forward(self, x):
        # x: [B, C, T, F]
        x = nn.functional.pad(x, (0, 0, self.time_pad, 0))  # pad time dim (past)
        return self.conv(x)


class CausalConvTranspose2d(nn.Module):
    """ConvTranspose2d causal in time: after the transpose conv (which adds
    time_kernel-1 extra frames symmetric-ish), trim the *future* extra frame
    so output frame t depends only on input frames <= t.
    """

    def __init__(self, in_ch, out_ch, time_kernel=TIME_KERNEL, freq_kernel=FREQ_KERNEL, freq_stride=FREQ_STRIDE, output_padding=0):
        super().__init__()
        self.time_kernel = time_kernel
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=(time_kernel, freq_kernel),
            stride=(1, freq_stride),
            padding=(0, 1),
            output_padding=(0, output_padding),
        )

    def forward(self, x):
        out = self.deconv(x)
        # ConvTranspose2d with time_kernel=2, stride=1 in time adds
        # (time_kernel - 1) extra frames at the *end*; drop them so the
        # output length matches the input length and stays causal.
        extra = self.time_kernel - 1
        if extra > 0:
            out = out[:, :, :-extra, :]
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = CausalConv2d(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, output_padding=0, final=False):
        super().__init__()
        self.deconv = CausalConvTranspose2d(in_ch, out_ch, output_padding=output_padding)
        self.final = final
        if not final:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ELU()

    def forward(self, x):
        out = self.deconv(x)
        if self.final:
            return out
        return self.act(self.bn(out))


class GroupedLSTM(nn.Module):
    """Multi-layer LSTM with a group strategy (Gao et al., 2018): the
    feature dimension is split into `groups` equal chunks, each processed
    by an independent LSTM, and between layers the group outputs are
    interleaved ("shuffled") so information mixes across groups over
    depth — analogous to a grouped-conv + channel-shuffle block, but for
    recurrent layers. This cuts LSTM parameter count by ~1/groups vs. a
    single full-width LSTM while keeping global context via shuffling.
    """

    def __init__(self, input_size: int, num_layers: int = 2, groups: int = 2):
        super().__init__()
        assert input_size % groups == 0
        self.groups = groups
        self.group_size = input_size // groups
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([nn.LSTM(self.group_size, self.group_size, batch_first=True) for _ in range(groups)])
                for _ in range(num_layers)
            ]
        )

    def _shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, groups*group_size] -> interleave channels across groups
        B, T, C = x.shape
        x = x.view(B, T, self.groups, self.group_size)
        x = x.transpose(2, 3).contiguous()  # [B, T, group_size, groups]
        return x.view(B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_size]
        for layer_idx, group_lstms in enumerate(self.layers):
            chunks = x.chunk(self.groups, dim=-1)
            outs = [lstm(chunk)[0] for lstm, chunk in zip(group_lstms, chunks)]
            x = torch.cat(outs, dim=-1)
            if layer_idx < self.num_layers - 1:
                x = self._shuffle(x)
        return x


class CRN(nn.Module):
    """Causal convolutional-recurrent network for complex spectral mapping.

    Args:
        n_freq: number of STFT frequency bins (161 for n_fft=320).
        channels: encoder channel progression, default per Fig. 3.
        lstm_layers, lstm_groups: grouped-LSTM bottleneck config.
    """

    def __init__(self, n_freq: int = 161, channels=None, lstm_layers: int = 2, lstm_groups: int = 2):
        super().__init__()
        channels = channels or ENCODER_CHANNELS
        self.n_freq = n_freq

        freqs = [n_freq]
        f = n_freq
        for _ in range(len(channels) - 1):
            f = (f + 2 * 1 - FREQ_KERNEL) // FREQ_STRIDE + 1
            freqs.append(f)
        self.freqs = freqs  # e.g. [161, 80, 39, 19, 9, 4]

        self.encoders = nn.ModuleList(
            [EncoderBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        bottleneck_dim = channels[-1] * freqs[-1]
        self.lstm = GroupedLSTM(bottleneck_dim, num_layers=lstm_layers, groups=lstm_groups)

        rev_channels = list(reversed(channels))  # [256,128,64,32,16,2]
        rev_freqs = list(reversed(freqs))  # e.g. [6,11,21,41,81,161]
        n_dec = len(rev_channels) - 1
        self.decoders = nn.ModuleList()
        for i in range(n_dec):
            in_ch = rev_channels[i] * 2  # concatenated with skip connection
            out_ch = rev_channels[i + 1]
            final = i == n_dec - 1
            in_f, target_f = rev_freqs[i], rev_freqs[i + 1]
            # invert out = (in-1)*stride - 2*pad + kernel + output_padding
            base = (in_f - 1) * FREQ_STRIDE - 2 * 1 + FREQ_KERNEL
            out_pad = target_f - base
            assert 0 <= out_pad < FREQ_STRIDE, (in_f, target_f, out_pad)
            self.decoders.append(DecoderBlock(in_ch, out_ch, output_padding=out_pad, final=final))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, T, F]
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        B, C, T, Fb = h.shape
        h_seq = h.permute(0, 2, 1, 3).reshape(B, T, C * Fb)
        h_seq = self.lstm(h_seq)
        h = h_seq.reshape(B, T, C, Fb).permute(0, 2, 1, 3)

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
