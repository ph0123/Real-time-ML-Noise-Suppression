"""Mask-estimator architectures + STFT helpers.

Five architectures are compared (see plot_results.py): CNN, RNN, BiRNN, LSTM, GRU.
All share one interface so training / evaluation / export are architecture-
agnostic::
    mask, state = model(log_mag, state)      # log_mag/mask: (B, T, 257)
Fixed STFT parameters shared 1:1 with the C++ engine:
  sample rate 16 kHz, FFT 512 (32 ms), hop 128 (8 ms), 257 bins, Hann window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1  # 257
HIDDEN = 256
LAYERS = 2

# Architectures compared in this project.
MODEL_NAMES = ["cnn1d", "cnn2d", "rnn", "birnn", "lstm", "gru"]
# Causal models that can be exported to a streaming ONNX graph. BiRNN is bidirectional (needs future frames) so it is comparison-only.
STREAMABLE = {"cnn1d", "cnn2d", "rnn", "lstm", "gru"}
# Of those, the ones that also run in real time: recurrent nets do O(1) work per frame, whereas the CNNs recompute their context window every hop. The 2-D CNN
# is the heaviest (convolves over frequency too) and blows the real-time budget (RTF > 1); deployment is therefore chosen among the recurrent trio.
DEPLOYABLE = {"rnn", "lstm", "gru"}


class RecurrentDenoiser(nn.Module):
    """RNN / GRU / LSTM (optionally bidirectional) magnitude-mask estimator."""

    def __init__(self, cell: str, bidirectional: bool = False):
        super().__init__()
        self.cell_name = "birnn" if (cell == "rnn" and bidirectional) else cell
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(N_BINS)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(N_BINS, HIDDEN, num_layers=LAYERS, batch_first=True,
                           bidirectional=bidirectional)
        out_dim = HIDDEN * (2 if bidirectional else 1)
        self.fc = nn.Sequential(nn.Linear(out_dim, N_BINS), nn.Sigmoid())

    def forward(self, log_mag, state=None):
        x = self.norm(log_mag)
        x, state = self.rnn(x, state)
        return self.fc(x), state


class _CausalConv1d(nn.Module):
    """Conv1d over time with left-only padding (freq bins are the channels),
    so frame t never sees the future."""

    def __init__(self, cin, cout, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):  # x: (B, C=freq, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class _CausalConv2d(nn.Module):
    """Conv2d over (time, frequency): causal (left-only) padding in time,
    symmetric padding in frequency, so frame t never sees the future."""

    def __init__(self, cin, cout, kt, kf, dilation_t):
        super().__init__()
        self.pad_t = (kt - 1) * dilation_t   # left pad on the time axis
        self.pad_f = (kf - 1) // 2           # symmetric pad on the freq axis
        self.conv = nn.Conv2d(cin, cout, (kt, kf), dilation=(dilation_t, 1))

    def forward(self, x):  # x: (B, C, T, F)
        # F.pad order for 4-D: (F_left, F_right, T_left, T_right)
        x = F.pad(x, (self.pad_f, self.pad_f, self.pad_t, 0))
        return self.conv(x)


class Cnn1dDenoiser(nn.Module):
    """1-D temporal CNN (TCN): dilated causal Conv1d over time, with the 257
    frequency bins as channels. Lighter than the 2-D CNN because it does not
    convolve across frequency -> cheaper per streaming step."""

    KERNEL = 3
    DILATIONS = (1, 2, 4)

    def __init__(self, channels: int = 128):
        super().__init__()
        self.cell_name = "cnn1d"
        self.bidirectional = False
        rf = sum((self.KERNEL - 1) * d for d in self.DILATIONS)  # 14
        self.context = 3 * rf  # 42
        self.norm = nn.LayerNorm(N_BINS)
        layers, cin = [], N_BINS
        for d in self.DILATIONS:
            layers += [_CausalConv1d(cin, channels, self.KERNEL, d), nn.ReLU()]
            cin = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(channels, N_BINS, 1), nn.Sigmoid())

    def conv(self, x_norm):
        """Causal 1-D conv stack on already-normalized frames (B, T, 257)."""
        x = x_norm.transpose(1, 2)               # (B, F, T)
        x = self.head(self.body(x))              # (B, F, T)
        return x.transpose(1, 2)                  # (B, T, F)

    def forward(self, log_mag, state=None):
        return self.conv(self.norm(log_mag)), state


class Cnn2dDenoiser(nn.Module):
    """2-D CNN over the magnitude spectrogram (time x frequency, 1 channel).
    Convolves over BOTH axes (spectrogram-as-image), dilated and causal in time
    so it still streams frame-by-frame. Heavier than the 1-D CNN.
    """

    KT, KF = 3, 3                # time / frequency kernel
    DILATIONS = (1, 2, 4)        # time dilations (frequency dilation is 1)

    def __init__(self, channels: int = 32):
        super().__init__()
        self.cell_name = "cnn2d"
        self.bidirectional = False
        # Streaming keeps this many past (normalized) frames as state; a margin over the time receptive field keeps a streaming step ~exact after a
        # short warm-up (deep dilated layers still pad the very first frames).
        rf_t = sum((self.KT - 1) * d for d in self.DILATIONS)  # 14
        self.context = 3 * rf_t  # 42
        self.norm = nn.LayerNorm(N_BINS)
        layers, cin = [], 1
        for d in self.DILATIONS:
            layers += [_CausalConv2d(cin, channels, self.KT, self.KF, d), nn.ReLU()]
            cin = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, 1)  # collapse channels -> mask

    def conv(self, x_norm):
        """Causal 2-D conv stack on already-normalized frames (B, T, 257)."""
        x = x_norm.unsqueeze(1)                  # (B, 1, T, F)
        x = torch.sigmoid(self.head(self.body(x)))
        return x.squeeze(1)                       # (B, T, F)

    def forward(self, log_mag, state=None):
        return self.conv(self.norm(log_mag)), state


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return RecurrentDenoiser("gru")
    if name == "rnn":
        return RecurrentDenoiser("rnn")
    if name == "lstm":
        return RecurrentDenoiser("lstm")
    if name == "birnn":
        return RecurrentDenoiser("rnn", bidirectional=True)
    if name == "cnn1d":
        return Cnn1dDenoiser()
    if name == "cnn2d":
        return Cnn2dDenoiser()
    raise ValueError(f"unknown model '{name}', pick from {MODEL_NAMES}")


# Backwards-compatible alias (the original project shipped only the GRU).
GruDenoiser = lambda: build_model("gru")


# --------------------------------------------------------------------------- #
# STFT helpers                                                                 #
# --------------------------------------------------------------------------- #
def stft_mag(wave: torch.Tensor):
    """wave (B, N) -> (mag (B, T, 257), complex spec)"""
    window = torch.hann_window(N_FFT, device=wave.device)
    spec = torch.stft(wave, N_FFT, HOP, win_length=N_FFT, window=window,
                      center=True, return_complex=True)      # (B, 257, T)
    mag = spec.abs().transpose(1, 2)                          # (B, T, 257)
    return mag, spec


def apply_mask_istft(spec: torch.Tensor, mask: torch.Tensor, length: int):
    """Multiply mask onto complex spec (phase preserved), back to waveform."""
    window = torch.hann_window(N_FFT, device=spec.device)
    masked = spec * mask.transpose(1, 2)                      # (B, 257, T)
    return torch.istft(masked, N_FFT, HOP, win_length=N_FFT,
                       window=window, center=True, length=length)


# --------------------------------------------------------------------------- #
# Streaming export helpers (one frame in, one mask out, single state tensor)   #
# --------------------------------------------------------------------------- #
class StreamingWrapper(nn.Module):
    """Run a causal denoiser one STFT frame at a time.
    Exposes a SINGLE state tensor (packed per architecture) so the ONNX graph
    and the C++ engine stay architecture-agnostic:
      * rnn / gru   : GRU/RNN hidden state           (LAYERS, 1, HIDDEN)
      * lstm        : [h ; c] stacked on dim 0        (2*LAYERS, 1, HIDDEN)
      * cnn1d/cnn2d : last `context` normalized frames (1, context, 257)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.kind = model.cell_name

    def forward(self, log_mag, state):
        if self.kind in ("rnn", "gru"):
            mask, new_state = self.model(log_mag, state)
            return mask, new_state
        if self.kind == "lstm":
            h, c = state[:LAYERS].contiguous(), state[LAYERS:].contiguous()
            mask, (h2, c2) = self.model(log_mag, (h, c))
            return mask, torch.cat([h2, c2], dim=0)
        if self.kind in ("cnn1d", "cnn2d"):
            # State holds NORMALIZED past frames so the zero initial state equals
            # the full-sequence conv's zero left-padding (which is applied AFTER
            # the LayerNorm). Normalizing here keeps a streaming step exact.
            xn = self.model.norm(log_mag)              # (1, 1, 257)
            buf = torch.cat([state, xn], dim=1)        # (1, context+1, 257)
            mask_full = self.model.conv(buf)
            return mask_full[:, -1:, :], buf[:, 1:, :]  # last frame, roll state
        raise ValueError(f"'{self.kind}' is not streamable")


def streaming_state(model: nn.Module) -> torch.Tensor:
    """Zero initial state tensor matching StreamingWrapper's packing."""
    k = model.cell_name
    if k in ("rnn", "gru"):
        return torch.zeros(LAYERS, 1, HIDDEN)
    if k == "lstm":
        return torch.zeros(2 * LAYERS, 1, HIDDEN)
    if k in ("cnn1d", "cnn2d"):
        return torch.zeros(1, model.context, N_BINS)
    raise ValueError(f"'{k}' is not streamable")
