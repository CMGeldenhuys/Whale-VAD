from typing import Callable, Literal, Optional
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from torchaudio.transforms import (
    Spectrogram,
)

from whalevad.utils import padding_mask, unpad_mean


class SpectrogramExtractor(Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft=256,
        win_length: Optional[int] = None,  # Default: n_fft
        hop_length: Optional[int] = None,  # Default: win_length//2
        pad: int = 0,
        power: float | None = 2.0,
        normalized: bool | Literal["window", "frame_length"] = False,
        window_fn: Callable[..., Tensor] | str = "hann_window",
        center: bool = False,
        onesided: bool = True,
        pad_mode: str = "reflect",
        norm_features: Literal["demean"] | None = None,
        complex_repr: (
            Literal["real+imag", "mag+phase", "real+imag+mag+phase", "trig"] | None
        ) = "real+imag",
        apply_log: bool = False,
    ) -> None:
        super().__init__()
        if win_length is not None and n_fft != win_length:
            warnings.warn(
                f"n_fft != window_len, note output rate is based on n_fft not window_len. (See torch.sftf, {n_fft} != {win_length})"
            )

        # TODO: move to fn
        n_freq_bins = n_fft // 2 + 1

        if isinstance(window_fn, str):
            window_fn = getattr(torch, window_fn)
        assert isinstance(window_fn, Callable)

        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            power=power,
            normalized=normalized,
            window_fn=window_fn,
            center=center,
            onesided=onesided,
            pad_mode=pad_mode,
        )
        self.num_features = n_freq_bins
        self.sample_rate = sample_rate
        self.norm_features = norm_features
        self.complex_repr = complex_repr
        self.power = power
        self.apply_log = apply_log
        self.eps = 1e-8

    def forward(
        self,
        audio: Tensor,
        *,
        lengths: Optional[Tensor] = None,
        **_,
    ):
        # add empty batch dim
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        feat: Tensor = self.spectrogram(audio)
        # shape: batch, *, channel_dim=1, feat_dim, time_dim
        # Remove empty channel dim
        feat = feat.squeeze(-3)
        # ensure features are in correct order
        feat = feat.transpose(-1, -2)
        # shape: batch, time_dim, feat_dim

        if lengths is not None and self.norm_features == "demean":
            feat_mean = unpad_mean(feat, dim=-2, keepdim=True, lengths=lengths)
            # shape: (batch, 1, feat_dim)
            feat = feat - feat_mean
            pad = padding_mask(lengths)
            feat[pad] = 0.0

        elif self.norm_features == "demean":
            feat_mean = torch.mean(feat, dim=-2, keepdim=True)
            # shape: (batch, 1, feat_dim)
            feat = feat - feat_mean

        elif self.norm_features is None:
            pass
        else:
            raise ValueError(f"Unknown norm_features={self.norm_features}")

        # power=1: energy signal
        # power=2: power signal
        power_scale = 10 if self.power == 2 else 20
        if not torch.is_complex(feat) or self.complex_repr is None:
            assert self.power is not None
            # Real valued signal, power or energy
            if self.apply_log:
                feat = torch.log10(feat) * power_scale

        elif self.complex_repr == "real+imag":
            assert not self.apply_log
            feat = torch.stack([feat.real, feat.imag], dim=1)

        elif self.complex_repr == "mag+phase":
            mag = feat.abs()
            if self.apply_log:
                mag = torch.log10(mag + self.eps) * power_scale

            feat = torch.stack([mag, feat.angle()], dim=1)

        elif self.complex_repr == "real+imag+mag+phase":
            mag = feat.abs()
            if self.apply_log:
                mag = torch.log10(mag + self.eps) * power_scale

            feat = torch.stack([feat.real, feat.imag, mag, feat.angle()], dim=1)

        elif self.complex_repr == "trig":
            mag = feat.abs()
            angle = feat.angle()

            if self.apply_log:
                mag = torch.log10(mag + self.eps) * power_scale

            feat = torch.stack([mag, torch.cos(angle), torch.sin(angle)], dim=1)
        else:
            raise ValueError(f"unknown complex_repr={self.complex_repr}")
        return feat, None
