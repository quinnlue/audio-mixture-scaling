"""EAT-compatible Kaldi fbank front end."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ams.config import ModelConfig

from .kaldi import fbank_batch


class KaldiFbank(torch.nn.Module):
    """GPU-friendly Kaldi fbank followed by the released EAT normalization."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.sample_rate = int(getattr(cfg, "sample_rate", 16000))
        self.num_mel_bins = int(getattr(cfg, "num_mel_bins", 128))
        self.target_length = int(getattr(cfg, "target_length", 1024))
        self.norm_mean = float(getattr(cfg, "fbank_norm_mean", -4.268))
        self.norm_std = float(getattr(cfg, "fbank_norm_std", 4.569))

    def forward(
        self, waveform: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must have shape (B, samples), got {tuple(waveform.shape)}")
        if padding_mask is not None:
            if padding_mask.shape != waveform.shape or padding_mask.dtype is not torch.bool:
                raise ValueError("padding_mask must be bool and match waveform")
            valid = padding_mask.to(waveform.dtype)
            count = valid.sum(1, keepdim=True).clamp_min(1)
            waveform = waveform - (waveform * valid).sum(1, keepdim=True) / count
        else:
            waveform = waveform - waveform.mean(1, keepdim=True)
        mel = fbank_batch(
            waveform,
            sample_frequency=self.sample_rate,
            frame_shift=10.0,
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            htk_compat=True,
            use_energy=False,
            window_type="hanning",
        )
        frames = mel.shape[1]
        if frames < self.target_length:
            mel = F.pad(mel, (0, 0, 0, self.target_length - frames))
        else:
            mel = mel[:, : self.target_length]
        return (mel - self.norm_mean) / (self.norm_std * 2.0)


__all__ = ["KaldiFbank"]
