"""Small, dependency-free batch container for pretraining waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WaveformBatch:
    """Fixed-length mono waveforms; ``mask`` is true at valid samples."""

    x: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> WaveformBatch:
        return WaveformBatch(
            self.x.to(device, non_blocking=non_blocking),
            self.mask.to(device, non_blocking=non_blocking),
        )


def train_collate(items: list[dict]) -> WaveformBatch:
    waveform = torch.stack([torch.from_numpy(item["audio"]) for item in items])
    return WaveformBatch(x=waveform, mask=torch.ones_like(waveform, dtype=torch.bool))
