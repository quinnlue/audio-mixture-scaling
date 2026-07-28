"""Patch tokenization for the fixed EAT fbank grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ams.config import ModelConfig


@dataclass(frozen=True)
class PatchLayout:
    time_patches: int
    frequency_patches: int
    patch_size: int

    @property
    def num_tokens(self) -> int:
        return self.time_patches * self.frequency_patches


class Patchify(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        patch_size = getattr(cfg, "patch_size", (16, 16))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError("EAT requires a square patch_size")
        self.patch_size = int(patch_size[0])

    def layout(self, features: torch.Tensor) -> PatchLayout:
        if features.ndim != 3:
            raise ValueError("features must have shape (B, frames, mel_bins)")
        _, frames, mel_bins = features.shape
        if frames % self.patch_size or mel_bins % self.patch_size:
            raise ValueError("EAT fbank dimensions must be divisible by patch_size")
        return PatchLayout(frames // self.patch_size, mel_bins // self.patch_size, self.patch_size)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, PatchLayout]:
        layout = self.layout(features)
        b, _, _ = features.shape
        p = self.patch_size
        tokens = features.unfold(1, p, p).unfold(2, p, p).contiguous()
        return tokens.reshape(b, layout.num_tokens, p * p), layout


__all__ = ["PatchLayout", "Patchify"]
