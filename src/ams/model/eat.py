"""Concrete EAT encoder with a small, training-friendly public API."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ams.config import ModelConfig

from .embedding import EATEmbedding
from .encoder import EATEncoder
from .fbank import KaldiFbank
from .patchify import Patchify, PatchLayout


@dataclass
class EATOutput:
    z: torch.Tensor
    mask: torch.Tensor
    cls: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


class EATModel(torch.nn.Module):
    """EAT waveform encoder.

    ``forward(waveforms, padding_mask)`` accepts ``(B, samples)`` tensors. Masks
    use ``True`` for valid samples/tokens; omitting one means every sample is valid.
    The stage methods are public for UFO so teacher and student share the fbank pass.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fbank = KaldiFbank(cfg)
        self.patchify = Patchify(cfg)
        self.embedding = EATEmbedding(cfg)
        self.encoder = EATEncoder(cfg)

    @property
    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def process(
        self, waveforms: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.fbank(waveforms, padding_mask)

    def tokenize(self, features: torch.Tensor) -> tuple[torch.Tensor, PatchLayout]:
        return self.patchify(features)

    def embed(self, tokens: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
        return self.embedding(tokens, layout)

    def encode_embedded(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> EATOutput:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        z, hidden = self.encoder(x, mask, return_hidden_states=return_hidden_states)
        return EATOutput(z=z, mask=mask, cls=z[:, 0], hidden_states=hidden)

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> EATOutput:
        features = self.process(waveforms, padding_mask)
        tokens, layout = self.tokenize(features)
        x = self.embed(tokens, layout)
        return self.encode_embedded(x, return_hidden_states=return_hidden_states)


__all__ = ["EATModel", "EATOutput"]
