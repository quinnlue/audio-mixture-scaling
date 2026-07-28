"""CNN frame-target decoder from EAT UFO."""

from __future__ import annotations

import torch

from ams.config import ModelConfig, UFOConfig
from ams.model.patchify import PatchLayout


class UFODecoder(torch.nn.Module):
    def __init__(self, model_cfg: ModelConfig, cfg: UFOConfig) -> None:
        super().__init__()
        d = int(model_cfg.d_model)
        depth = int(getattr(model_cfg, "decoder_depth", getattr(cfg, "decoder_depth", 6)))
        eps = float(getattr(model_cfg, "layer_norm_eps", 1e-6))
        layers: list[torch.nn.Module] = []
        for _ in range(depth):
            layers.extend(
                [
                    torch.nn.Conv2d(d, d, 3, padding=1),
                    torch.nn.LayerNorm(d, eps=eps),
                    torch.nn.GELU(),
                ]
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, tokens: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
        b, n, d = tokens.shape
        if n != layout.num_tokens:
            raise ValueError("token count does not match patch layout")
        x = tokens.reshape(b, layout.time_patches, layout.frequency_patches, d).permute(0, 3, 1, 2)
        for layer in self.layers:
            if isinstance(layer, torch.nn.LayerNorm):
                x = layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = layer(x)
        return x.permute(0, 2, 3, 1).reshape(b, n, d)


__all__ = ["UFODecoder"]
