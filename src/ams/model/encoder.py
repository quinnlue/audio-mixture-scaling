"""EAT transformer encoder."""

from __future__ import annotations

import torch

from ams.config import ModelConfig

from .blocks import AltTransformerBlock


class EATEncoder(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(AltTransformerBlock(cfg) for _ in range(int(cfg.depth)))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        hidden: list[torch.Tensor] | None = [] if return_hidden_states else None
        for block in self.blocks:
            x = block(x, mask)
            if hidden is not None:
                hidden.append(x)
        return x, tuple(hidden) if hidden is not None else None


__all__ = ["EATEncoder"]
