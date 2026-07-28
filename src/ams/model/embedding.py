"""EAT patch projection, fixed 2-D sin/cos positions and CLS token."""

from __future__ import annotations

import torch

from ams.config import ModelConfig

from .patchify import PatchLayout


def _standard_stride(gradient: torch.Tensor) -> torch.Tensor:
    """Materialize singleton dimensions with PyTorch's canonical contiguous strides."""
    return torch.empty(
        gradient.shape,
        dtype=gradient.dtype,
        device=gradient.device,
    ).copy_(gradient)


def _sincos_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)))
    values = position.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([values.sin(), values.cos()], dim=1)


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> torch.Tensor:
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    return torch.cat(
        [
            _sincos_1d(embed_dim // 2, grid_w.reshape(-1)),
            _sincos_1d(embed_dim // 2, grid_h.reshape(-1)),
        ],
        dim=1,
    )


class EATEmbedding(torch.nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = int(cfg.d_model)
        patch_size = getattr(cfg, "patch_size", (16, 16))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError("EAT requires a square patch_size")
        patch_size = int(patch_size[0])
        n_mels = int(getattr(cfg, "num_mel_bins", 128))
        max_time = int(getattr(cfg, "max_time_patches", 768))
        self.projection = torch.nn.Linear(patch_size * patch_size, self.d_model)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cls_token.register_hook(_standard_stride)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(self.d_model, (max_time, n_mels // patch_size)).unsqueeze(0),
            persistent=True,
        )
        self.norm = torch.nn.LayerNorm(
            self.d_model, eps=float(getattr(cfg, "layer_norm_eps", 1e-6))
        )
        self.dropout = torch.nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

    def forward(self, tokens: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
        pos_embed = self.pos_embed
        assert isinstance(pos_embed, torch.Tensor)
        if tokens.shape[1] > pos_embed.shape[1]:
            raise ValueError("token sequence exceeds configured positional embedding grid")
        x = self.projection(tokens) + pos_embed[:, : tokens.shape[1]].to(tokens.dtype)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        return self.dropout(self.norm(x))


__all__ = ["EATEmbedding", "build_2d_sincos_pos_embed"]
