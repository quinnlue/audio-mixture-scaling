"""AltTransformer blocks used by EAT/data2vec."""

from __future__ import annotations

import torch

from ams.config import ModelConfig


class AltAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.proj_drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else mask[:, None, None, :]
        out = torch.nn.functional.scaled_dot_product_attention(
            qkv[0],
            qkv[1],
            qkv[2],
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, d)))


class AltTransformerBlock(torch.nn.Module):
    """Post-norm EAT AltBlock (the released recipe's layout)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = int(cfg.d_model)
        eps = float(getattr(cfg, "layer_norm_eps", 1e-6))
        dropout = float(getattr(cfg, "dropout", 0.0))
        self.norm1, self.norm2 = torch.nn.LayerNorm(d, eps=eps), torch.nn.LayerNorm(d, eps=eps)
        self.attn = AltAttention(d, int(cfg.num_heads), dropout)
        self.fc1 = torch.nn.Linear(d, d * int(getattr(cfg, "mlp_ratio", 4)))
        self.fc2 = torch.nn.Linear(d * int(getattr(cfg, "mlp_ratio", 4)), d)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(x, mask))
        residual = self.norm1(x)
        x = self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(residual))))
        return self.norm2(residual + self.drop(x))


__all__ = ["AltAttention", "AltTransformerBlock"]
