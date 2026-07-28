"""Inverse block masking used by the EAT UFO student."""

from __future__ import annotations

import math

import torch


def num_visible_tokens(*, time_patches: int, frequency_patches: int, mask_ratio: float) -> int:
    return max(1, int(round(time_patches * frequency_patches * (1.0 - float(mask_ratio)))))


def inverse_block_masking(
    *,
    batch_size: int,
    time_patches: int,
    frequency_patches: int,
    mask_ratio: float,
    block_size: tuple[int, int],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    n = time_patches * frequency_patches
    k = num_visible_tokens(
        time_patches=time_patches, frequency_patches=frequency_patches, mask_ratio=mask_ratio
    )
    bt, bf = min(block_size[0], time_patches), min(block_size[1], frequency_patches)
    cells = bt * bf
    blocks = 4 * math.ceil(k / cells) + 16
    st = torch.randint(
        max(time_patches - bt + 1, 1), (batch_size, blocks), device=device, generator=generator
    )
    sf = torch.randint(
        max(frequency_patches - bf + 1, 1), (batch_size, blocks), device=device, generator=generator
    )
    ot, of = torch.meshgrid(
        torch.arange(bt, device=device), torch.arange(bf, device=device), indexing="ij"
    )
    indices = (
        (st[..., None] + ot.reshape(1, 1, -1)) * frequency_patches
        + sf[..., None]
        + of.reshape(1, 1, -1)
    ).reshape(batch_size, -1)
    order = (
        torch.arange(blocks, device=device, dtype=torch.float32)
        .view(1, blocks, 1)
        .expand(batch_size, -1, cells)
        .reshape(batch_size, -1)
    )
    rank = torch.full((batch_size, n), float(blocks + 1), device=device)
    rank.scatter_reduce_(1, indices, order, reduce="amin", include_self=True)
    ids_shuffle = torch.argsort(
        rank + torch.rand((batch_size, n), device=device, generator=generator), dim=1
    )
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :k]
    visible = torch.zeros((batch_size, n), dtype=torch.bool, device=device)
    visible.scatter_(1, ids_keep, True)
    return visible, ~visible, ids_restore, ids_keep


def restore_patch_tokens(
    visible_tokens: torch.Tensor, *, mask_token: torch.Tensor, ids_restore: torch.Tensor
) -> torch.Tensor:
    b, n = ids_restore.shape
    missing = n - visible_tokens.shape[1]
    full = torch.cat([visible_tokens, mask_token.reshape(1, 1, -1).expand(b, missing, -1)], dim=1)
    return torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]))


__all__ = ["inverse_block_masking", "num_visible_tokens", "restore_patch_tokens"]
