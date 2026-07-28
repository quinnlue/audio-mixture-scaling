"""Frame pooling helpers."""

import torch


def pool_scene(x: torch.Tensor, mask: torch.Tensor, *, mode: str = "mean") -> torch.Tensor:
    """Pool `(B,T,D)` frame embeddings using a `True`-for-valid mask."""
    if x.ndim != 3 or mask.shape != x.shape[:2] or torch.any(mask.sum(1) == 0):
        raise ValueError("Invalid frame tensor or mask")
    mean = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    if mode == "mean":
        return mean
    if mode == "mean_max":
        return torch.cat(
            (mean, x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).amax(1)), 1
        )
    raise ValueError(f"Unknown pooling mode: {mode!r}")
