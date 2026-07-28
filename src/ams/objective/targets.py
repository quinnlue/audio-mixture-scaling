"""data2vec/EAT teacher-target construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_targets(hidden_states: tuple[torch.Tensor, ...], cfg: object) -> torch.Tensor:
    """Normalize selected teacher layers then average them, in data2vec order.

    The released EAT AS2M recipe enables per-layer instance norm, averages all
    12 layers, and enables post-average layer norm.
    """
    if not hidden_states:
        raise ValueError("teacher hidden states must not be empty")
    top_k = int(getattr(cfg, "average_top_k_layers", len(hidden_states)))
    if top_k < 1 or top_k > len(hidden_states):
        raise ValueError("average_top_k_layers must be between 1 and encoder depth")
    normalized: list[torch.Tensor] = []
    for layer in hidden_states[-top_k:]:
        if bool(getattr(cfg, "instance_norm_target_layer", False)):
            layer = F.instance_norm(layer.transpose(1, 2)).transpose(1, 2)
        elif bool(getattr(cfg, "batch_norm_target_layer", False)):
            layer = F.batch_norm(
                layer.transpose(1, 2), running_mean=None, running_var=None, training=True
            ).transpose(1, 2)
        elif bool(getattr(cfg, "layer_norm_target_layer", False)):
            layer = F.layer_norm(layer, (layer.shape[-1],))
        normalized.append(layer)
    targets = torch.stack(normalized).mean(0)
    if bool(getattr(cfg, "layer_norm_targets", False)):
        targets = F.layer_norm(targets, (targets.shape[-1],))
    elif bool(getattr(cfg, "instance_norm_targets", False)):
        targets = F.instance_norm(targets.transpose(1, 2)).transpose(1, 2)
    return targets


__all__ = ["make_targets"]
