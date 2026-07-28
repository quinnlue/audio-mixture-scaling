"""The locked AdamW optimizer and warmup-cosine schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from ams.config import OptimConfig


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split biases and normalization/scale vectors out of weight decay."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: OptimConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


def lr_scale(step: int, *, warmup_steps: int, total_steps: int, minimum: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_scale(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            minimum=config.min_lr_scale,
        ),
    )


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "lr_scale",
    "parameter_groups",
    "trainable_parameters",
]
