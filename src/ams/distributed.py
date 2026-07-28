"""Small torch.distributed boundary used by training and artifact code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool


def init_distributed(timeout_minutes: int = 30) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        initialized_here = True
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def cleanup(context: DistributedContext | None = None) -> None:
    if dist.is_initialized() and (context is None or context.initialized_here):
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_gather_object(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


__all__ = [
    "DistributedContext",
    "all_gather_object",
    "barrier",
    "cleanup",
    "init_distributed",
    "is_main",
    "rank",
    "world_size",
]
