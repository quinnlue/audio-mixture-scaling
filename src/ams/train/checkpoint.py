"""Atomic checkpoints with exact RNG and distributed resume state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch

from ams.config import RunConfig, resume_fingerprint, to_dict
from ams.seeding import capture_rng_state, restore_rng_state


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def build_payload(
    *,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    config: RunConfig,
    epoch: int,
    batch_in_epoch: int,
    epoch_complete: bool,
    optimizer_step: int,
    world_size: int,
    rank_rng_states: list[dict[str, Any]],
    data_generator: torch.Generator | None,
) -> dict[str, Any]:
    raw_model = cast(torch.nn.Module, getattr(model, "module", model))
    objective_generator = getattr(objective, "generator", None)
    return {
        "checkpoint_schema_version": 1,
        "model_state": raw_model.state_dict(),
        "objective_state": objective.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch_in_epoch": 0 if epoch_complete else batch_in_epoch,
        "completed_epoch": epoch if epoch_complete else epoch - 1,
        "next_epoch": epoch + 1 if epoch_complete else epoch,
        "optimizer_step": optimizer_step,
        "world_size": world_size,
        "rng": capture_rng_state(),
        "rank_rng_states": rank_rng_states,
        "data_generator_state": (
            data_generator.get_state() if data_generator is not None else None
        ),
        "objective_generator_state": (
            objective_generator.get_state()
            if isinstance(objective_generator, torch.Generator)
            else None
        ),
        "config": to_dict(config),
        "config_fingerprint": resume_fingerprint(config),
    }


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    config: RunConfig,
    rank: int,
    world_size: int,
    data_generator: torch.Generator | None,
) -> tuple[int, int, int, int]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("config_fingerprint") != resume_fingerprint(config):
        raise ValueError("resume config fingerprint mismatch (only loop.max_epochs may change)")
    if int(payload["world_size"]) != world_size:
        raise ValueError(
            f"resume world size mismatch: checkpoint={payload['world_size']} current={world_size}"
        )
    raw_model = cast(torch.nn.Module, getattr(model, "module", model))
    raw_model.load_state_dict(payload["model_state"])
    objective.load_state_dict(payload["objective_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    states = payload.get("rank_rng_states") or [payload["rng"]]
    restore_rng_state(states[rank])
    if data_generator is not None and payload.get("data_generator_state") is not None:
        data_generator.set_state(payload["data_generator_state"])
    objective_generator = getattr(objective, "generator", None)
    if (
        isinstance(objective_generator, torch.Generator)
        and payload.get("objective_generator_state") is not None
    ):
        objective_generator.set_state(payload["objective_generator_state"])
    next_epoch = int(payload["next_epoch"])
    return (
        next_epoch,
        int(payload["optimizer_step"]),
        int(payload.get("batch_in_epoch", 0)),
        int(payload.get("completed_epoch", next_epoch - 1)),
    )


__all__ = ["atomic_torch_save", "build_payload", "load_checkpoint"]
