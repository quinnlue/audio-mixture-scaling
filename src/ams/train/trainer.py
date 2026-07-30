"""One explicit EAT/UFO training loop with DDP and exact epoch-boundary resume."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from ams.config import RunConfig, fingerprint, resume_fingerprint
from ams.distributed import DistributedContext, all_gather_object, is_main
from ams.seeding import capture_rng_state

from .callbacks import Callback
from .checkpoint import atomic_torch_save, build_payload, load_checkpoint
from .optim import build_optimizer, build_scheduler


@dataclass
class TrainerState:
    model: torch.nn.Module
    objective: torch.nn.Module
    config: RunConfig
    context: DistributedContext
    epoch: int = 0
    batch_in_epoch: int = 0
    completed_epoch: int = -1
    optimizer_step: int = 0
    global_step: int = 0
    last_loss: float | None = None
    last_grad_norm: float | None = None
    last_val_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.context.world_size

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.config)

    @property
    def resume_fingerprint(self) -> str:
        return resume_fingerprint(self.config)


class Trainer:
    """Train the only supported model/objective pair.

    The objective owns the student model, decoder, mask token, and EMA teacher;
    consequently DDP wraps the objective once and AdamW sees one parameter tree.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        objective: torch.nn.Module,
        train_loader: Any,
        config: RunConfig,
        context: DistributedContext,
        output_dir: str | Path,
        callbacks: list[Callback] | None = None,
        resume: str | Path | None = None,
        evaluator: Callable[[torch.nn.Module, TrainerState], dict[str, float]] | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.output_dir = Path(output_dir)
        self.train_loader = train_loader
        self.callbacks = callbacks or []
        self.evaluator = evaluator
        self.raw_objective = objective.to(context.device)
        self.model = model
        if context.world_size > 1:
            device_ids = [context.local_rank] if context.device.type == "cuda" else None
            self.objective: torch.nn.Module = DistributedDataParallel(
                self.raw_objective,
                device_ids=device_ids,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.objective = self.raw_objective
        accumulation = config.loop.gradient_accumulation_steps
        steps_per_epoch = len(train_loader) // accumulation
        if steps_per_epoch <= 0:
            raise ValueError("train loader must provide at least one optimizer step per epoch")
        self.total_steps = config.optim.total_steps or steps_per_epoch * config.loop.max_epochs
        if config.optim.warmup_steps >= self.total_steps:
            raise ValueError("optim.warmup_steps must be smaller than total optimizer steps")
        self.optimizer = build_optimizer(self.raw_objective, config.optim)
        self.scheduler = build_scheduler(self.optimizer, config.optim, self.total_steps)
        use_scaler = config.loop.precision == "float16" and context.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.state = TrainerState(model, self.raw_objective, config, context)
        if resume is not None:
            (
                self.state.epoch,
                self.state.optimizer_step,
                self.state.batch_in_epoch,
                self.state.completed_epoch,
            ) = load_checkpoint(
                resume,
                model=model,
                objective=self.raw_objective,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if use_scaler else None,
                config=config,
                rank=context.rank,
                world_size=context.world_size,
                data_generator=getattr(self.train_loader, "generator", None),
            )

    def _call(self, method: str, *args: Any) -> None:
        if not is_main():
            return
        for callback in self.callbacks:
            # Callbacks are structural, not subclasses, so a hook only some of them
            # care about (on_export) is simply absent on the rest.
            handler = getattr(callback, method, None)
            if handler is not None:
                handler(self.state, *args)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.context.device.type != "cuda" or self.config.loop.precision == "float32":
            return contextlib.nullcontext()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.loop.precision]
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _checkpoint(self, *, epoch_complete: bool) -> None:
        rank_states = all_gather_object(capture_rng_state())
        if not is_main():
            return
        name = (
            f"epoch_{self.state.epoch + 1:04d}.pt"
            if epoch_complete
            else f"step_{self.state.optimizer_step:08d}.pt"
        )
        path = self.output_dir / "checkpoints" / name
        payload = build_payload(
            model=self.model,
            objective=self.raw_objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.scaler.is_enabled() else None,
            config=self.config,
            epoch=self.state.epoch,
            batch_in_epoch=self.state.batch_in_epoch,
            epoch_complete=epoch_complete,
            optimizer_step=self.state.optimizer_step,
            world_size=self.context.world_size,
            rank_rng_states=rank_states,
            data_generator=getattr(self.train_loader, "generator", None),
        )
        atomic_torch_save(payload, path)
        self._call("on_checkpoint", path)
        prefix = "epoch" if epoch_complete else "step"
        snapshots = sorted(path.parent.glob(f"{prefix}_*.pt"))
        for stale in snapshots[: max(0, len(snapshots) - self.config.checkpoint.keep_last_n)]:
            stale.unlink(missing_ok=True)

    def _evaluate(self) -> None:
        if self.evaluator is None:
            return
        if self.context.world_size > 1:
            torch.distributed.barrier(device_ids=[self.context.local_rank])
        if is_main():
            self.model.eval()
            with torch.no_grad():
                metrics = self.evaluator(self.model, self.state)
            self.state.last_val_metrics = {
                str(name): float(value) for name, value in metrics.items()
            }
            self._call("on_validation", self.state.last_val_metrics)
            self.raw_objective.train()
            teacher = getattr(self.raw_objective, "teacher_model", None)
            if isinstance(teacher, torch.nn.Module):
                teacher.eval()
        if self.context.world_size > 1:
            torch.distributed.barrier(device_ids=[self.context.local_rank])

    def fit(self) -> TrainerState:
        self._call("on_start")
        self.objective.train()
        accumulation = self.config.loop.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        log_started = time.perf_counter()
        log_loss = 0.0
        log_batches = 0
        try:
            for epoch in range(self.state.epoch, self.config.loop.max_epochs):
                self.state.epoch = epoch
                sampler = getattr(self.train_loader, "sampler", None)
                set_epoch = getattr(sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
                iterator = iter(self.train_loader)
                resume_batches = self.state.batch_in_epoch
                for _ in range(resume_batches):
                    try:
                        next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "resume batch offset exceeds the reconstructed epoch"
                        ) from exc
                for batch_index, batch in enumerate(iterator, start=resume_batches):
                    self.state.batch_in_epoch = batch_index + 1
                    self.state.global_step += 1
                    batch = batch.to(self.context.device, non_blocking=True)
                    should_step = (batch_index + 1) % accumulation == 0
                    sync: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
                    if isinstance(self.objective, DistributedDataParallel) and not should_step:
                        sync = self.objective.no_sync()
                    with sync, self._autocast():
                        output = self.objective(batch.x, batch.mask)
                        loss = output.loss / accumulation
                    self.scaler.scale(loss).backward()
                    loss_value = float(output.loss.detach())
                    self.state.last_loss = loss_value
                    log_loss += loss_value
                    log_batches += 1
                    if not should_step:
                        continue
                    self.scaler.unscale_(self.optimizer)
                    clip = self.config.loop.gradient_clip_norm
                    if clip is None:
                        grad_norm = torch.nn.utils.get_total_norm(
                            [p.grad for p in self.raw_objective.parameters() if p.grad is not None]
                        )
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.raw_objective.parameters(),
                            clip,
                        )
                    self.state.last_grad_norm = float(grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    update_teacher = getattr(self.raw_objective, "update_teacher", None)
                    if callable(update_teacher):
                        update_teacher()
                    self.state.optimizer_step += 1
                    if self.state.optimizer_step % self.config.loop.log_every_n_steps == 0:
                        elapsed = max(time.perf_counter() - log_started, 1e-9)
                        metrics = {
                            "train/loss": log_loss / max(log_batches, 1),
                            "train/grad_norm": self.state.last_grad_norm,
                            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                            "train/steps_per_second": log_batches / elapsed,
                        }
                        self._call("on_log", metrics)
                        log_started, log_loss, log_batches = time.perf_counter(), 0.0, 0
                    every = self.config.checkpoint.save_every_n_steps
                    if every and self.state.optimizer_step % every == 0:
                        self._checkpoint(epoch_complete=False)
                    export_every = self.config.checkpoint.export_every_n_steps
                    if export_every and self.state.optimizer_step % export_every == 0:
                        self._call("on_export")
                    if self.state.optimizer_step >= self.total_steps:
                        break
                epoch_complete = self.state.batch_in_epoch >= len(self.train_loader)
                if epoch_complete:
                    self.state.completed_epoch = epoch
                    self.state.batch_in_epoch = 0
                if (
                    epoch_complete
                    and (epoch + 1) % (self.config.checkpoint.save_every_n_epochs or math.inf) == 0
                ):
                    self._checkpoint(epoch_complete=True)
                should_evaluate = (
                    epoch_complete
                    and self.config.eval.enabled
                    and (epoch + 1) % self.config.eval.every_n_epochs == 0
                )
                if should_evaluate:
                    self._evaluate()
                if self.state.optimizer_step >= self.total_steps:
                    break
            self._call("on_end")
            return self.state
        except BaseException as error:
            self._call("on_failure", error)
            raise


__all__ = ["Trainer", "TrainerState"]
