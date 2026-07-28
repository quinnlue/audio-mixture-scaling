"""Atomic runner-compatible local artifact contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

ARTIFACT_SCHEMA_VERSION = 1


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class LocalArtifacts:
    """Write exactly the paths consumed by ``train-runner``."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.result_dir = self.output_dir / "results"
        self.export_dir = self.output_dir / "exports"
        self.metadata = dict(metadata)

    def _event(self, event: str, **values: Any) -> None:
        append_jsonl(
            self.output_dir / "events.jsonl",
            {"time_unix": time.time(), "event": event, **values},
        )

    def write_status(self, status: str, **values: Any) -> None:
        atomic_json(
            self.output_dir / "status.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "status": status,
                "updated_unix": time.time(),
                **self.metadata,
                **values,
            },
        )
        self._event("status", status=status, **values)

    def on_start(self, state: Any) -> None:
        for directory in (self.checkpoint_dir, self.result_dir, self.export_dir):
            directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.output_dir / "run.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "config_fingerprint": state.config_fingerprint,
                **self.metadata,
            },
        )
        self.write_status("running", optimizer_step=state.optimizer_step, next_epoch=state.epoch)

    def on_log(self, state: Any, metrics: dict[str, float]) -> None:
        self._event("metrics", optimizer_step=state.optimizer_step, metrics=metrics)

    def on_checkpoint(self, state: Any, path: Path) -> None:
        resume_path = self.checkpoint_dir / "resume.pt"
        atomic_link_or_copy(path, resume_path)
        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "checkpoint": resume_path.name,
            "sha256": sha256(resume_path),
            "size": resume_path.stat().st_size,
            "completed_epoch": state.completed_epoch,
            "next_epoch": (
                state.epoch + 1 if state.completed_epoch == state.epoch else state.epoch
            ),
            "batch_in_epoch": state.batch_in_epoch,
            "optimizer_step": state.optimizer_step,
            "world_size": state.world_size,
            "config_fingerprint": state.resume_fingerprint,
            **self.metadata,
        }
        atomic_json(self.checkpoint_dir / "resume.json", metadata)
        self._event("checkpoint_ready", path=str(resume_path), sha256=metadata["sha256"])

    def write_result(self, state: Any, metrics: dict[str, float] | None = None) -> None:
        row = {
            "optimizer_step": state.optimizer_step,
            "metrics": metrics or {},
            "inventory_only": not bool(metrics),
            **self.metadata,
        }
        append_jsonl(self.result_dir / "hear_metrics.jsonl", row)

    def export_model(self, state: Any) -> Path:
        model = state.model.module if hasattr(state.model, "module") else state.model
        destination_dir = self.export_dir / f"step_{state.optimizer_step:08d}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "model.safetensors"
        temporary = destination.with_suffix(".safetensors.tmp")
        tensors = {
            name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
        }
        save_file(tensors, str(temporary))
        os.replace(temporary, destination)
        return destination

    def on_validation(self, state: Any, metrics: dict[str, float]) -> None:
        self.write_result(state, metrics)
        self.export_model(state)
        self._event(
            "evaluation_ready",
            optimizer_step=state.optimizer_step,
            metrics=metrics,
        )

    def on_end(self, state: Any) -> None:
        if not (self.result_dir / "hear_metrics.jsonl").is_file():
            self.write_result(state)
        final_export = self.export_dir / f"step_{state.optimizer_step:08d}" / "model.safetensors"
        if not final_export.is_file():
            self.export_model(state)
        resume = self.checkpoint_dir / "resume.pt"
        if resume.is_file():
            atomic_link_or_copy(resume, self.checkpoint_dir / "final.pt")
        self.write_status(
            "complete",
            completed_epoch=state.completed_epoch,
            optimizer_step=state.optimizer_step,
            last_loss=state.last_loss,
        )

    def on_failure(self, state: Any, error: BaseException) -> None:
        self.write_status(
            "failed",
            optimizer_step=state.optimizer_step,
            error_type=type(error).__name__,
            error=str(error),
        )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "LocalArtifacts",
    "append_jsonl",
    "atomic_json",
    "sha256",
]
