"""Waveform embedding extraction and cache management."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import HearEvalConfig
from .data import HearSample, LabelMap, manifest_fingerprint
from .tasks import TaskSpec


@dataclass(frozen=True, kw_only=True)
class SceneExtraction:
    embeddings: np.ndarray
    labels: np.ndarray
    raw_labels: np.ndarray
    paths: tuple[str, ...]
    cache_hit: bool


@dataclass(frozen=True, kw_only=True)
class FrameClip:
    frames: np.ndarray
    timestamps_ms: np.ndarray
    duration_ms: float


@dataclass(frozen=True, kw_only=True)
class FrameExtraction:
    clips: tuple[FrameClip, ...]
    paths: tuple[str, ...]
    cache_hit: bool


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available."""
    return torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )


def _fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cache(
    config: HearEvalConfig,
    task: TaskSpec,
    samples: list[HearSample],
    labels: LabelMap,
    model: torch.nn.Module,
    kind: str,
) -> tuple[Path, dict[str, object]]:
    root = config.resolved_cache_dir / config.model_id / task.task_name
    return root, {
        "kind": kind,
        "manifest": manifest_fingerprint(samples, labels),
        "model": _fingerprint(model),
        "pooling": config.pooling,
        "sample_rate": config.sample_rate,
        "source": config.source,
    }


def _forward(
    model: torch.nn.Module, chunks: list[np.ndarray], device: torch.device, amp: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    size = max(len(chunk) for chunk in chunks)
    waveforms = torch.zeros((len(chunks), size), dtype=torch.float32, device=device)
    padding = torch.zeros((len(chunks), size), dtype=torch.bool, device=device)
    for row, chunk in enumerate(chunks):
        waveforms[row, : len(chunk)] = torch.from_numpy(chunk).to(device)
        padding[row, : len(chunk)] = True
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if amp and device.type == "cuda"
        else nullcontext()
    )
    with context:
        output = model(waveforms, padding)
    return output.z.float(), output.mask


def _extract(
    model: torch.nn.Module, samples: list[HearSample], config: HearEvalConfig, device: torch.device
) -> list[tuple[np.ndarray, np.ndarray]]:
    waveforms = [sample.source.read(config.sample_rate) for sample in samples]
    output: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(samples)
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(samples), config.batch_size):
            batch = waveforms[start : start + config.batch_size]
            z, mask = _forward(model, batch, device, config.amp)
            for row, waveform in enumerate(batch):
                frames = z[row][mask[row]].cpu().numpy().astype(np.float32)
                timestamps = np.arange(len(frames), dtype=np.float32) * (
                    len(waveform) / max(len(frames), 1) / config.sample_rate * 1000
                )
                output[start + row] = (frames, timestamps)
    return [item for item in output if item is not None]


def extract_scene(
    model: torch.nn.Module,
    task: TaskSpec,
    samples: list[HearSample],
    label_map: LabelMap,
    config: HearEvalConfig,
    *,
    device: torch.device,
) -> SceneExtraction:
    """Extract mean or mean/max pooled scene embeddings, reusing a valid cache."""
    root, manifest = _cache(config, task, samples, label_map, model, "scene")
    data_path, meta_path = root / "scene.npz", root / "scene_manifest.json"
    if (
        config.reuse_cache
        and data_path.exists()
        and meta_path.exists()
        and json.loads(meta_path.read_text()) == manifest
    ):
        data = np.load(data_path, allow_pickle=True)
        return SceneExtraction(
            embeddings=data["embeddings"],
            labels=data["labels"],
            raw_labels=data["raw_labels"],
            paths=tuple(data["paths"].tolist()),
            cache_hit=True,
        )
    frames = _extract(model, samples, config, device)
    vectors = [
        np.concatenate((item[0].mean(0), item[0].max(0)))
        if config.pooling == "mean_max"
        else item[0].mean(0)
        for item in frames
    ]
    result = SceneExtraction(
        embeddings=np.asarray(vectors, dtype=np.float32),
        labels=np.asarray([s.label for s in samples], dtype=np.int64),
        raw_labels=np.asarray([s.label_raw for s in samples], dtype=object),
        paths=tuple(s.source.uid for s in samples),
        cache_hit=False,
    )
    if config.reuse_cache:
        root.mkdir(parents=True, exist_ok=True)
        # NumPy 2.4 stubs incorrectly constrain all named arrays to the
        # ``allow_pickle`` boolean type.
        np.savez(  # type: ignore[arg-type]
            data_path,
            embeddings=result.embeddings,
            labels=result.labels,
            raw_labels=result.raw_labels,
            paths=np.asarray(result.paths),
        )
        meta_path.write_text(json.dumps(manifest, sort_keys=True))
    return result


def extract_frames(
    model: torch.nn.Module,
    task: TaskSpec,
    samples: list[HearSample],
    label_map: LabelMap,
    config: HearEvalConfig,
    *,
    device: torch.device,
) -> FrameExtraction:
    """Extract variable-length frame embeddings for an event task."""
    root, manifest = _cache(config, task, samples, label_map, model, "frames")
    data_path, meta_path = root / "frames.npz", root / "frames_manifest.json"
    if (
        config.reuse_cache
        and data_path.exists()
        and meta_path.exists()
        and json.loads(meta_path.read_text()) == manifest
    ):
        data = np.load(data_path, allow_pickle=True)
        clips = tuple(
            FrameClip(
                frames=data[f"frames_{index}"],
                timestamps_ms=data[f"timestamps_{index}"],
                duration_ms=float(data["durations_ms"][index]),
            )
            for index in range(int(data["num_clips"][0]))
        )
        return FrameExtraction(
            clips=clips,
            paths=tuple(data["paths"].tolist()),
            cache_hit=True,
        )
    rows = _extract(model, samples, config, device)
    result = FrameExtraction(
        clips=tuple(
            FrameClip(
                frames=frames,
                timestamps_ms=timestamps,
                duration_ms=sample.source.frame_count() / config.sample_rate * 1000.0,
            )
            for (frames, timestamps), sample in zip(rows, samples, strict=True)
        ),
        paths=tuple(s.source.uid for s in samples),
        cache_hit=False,
    )
    if config.reuse_cache:
        root.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: value
            for index, clip in enumerate(result.clips)
            for key, value in {
                f"frames_{index}": clip.frames,
                f"timestamps_{index}": clip.timestamps_ms,
            }.items()
        }
        # NumPy 2.4's ``savez`` stub treats arbitrary named arrays as booleans.
        np.savez(
            data_path,
            **arrays,  # type: ignore[arg-type]
            durations_ms=np.asarray([clip.duration_ms for clip in result.clips], dtype=np.float32),
            paths=np.asarray(result.paths),
            num_clips=np.asarray([len(result.clips)], dtype=np.int64),
        )
        meta_path.write_text(json.dumps(manifest, sort_keys=True))
    return result
