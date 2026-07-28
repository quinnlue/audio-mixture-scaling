"""Frozen run configuration, YAML composition, CLI overrides, and fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")
_MISSING = object()


@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    d_model: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_size: tuple[int, int] = (16, 16)
    num_mel_bins: int = 128
    target_length: int = 1024
    sample_rate: int = 16_000
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    max_time_patches: int = 768
    decoder_dim: int = 384
    decoder_depth: int = 6
    decoder_heads: int = 6

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.depth <= 0 or self.num_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if len(self.patch_size) != 2 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain two positive integers")


@dataclass(kw_only=True, frozen=True)
class UFOConfig:
    mask_ratio: float = 0.8
    block_size: tuple[int, int] = (2, 2)
    num_masks: int = 16
    mask_token_init_scale: float = 0.02
    ema_decay: float = 0.9998
    ema_end_decay: float = 0.99999
    ema_anneal_steps: int = 39_833
    loss_beta: float = 1.0
    average_top_k_layers: int = 12
    instance_norm_target_layer: bool = True
    batch_norm_target_layer: bool = False
    layer_norm_target_layer: bool = False
    layer_norm_targets: bool = True
    instance_norm_targets: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        if self.num_masks <= 0:
            raise ValueError("num_masks must be positive")
        if len(self.block_size) != 2 or min(self.block_size) <= 0:
            raise ValueError("block_size must contain two positive integers")
        if not 0.0 <= self.ema_decay <= self.ema_end_decay <= 1.0:
            raise ValueError("EMA decay must satisfy 0 <= start <= end <= 1")
        if self.average_top_k_layers <= 0:
            raise ValueError("average_top_k_layers must be positive")


@dataclass(kw_only=True, frozen=True)
class DataConfig:
    format: str = "parquet"
    dataset_dir: str | None = None
    blob_dir: str | None = None
    revision: str = "a725d7cf1fea563c6eb9f6127dbd1d75b294668d"
    batch_size: int = 48
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 4
    drop_last: bool = True
    sample_rate: int = 16_000
    clip_samples: int = 160_000
    seed: int = 0
    budget: int | None = None

    def __post_init__(self) -> None:
        if self.format not in {"parquet", "blob"}:
            raise ValueError("data.format must be 'parquet' or 'blob'")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.num_workers == 0 and self.persistent_workers:
            object.__setattr__(self, "persistent_workers", False)
        if self.num_workers == 0 and self.prefetch_factor is not None:
            object.__setattr__(self, "prefetch_factor", None)


@dataclass(kw_only=True, frozen=True)
class OptimConfig:
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    warmup_steps: int = 3_983
    total_steps: int | None = None
    min_lr_scale: float = 0.002

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in [0, 1]")


@dataclass(kw_only=True, frozen=True)
class LoopConfig:
    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    log_every_n_steps: int = 20
    precision: str = "bfloat16"
    compile: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("epoch and accumulation counts must be positive")
        if self.precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("precision must be float32, bfloat16, or float16")


@dataclass(kw_only=True, frozen=True)
class CheckpointConfig:
    save_every_n_epochs: int | None = 1
    save_every_n_steps: int | None = None
    keep_last_n: int = 4
    export_every_n_steps: int | None = None

    def __post_init__(self) -> None:
        if self.keep_last_n <= 0:
            raise ValueError("keep_last_n must be positive")


@dataclass(kw_only=True, frozen=True)
class TrackingConfig:
    campaign_id: str = "local"
    trial_id: str = "trial"
    code_revision: str = "local"
    wandb_project: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str = "disabled"
    hub_repo_id: str | None = None
    hub_private: bool = True

    def __post_init__(self) -> None:
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")


@dataclass(kw_only=True, frozen=True)
class EvalConfig:
    enabled: bool = False
    every_n_epochs: int = 1
    dataset_dir: str | None = None
    tasks: tuple[str, ...] = ()


@dataclass(kw_only=True, frozen=True)
class MixtureConfig:
    dist_id: int | None = None
    distributions: str | None = None
    budget: int | None = None

    def __post_init__(self) -> None:
        if self.dist_id is not None and self.dist_id < 0:
            raise ValueError("dist_id must be non-negative")
        if self.budget is not None and self.budget <= 0:
            raise ValueError("mixture budget must be positive")


@dataclass(kw_only=True, frozen=True)
class RuntimeConfig:
    seed: int = 0
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(kw_only=True, frozen=True)
class RunConfig:
    model: ModelConfig = ModelConfig()
    ufo: UFOConfig = UFOConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    loop: LoopConfig = LoopConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    tracking: TrackingConfig = TrackingConfig()
    eval: EvalConfig = EvalConfig()
    mixture: MixtureConfig = MixtureConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def __post_init__(self) -> None:
        if self.ufo.average_top_k_layers > self.model.depth:
            raise ValueError("average_top_k_layers cannot exceed model.depth")


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return True, non_none[0]
    return False, annotation


def _convert(annotation: Any, value: Any, path: str) -> Any:
    optional, inner = _is_optional(annotation)
    if value is None:
        if optional:
            return None
        raise TypeError(f"{path} may not be null")
    annotation = inner
    origin = get_origin(annotation)
    if hasattr(annotation, "__dataclass_fields__"):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be a mapping")
        return _from_mapping(annotation, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a sequence")
        args = get_args(annotation)
        item_type = args[0] if len(args) == 2 and args[1] is Ellipsis else None
        return tuple(
            _convert(item_type or args[min(i, len(args) - 1)], item, f"{path}[{i}]")
            for i, item in enumerate(value)
        )
    if annotation is float:
        if isinstance(value, bool):
            raise TypeError(f"{path} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path} must be a number") from exc
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return value
    return value


def _from_mapping(cls: type[T], values: dict[str, Any], path: str = "config") -> T:
    dataclass_type = cast(Any, cls)
    known = {field.name for field in fields(dataclass_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown {path} field(s): {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {
        field.name: _convert(hints[field.name], values[field.name], f"{path}.{field.name}")
        for field in fields(dataclass_type)
        if field.name in values
    }
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override path: {dotted!r}")
    cursor = mapping
    for part in parts[:-1]:
        child = cursor.get(part, _MISSING)
        if child is _MISSING:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(
    paths: list[str | Path] | tuple[str | Path, ...],
    overrides: list[str] | tuple[str, ...] = (),
) -> RunConfig:
    """Load and compose YAML files, then apply typed ``key=value`` overrides."""
    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a YAML mapping")
        merged = _deep_merge(merged, payload)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        _set_path(merged, key, yaml.safe_load(raw_value))
    return _from_mapping(RunConfig, merged)


def to_dict(config: RunConfig) -> dict[str, Any]:
    return asdict(config)


def dump_config(config: RunConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(to_dict(config), sort_keys=False),
        encoding="utf-8",
    )


def fingerprint(config: RunConfig, *, ignore_max_epochs: bool = False) -> str:
    payload = to_dict(config)
    if ignore_max_epochs:
        payload["loop"] = {**payload["loop"], "max_epochs": None}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resume_fingerprint(config: RunConfig) -> str:
    """Fingerprint used for resume compatibility; max_epochs may be extended."""
    return fingerprint(config, ignore_max_epochs=True)


def with_total_steps(config: RunConfig, total_steps: int) -> RunConfig:
    return replace(config, optim=replace(config.optim, total_steps=total_steps))


__all__ = [
    "CheckpointConfig",
    "DataConfig",
    "EvalConfig",
    "LoopConfig",
    "MixtureConfig",
    "ModelConfig",
    "OptimConfig",
    "RunConfig",
    "RuntimeConfig",
    "TrackingConfig",
    "UFOConfig",
    "dump_config",
    "fingerprint",
    "load_config",
    "resume_fingerprint",
    "to_dict",
    "with_total_steps",
]
