"""Configuration and result types for HEAR evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

PoolingMode = Literal["mean", "mean_max"]


@dataclass(frozen=True, kw_only=True)
class HearEvalConfig:
    """Local HEAR benchmark settings.

    Audio archives are self-contained Parquet files.  Results and optional
    embedding caches are scoped by ``model_id`` below the configured roots.
    """

    data_root: Path = Path("data/hear_parquet")
    results_dir: Path = Path("results/hear")
    cache_dir: Path | None = None
    tasks: tuple[str, ...] | None = None
    model_id: str = "model"
    sample_rate: int = 16_000
    batch_size: int = 256
    amp: bool = True
    num_workers: int = 16
    device: str = "auto"
    pooling: PoolingMode = "mean"
    ks: tuple[int, ...] = (1, 5, 10)
    probe_epochs: int = 20
    event_epochs: int = 20
    event_hidden_dim: int = 256
    event_dropout: float = 0.1
    event_lr: float = 1e-3
    event_frame_batch_size: int = 65_536
    limit: int | None = None
    max_folds: int | None = None
    run_knn: bool = True
    run_linear: bool = True
    reuse_cache: bool = True
    progress: bool = False
    source: str = "final_z"
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.tasks is not None:
            object.__setattr__(self, "tasks", tuple(map(str, self.tasks)))
        object.__setattr__(self, "ks", tuple(map(int, self.ks)))
        if self.pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unknown pooling mode: {self.pooling!r}")

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir if self.cache_dir is not None else self.results_dir / "cache"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("data_root", "results_dir", "cache_dir"):
            if result[name] is not None:
                result[name] = str(result[name])
        return result


@dataclass(frozen=True, kw_only=True)
class HearEvalResult:
    """Aggregated task payloads and scalar summaries for one model."""

    model_id: str
    task_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
