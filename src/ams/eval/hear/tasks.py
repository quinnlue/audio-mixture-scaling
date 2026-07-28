"""HEAR task discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

EmbeddingType = Literal["scene", "event"]


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    """Metadata required to evaluate one archive."""

    task_name: str
    root: Path
    archive_path: Path
    metadata: dict[str, Any]
    embedding_type: EmbeddingType
    prediction_type: str
    split_mode: str
    splits: tuple[str, ...]
    metrics: tuple[str, ...]
    nfolds: int | None
    sample_duration: float | None

    @property
    def is_kfold(self) -> bool:
        return self.split_mode in {"new_split_kfold", "presplit_kfold"}


def _spec(path: Path) -> TaskSpec:
    from .archive import read_archive_metadata

    m = read_archive_metadata(path)
    folds = m.get("nfolds")
    splits = tuple(
        map(
            str,
            m.get("splits")
            or (
                [f"fold{i:02d}" for i in range(int(folds))]
                if folds is not None
                else ["train", "valid", "test"]
            ),
        )
    )
    embedding_type = str(m["embedding_type"])
    if embedding_type not in {"scene", "event"}:
        raise ValueError(f"invalid embedding_type in {path}: {embedding_type!r}")
    return TaskSpec(
        task_name=str(m["task_name"]),
        root=path.parent,
        archive_path=path,
        metadata=m,
        embedding_type=cast(EmbeddingType, embedding_type),
        prediction_type=str(m["prediction_type"]),
        split_mode=str(m["split_mode"]),
        splits=splits,
        metrics=tuple(map(str, m.get("evaluation", ()))),
        nfolds=None if folds is None else int(folds),
        sample_duration=m.get("sample_duration"),
    )


def discover_tasks(data_root: Path) -> list[TaskSpec]:
    """Discover and de-duplicate task archives below ``data_root``."""
    if not Path(data_root).exists():
        raise FileNotFoundError(f"HEAR data root does not exist: {data_root}")
    selected: dict[str, TaskSpec] = {}
    for path in sorted(Path(data_root).glob("*.parquet")):
        item = _spec(path)
        previous = selected.get(item.task_name)
        if previous is None or "hear2021" in previous.archive_path.name.lower():
            selected[item.task_name] = item
    return [selected[name] for name in sorted(selected)]
