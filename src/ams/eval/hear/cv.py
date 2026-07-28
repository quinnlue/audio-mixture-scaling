"""Cross-validation split helpers."""

from dataclasses import dataclass

import numpy as np

from .data import HearSample
from .tasks import TaskSpec


@dataclass(frozen=True)
class SplitIndices:
    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def iter_splits(
    task: TaskSpec, samples: list[HearSample], *, max_folds: int | None = None
) -> list[SplitIndices]:
    """Build HEAR's k-fold or train/valid/test probe splits."""
    if task.is_kfold:
        folds = sorted({s.fold for s in samples if s.fold is not None})[: max_folds or None]
        return [
            SplitIndices(
                f"fold{fold:02d}",
                np.asarray([i for i, s in enumerate(samples) if s.fold != fold]),
                np.empty(0, dtype=int),
                np.asarray([i for i, s in enumerate(samples) if s.fold == fold]),
            )
            for fold in folds
        ]

    def index(split: str) -> np.ndarray:
        return np.asarray([i for i, sample in enumerate(samples) if sample.split == split])

    return [SplitIndices("trainvaltest", index("train"), index("valid"), index("test"))]
