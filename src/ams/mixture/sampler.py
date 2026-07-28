"""Exact cluster-mass replacement sampler."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def load_distribution(path: str | Path, dist_id: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("distributions", [])
    if not 0 <= dist_id < len(rows) or rows[dist_id].get("dist_id") != dist_id:
        raise ValueError(f"distribution {dist_id} not found in {path}")
    return np.asarray(rows[dist_id]["weights"], dtype=np.float64)


class MixtureSampler(Sampler[int]):
    """Draws each example with probability ``mixture[c] / count[c]`` exactly."""

    def __init__(
        self,
        *,
        cluster_index: Sequence[int] | np.ndarray,
        distribution: Sequence[float] | np.ndarray,
        budget: int,
        seed: int,
    ) -> None:
        clusters, weights = (
            np.asarray(cluster_index, dtype=np.int64),
            np.asarray(distribution, dtype=np.float64),
        )
        if clusters.ndim != 1 or not len(clusters):
            raise ValueError("cluster_index must be nonempty and one-dimensional")
        if (
            weights.ndim != 1
            or len(weights) < 2
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError("invalid distribution")
        if np.any(clusters < 0) or np.any(clusters >= len(weights)):
            raise ValueError("cluster id outside distribution")
        if budget < 1 or seed < 0:
            raise ValueError("budget must be positive and seed non-negative")
        counts = np.bincount(clusters, minlength=len(weights))
        empty = np.flatnonzero((weights > 0) & (counts == 0))
        if len(empty):
            raise ValueError(f"distribution assigns mass to empty clusters {empty.tolist()}")
        weights /= weights.sum()
        self._weights = torch.from_numpy(weights[clusters] / counts[clusters])
        self._budget, self._seed, self._epoch = int(budget), int(seed), 0

    def __len__(self) -> int:
        return self._budget

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self._weights,
            self._budget,
            replacement=True,
            generator=torch.Generator().manual_seed(self._seed + self._epoch),
        )
        yield from indices.tolist()
