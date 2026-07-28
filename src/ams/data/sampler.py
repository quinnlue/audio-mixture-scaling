"""Deterministic parquet locality sampler."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from .parquet import AudioSetOpusDataset


class RowGroupShuffleSampler(Sampler[int]):
    """Three-level shuffle, with DDP sharded round-robin by complete batches."""

    def __init__(
        self,
        dataset: AudioSetOpusDataset,
        *,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        batch_size: int = 1,
    ) -> None:
        if seed < 0 or num_replicas < 1 or not 0 <= rank < num_replicas or batch_size < 1:
            raise ValueError("invalid sampler arguments")
        self.dataset, self.seed, self.num_replicas, self.rank, self.batch_size = (
            dataset,
            seed,
            num_replicas,
            rank,
            batch_size,
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return (
            len(self.dataset)
            if self.num_replicas == 1
            else (len(self.dataset) // self.batch_size // self.num_replicas) * self.batch_size
        )

    def _global(self) -> Iterator[int]:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        for file_idx in torch.randperm(len(self.dataset.file_groups), generator=gen).tolist():
            groups = self.dataset.file_groups[file_idx]
            for local_idx in torch.randperm(len(groups), generator=gen).tolist():
                group = self.dataset.group_range(groups[local_idx])
                for offset in torch.randperm(len(group), generator=gen).tolist():
                    yield group.start + offset

    def __iter__(self) -> Iterator[int]:
        if self.num_replicas == 1:
            yield from self._global()
            return
        limit = (len(self.dataset) // self.batch_size // self.num_replicas) * self.num_replicas
        for position, index in enumerate(self._global()):
            batch = position // self.batch_size
            if batch >= limit:
                break
            if batch % self.num_replicas == self.rank:
                yield index
