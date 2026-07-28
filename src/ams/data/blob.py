"""Random-access mmap-backed Opus dataset used by RegMix proxy runs."""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import BinaryIO

import numpy as np
from torch.utils.data import Dataset

from .parquet import decode_to_16k_mono, fixed_length


class OpusBlobDataset(Dataset[dict]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blob_path = self.root / "opus_blob.bin"
        offsets_path, clusters_path = (
            self.root / "opus_offsets.npy",
            self.root / "cluster_index.npy",
        )
        for path in (self.blob_path, offsets_path, clusters_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.cluster_index = np.load(clusters_path, mmap_mode="r")
        if self.offsets.ndim != 1 or self.offsets.dtype != np.int64 or len(self.offsets) == 0:
            raise ValueError("offsets must be a nonempty int64 vector")
        if self.cluster_index.ndim != 1 or len(self.cluster_index) != len(self.offsets):
            raise ValueError("cluster index must align with offsets")
        self._blob_size = self.blob_path.stat().st_size
        if (
            int(self.offsets[0]) != 0
            or np.any(np.diff(self.offsets) <= 0)
            or int(self.offsets[-1]) >= self._blob_size
        ):
            raise ValueError("invalid blob offsets")
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _ensure_open(self) -> mmap.mmap:
        if self._mmap is None:
            file_handle = self.blob_path.open("rb")
            self._file = file_handle
            self._mmap = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def encoded_bytes(self, index: int) -> bytes:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1]) if index + 1 < len(self) else self._blob_size
        return bytes(self._ensure_open()[start:end])

    def __getitem__(self, index: int) -> dict:
        return {
            "audio": fixed_length(decode_to_16k_mono(self.encoded_bytes(index))),
            "id": str(index),
            "path": str(index),
            "index": int(index),
            "cluster": int(self.cluster_index[index]),
            "labels": [],
        }

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        for array in (getattr(self, "offsets", None), getattr(self, "cluster_index", None)):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_mmap"] = None
        return state

    def __del__(self) -> None:
        self.close()
