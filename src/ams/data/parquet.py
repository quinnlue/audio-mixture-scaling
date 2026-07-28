"""Random-access AudioSet Opus parquet dataset."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

DATASET_ID = "danjacobellis/audioset_opus_24kbps"
DATASET_REVISION = "a725d7cf1fea563c6eb9f6127dbd1d75b294668d"
TARGET_SAMPLE_RATE = 16_000
TARGET_SAMPLES = 160_000


def audio_bytes(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray, memoryview)):
        return bytes(value["bytes"])
    raise ValueError(f"unsupported opus value: {type(value).__name__}")


def decode_to_16k_mono(source: bytes | object) -> np.ndarray:
    """Decode raw Opus using the production decoder, resampling only when needed."""
    import soxr
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(source).get_all_samples()
    waveform = samples.data
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    array = waveform.to(torch.float32).contiguous().cpu().numpy()
    if int(samples.sample_rate) != TARGET_SAMPLE_RATE:
        array = soxr.resample(array, int(samples.sample_rate), TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(array, dtype=np.float32)


def fixed_length(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) >= TARGET_SAMPLES:
        return np.ascontiguousarray(waveform[:TARGET_SAMPLES], dtype=np.float32)
    result = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    result[: len(waveform)] = waveform
    return result


class AudioSetOpusDataset(Dataset[dict]):
    """A runner-materialized ``data/*.parquet`` snapshot, cached by row group."""

    def __init__(self, source: str | Path, *, limit: int | None = None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        root = Path(source)
        data = root / "data"
        if not data.is_dir():
            raise FileNotFoundError(f"expected a materialized snapshot at {data}")
        self._parquet_files = tuple(sorted(data.glob("*.parquet")))
        if not self._parquet_files:
            raise FileNotFoundError(f"no parquet shards found under {data}")
        self._group_starts: list[int] = []
        self._group_rows: list[int] = []
        self._group_locations: list[tuple[int, int]] = []
        self._file_groups: list[tuple[int, ...]] = []
        rows = 0
        for file_index, path in enumerate(self._parquet_files):
            groups: list[int] = []
            for group_index in range(pq.ParquetFile(path).num_row_groups):
                remaining = None if limit is None else limit - rows
                if remaining is not None and remaining <= 0:
                    break
                count = int(pq.ParquetFile(path).metadata.row_group(group_index).num_rows)
                if remaining is not None:
                    count = min(count, remaining)
                groups.append(len(self._group_starts))
                self._group_starts.append(rows)
                self._group_rows.append(count)
                self._group_locations.append((file_index, group_index))
                rows += count
            self._file_groups.append(tuple(groups))
            if limit is not None and rows >= limit:
                break
        self._parquet_files = self._parquet_files[: len(self._file_groups)]
        self._length = rows
        self._cached_location: tuple[int, int] | None = None
        self._cached_table: pa.Table | None = None

    def __len__(self) -> int:
        return self._length

    @property
    def file_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._file_groups)

    @property
    def uses_direct_parquet(self) -> bool:
        return True

    def group_range(self, group_index: int) -> range:
        return range(
            self._group_starts[group_index],
            self._group_starts[group_index] + self._group_rows[group_index],
        )

    def __getitem__(self, index: int) -> dict:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        group = bisect_right(self._group_starts, int(index)) - 1
        location = self._group_locations[group]
        if location != self._cached_location:
            file_index, group_index = location
            self._cached_table = pq.ParquetFile(self._parquet_files[file_index]).read_row_group(
                group_index, columns=("path", "label", "opus")
            )
            self._cached_location = location
        if self._cached_table is None:
            raise RuntimeError("row-group cache was not populated")
        row = self._cached_table.slice(int(index) - self._group_starts[group], 1).to_pylist()[0]
        return {
            "audio": fixed_length(decode_to_16k_mono(audio_bytes(row["opus"]))),
            "id": PurePosixPath(row["path"]).stem,
            "path": row["path"],
            "index": int(index),
            "labels": list(row.get("label") or ()),
        }
