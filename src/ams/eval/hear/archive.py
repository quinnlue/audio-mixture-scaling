"""Self-contained Parquet HEAR task archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

TASK_METADATA_KEY = b"hear_task_metadata"
LABEL_VOCAB_KEY = b"hear_label_vocabulary"
FORMAT_VERSION = "1"


def archive_schema(*, task_metadata: dict[str, Any], vocab_pairs: list[list[Any]]) -> pa.Schema:
    """Return the archive schema with metadata embedded in its footer."""
    return pa.schema(
        [
            ("clip_id", pa.string()),
            ("split", pa.string()),
            ("value_json", pa.string()),
            ("num_frames", pa.int32()),
            ("sample_rate", pa.int32()),
            ("audio", pa.large_binary()),
        ],
        metadata={
            TASK_METADATA_KEY: json.dumps(task_metadata).encode(),
            LABEL_VOCAB_KEY: json.dumps(vocab_pairs).encode(),
            b"hear_archive_version": FORMAT_VERSION.encode(),
        },
    )


def write_task_archive(
    path: Path,
    *,
    clip_ids: list[str],
    splits: list[str],
    value_jsons: list[str],
    num_frames: list[int],
    sample_rates: list[int],
    audio: list[bytes],
    task_metadata: dict[str, Any],
    vocab_pairs: list[list[Any]],
    row_group_size: int = 2048,
) -> None:
    """Write a portable HEAR archive used by the offline preparation command."""
    schema = archive_schema(task_metadata=task_metadata, vocab_pairs=vocab_pairs)
    table = pa.table(
        {
            "clip_id": clip_ids,
            "split": splits,
            "value_json": value_jsons,
            "num_frames": num_frames,
            "sample_rate": sample_rates,
            "audio": audio,
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression={
            "clip_id": "ZSTD",
            "split": "ZSTD",
            "value_json": "ZSTD",
            "num_frames": "ZSTD",
            "sample_rate": "ZSTD",
            "audio": "NONE",
        },
        row_group_size=row_group_size,
    )


def read_archive_metadata(path: Path) -> dict[str, Any]:
    """Read task metadata without materialising audio."""
    metadata = pq.read_schema(path).metadata or {}
    if TASK_METADATA_KEY not in metadata:
        raise ValueError(f"{path} is not a HEAR task archive")
    return json.loads(metadata[TASK_METADATA_KEY])


class TaskArchive:
    """In-memory random-access view over one task archive."""

    def __init__(
        self, path: Path, table: pa.Table, metadata: dict[str, Any], vocab: dict[str | int, str]
    ) -> None:
        self.path, self._table, self.metadata, self.vocab = path, table, metadata, vocab
        self._columns = {
            name: table.column(name).to_pylist()
            for name in ("clip_id", "split", "value_json", "num_frames", "sample_rate", "audio")
        }

    @classmethod
    def open(cls, path: Path) -> TaskArchive:
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        task = json.loads(metadata[TASK_METADATA_KEY])
        vocab = {key: str(value) for key, value in json.loads(metadata.get(LABEL_VOCAB_KEY, b"[]"))}
        return cls(Path(path), table, task, vocab)

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    def clip_id(self, row: int) -> str:
        return str(self._columns["clip_id"][row])

    def split(self, row: int) -> str:
        return str(self._columns["split"][row])

    def value(self, row: int) -> Any:
        return json.loads(self._columns["value_json"][row])

    def num_frames(self, row: int) -> int:
        return int(self._columns["num_frames"][row])

    def sample_rate(self, row: int) -> int:
        return int(self._columns["sample_rate"][row])

    def audio_bytes(self, row: int) -> bytes:
        return self._columns["audio"][row]
