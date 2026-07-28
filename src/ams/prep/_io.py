from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def atomic_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def parquet_files(source: str | Path, *, revision: str | None = None) -> tuple[Path, ...]:
    path = Path(source)
    if path.is_file():
        return (path,)
    if path.is_dir():
        files = tuple(sorted(path.glob("data/*.parquet")) or sorted(path.rglob("*.parquet")))
        if files:
            return files
        raise FileNotFoundError(f"no parquet files under {path}")
    from huggingface_hub import snapshot_download

    snap = Path(
        snapshot_download(
            repo_id=str(source),
            repo_type="dataset",
            revision=revision,
            allow_patterns=["*.parquet", "**/*.parquet"],
        )
    )
    return parquet_files(snap)


def iter_parquet_batches(
    files: tuple[Path, ...], *, columns: tuple[str, ...], batch_size: int
) -> Iterator[pa.RecordBatch]:
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        yield from parquet.iter_batches(batch_size=batch_size, columns=list(columns))
