"""Validate representative AudioSet shards with the production decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .blob import OpusBlobDataset
from .parquet import audio_bytes, decode_to_16k_mono


def validate_shards(dataset_dir: Path) -> list[dict]:
    if (dataset_dir / "opus_blob.bin").is_file():
        dataset = OpusBlobDataset(dataset_dir)
        try:
            indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
            results = []
            for index in indices:
                waveform = dataset[index]["audio"]
                if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                    raise ValueError(f"invalid decoded waveform at blob index {index}")
                results.append(
                    {
                        "index": index,
                        "samples": int(waveform.size),
                        "dtype": str(waveform.dtype),
                    }
                )
            return results
        finally:
            dataset.close()
    shards = sorted((dataset_dir / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {dataset_dir / 'data'}")
    results = []
    for index in sorted({0, len(shards) // 2, len(shards) - 1}):
        shard = shards[index]
        table = pq.ParquetFile(shard).read_row_group(0, columns=("path", "opus"))
        if not table.num_rows:
            raise ValueError(f"first row group is empty: {shard}")
        row = table.slice(0, 1).to_pylist()[0]
        waveform = decode_to_16k_mono(audio_bytes(row["opus"]))
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"invalid decoded waveform from {shard}")
        results.append(
            {
                "shard": shard.name,
                "path": row["path"],
                "samples": int(waveform.size),
                "dtype": str(waveform.dtype),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_shards(args.dataset_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
