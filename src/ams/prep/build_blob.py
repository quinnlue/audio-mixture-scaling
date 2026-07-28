"""Pack source-order Opus payloads and cluster labels into mmap-ready artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ams.data.parquet import audio_bytes, decode_to_16k_mono

from ._io import atomic_json, iter_parquet_batches, parquet_files


def _labels(path: str | Path) -> tuple[dict[str, int], int]:
    path = Path(path)
    path = path / "cluster_labels.parquet" if path.is_dir() else path
    table = pq.read_table(path, columns=["path", "cluster"])
    paths = table.column("path").to_pylist()
    clusters = np.asarray(table.column("cluster").to_numpy(), dtype=np.int64)
    if (
        len(paths) != len(set(paths))
        or not len(paths)
        or np.any(clusters < 0)
        or np.any(clusters > 127)
    ):
        raise ValueError("invalid cluster labels")
    return dict(zip(paths, clusters.tolist(), strict=True)), int(clusters.max()) + 1


def build_blob(
    *,
    audio_source: str | Path,
    cluster_labels: str | Path,
    output_dir: str | Path,
    audio_revision: str | None = None,
    batch_size: int = 2048,
    verify_samples: int = 3,
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob, offsets_path, clusters_path = (
        out / "opus_blob.bin",
        out / "opus_offsets.npy",
        out / "cluster_index.npy",
    )
    if not overwrite and any(p.exists() for p in (blob, offsets_path, clusters_path)):
        raise FileExistsError("blob outputs already exist; pass --overwrite")
    mapping, nclusters = _labels(cluster_labels)
    files = parquet_files(audio_source, revision=audio_revision)
    offsets = []
    clusters = []
    seen = set()
    tmp = blob.with_suffix(".bin.tmp")
    total = 0
    try:
        with tmp.open("wb") as fp:
            for batch in iter_parquet_batches(
                files, columns=("path", "opus"), batch_size=batch_size
            ):
                for path, opus in zip(
                    batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
                ):
                    if path in seen or path not in mapping:
                        raise ValueError(f"missing/duplicate path {path!r}")
                    payload = audio_bytes(opus)
                    offsets.append(total)
                    clusters.append(mapping[path])
                    fp.write(payload)
                    total += len(payload)
                    seen.add(path)
            fp.flush()
            os.fsync(fp.fileno())
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if len(seen) != len(mapping):
        tmp.unlink(missing_ok=True)
        raise ValueError("not all labels appeared in source")
    # Decode checks are deliberately after packing, keeping the hot loop byte-copy only.
    with tmp.open("rb") as fp:
        for index in np.linspace(0, len(offsets) - 1, min(verify_samples, len(offsets)), dtype=int):
            fp.seek(offsets[int(index)])
            end = offsets[int(index) + 1] if int(index) + 1 < len(offsets) else total
            decode_to_16k_mono(fp.read(end - offsets[int(index)]))
    np.save(out / "opus_offsets.tmp.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "cluster_index.tmp.npy", np.asarray(clusters, dtype=np.int8))
    tmp.replace(blob)
    (out / "opus_offsets.tmp.npy").replace(offsets_path)
    (out / "cluster_index.tmp.npy").replace(clusters_path)
    metadata = {
        "schema_version": 1,
        "audio_source": str(audio_source),
        "audio_revision": audio_revision,
        "cluster_labels": str(cluster_labels),
        "rows": len(offsets),
        "num_clusters": nclusters,
        "blob_bytes": total,
        "offset_semantics": "start_offsets; final end is opus_blob.bin size",
        "verified_samples": min(verify_samples, len(offsets)),
    }
    atomic_json(out / "preprocess_metadata.json", metadata)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-source", required=True)
    p.add_argument("--audio-revision")
    p.add_argument("--cluster-labels", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--verify-samples", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    build_blob(
        audio_source=a.audio_source,
        audio_revision=a.audio_revision,
        cluster_labels=a.cluster_labels,
        output_dir=a.output_dir,
        batch_size=a.batch_size,
        verify_samples=a.verify_samples,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
