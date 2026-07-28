"""Fit MiniBatchKMeans over embedding Parquet and publish aligned labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._io import atomic_json, iter_parquet_batches, parquet_files


def _matrix(column) -> np.ndarray:
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
            len(array), array.type.list_size
        )
    return np.asarray(array.to_pylist(), dtype=np.float32)


def cluster_embeddings(
    *,
    source: str | Path,
    output_dir: str | Path,
    revision: str | None = None,
    num_clusters: int = 20,
    seed: int = 0,
    batch_size: int = 8192,
) -> dict:
    from sklearn.cluster import MiniBatchKMeans

    if num_clusters < 2 or seed < 0 or batch_size < num_clusters:
        raise ValueError("invalid clustering arguments")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = parquet_files(source, revision=revision)
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=batch_size,
        n_init=1,
        reassignment_ratio=0.0,
    )
    rows = 0
    dim = None
    for batch in iter_parquet_batches(files, columns=("emb",), batch_size=batch_size):
        emb = _matrix(batch.column(0))
        dim = emb.shape[1] if dim is None else dim
        if emb.shape[1] != dim:
            raise ValueError("embedding dimensions vary")
        model.partial_fit(emb)
        rows += len(emb)
    if rows < num_clusters or dim is None:
        raise ValueError("not enough embeddings")
    labels_path = output / "cluster_labels.parquet"
    tmp = labels_path.with_suffix(".parquet.tmp")
    writer = None
    sizes = np.zeros(num_clusters, dtype=np.int64)
    try:
        for batch in iter_parquet_batches(files, columns=("path", "emb"), batch_size=batch_size):
            labels = model.predict(_matrix(batch.column(1))).astype(np.int8)
            sizes += np.bincount(labels, minlength=num_clusters)
            table = pa.table({"path": batch.column(0), "cluster": pa.array(labels)})
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    if np.any(sizes == 0):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"empty clusters: {np.flatnonzero(sizes == 0).tolist()}")
    tmp.replace(labels_path)
    np.save(output / "centroids.npy", np.asarray(model.cluster_centers_, dtype=np.float32))
    stats = {
        "schema_version": 1,
        "source": str(source),
        "source_revision": revision,
        "algorithm": "sklearn.cluster.MiniBatchKMeans",
        "num_clusters": num_clusters,
        "seed": seed,
        "batch_size": batch_size,
        "rows": rows,
        "embedding_dim": dim,
        "sizes": sizes.tolist(),
        "proportions": (sizes / sizes.sum()).tolist(),
    }
    atomic_json(output / "cluster_stats.json", stats)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--revision")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--num-clusters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8192)
    a = p.parse_args()
    cluster_embeddings(
        source=a.source,
        revision=a.revision,
        output_dir=a.output_dir,
        num_clusters=a.num_clusters,
        seed=a.seed,
        batch_size=a.batch_size,
    )


if __name__ == "__main__":
    main()
