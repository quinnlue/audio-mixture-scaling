from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ams.data.parquet import AudioSetOpusDataset
from ams.data.sampler import RowGroupShuffleSampler
from ams.mixture.distributions import (
    generate_distributions,
    generate_regmix_paper_distributions,
    validate_distributions,
)
from ams.mixture.sampler import MixtureSampler


def test_parquet_sampler_is_deterministic_and_ddp_batches_are_disjoint(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for shard in range(2):
        table = pa.table(
            {
                "path": [f"{shard}-{i}.opus" for i in range(8)],
                "label": [[]] * 8,
                "opus": pa.array([{"bytes": b"x", "path": None}] * 8),
            }
        )
        pq.write_table(table, data / f"{shard}.parquet", row_group_size=2)
    dataset = AudioSetOpusDataset(tmp_path)
    first = list(RowGroupShuffleSampler(dataset, seed=7))
    assert first == list(RowGroupShuffleSampler(dataset, seed=7))
    assert sorted(first) == list(range(len(dataset)))
    left = list(RowGroupShuffleSampler(dataset, seed=7, num_replicas=2, rank=0, batch_size=2))
    right = list(RowGroupShuffleSampler(dataset, seed=7, num_replicas=2, rank=1, batch_size=2))
    assert len(left) == len(right) and not set(left).intersection(right)


def test_mixture_sampler_preserves_requested_cluster_mass() -> None:
    clusters = np.repeat(np.arange(3), [5, 25, 70])
    target = np.array([0.7, 0.2, 0.1])
    values = list(
        MixtureSampler(cluster_index=clusters, distribution=target, budget=50_000, seed=11)
    )
    assert values == list(
        MixtureSampler(cluster_index=clusters, distribution=target, budget=50_000, seed=11)
    )
    np.testing.assert_allclose(
        np.bincount(clusters[values], minlength=3) / len(values), target, atol=0.01
    )


def test_distribution_generation_is_reproducible() -> None:
    left = generate_distributions(np.array([0.1, 0.2, 0.3, 0.4]), seed=7)
    assert left == generate_distributions(np.array([0.1, 0.2, 0.3, 0.4]), seed=7)
    validate_distributions(left)
    assert left["distributions"][0]["kind"] == "natural"
    assert left["distributions"][1]["kind"] == "uniform"


def test_paper_regmix_generation_is_reproducible_and_feasible() -> None:
    proportions = np.array([0.1, 0.2, 0.3, 0.4])
    left = generate_regmix_paper_distributions(
        proportions, num_distributions=8, proxy_budget=10_000, seed=42
    )
    right = generate_regmix_paper_distributions(
        proportions, num_distributions=8, proxy_budget=10_000, seed=42
    )
    assert left == right
    validate_distributions(left)
    assert left["method"] == "regmix_paper_dirichlet_rejection_v1"
    assert len(left["distributions"]) == 8
    assert {row["kind"] for row in left["distributions"]} == {"regmix_proxy_candidate"}
    upper_bounds = np.asarray(left["upper_bounds"])
    for row in left["distributions"]:
        assert np.all(np.asarray(row["weights"]) <= upper_bounds + 1e-12)
