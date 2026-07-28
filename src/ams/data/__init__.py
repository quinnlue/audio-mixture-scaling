"""Training-time audio input pipelines."""

from .blob import OpusBlobDataset
from .collate import WaveformBatch, train_collate
from .parquet import AudioSetOpusDataset
from .sampler import RowGroupShuffleSampler


def build_dataset(data_cfg):
    """Construct the configured training dataset without importing preparation code."""
    if data_cfg.format == "parquet":
        if not data_cfg.dataset_dir:
            raise ValueError("parquet input requires data.dataset_dir")
        return AudioSetOpusDataset(data_cfg.dataset_dir)
    if data_cfg.format == "blob":
        if not data_cfg.blob_dir:
            raise ValueError("blob input requires data.blob_dir")
        return OpusBlobDataset(data_cfg.blob_dir)
    raise ValueError(f"unsupported data format {data_cfg.format!r}")


def build_sampler(dataset, data_cfg, rank: int = 0, world: int = 1, mixture_cfg=None):
    if data_cfg.format == "parquet":
        return RowGroupShuffleSampler(
            dataset,
            seed=data_cfg.seed,
            num_replicas=world,
            rank=rank,
            batch_size=data_cfg.batch_size,
        )
    if data_cfg.format == "blob":
        from ams.mixture.sampler import MixtureSampler, load_distribution

        if mixture_cfg is None:
            raise ValueError("blob input requires RunConfig.mixture")
        if (
            not mixture_cfg.distributions
            or mixture_cfg.budget is None
            or mixture_cfg.dist_id is None
        ):
            raise ValueError(
                "blob input requires mixture.distributions, mixture.dist_id, and mixture.budget"
            )
        path = mixture_cfg.distributions
        dist_id = mixture_cfg.dist_id
        budget = mixture_cfg.budget
        return MixtureSampler(
            cluster_index=dataset.cluster_index,
            distribution=load_distribution(path, dist_id),
            budget=budget,
            seed=data_cfg.seed,
        )
    raise ValueError(f"unsupported data format {data_cfg.format!r}")


__all__ = [
    "AudioSetOpusDataset",
    "OpusBlobDataset",
    "RowGroupShuffleSampler",
    "WaveformBatch",
    "build_dataset",
    "build_sampler",
    "train_collate",
]
