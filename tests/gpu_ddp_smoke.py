"""Two-rank CUDA smoke for the real EAT/UFO training and artifact path.

Run from an installed checkout:

    torchrun --standalone --nproc_per_node=2 tests/gpu_ddp_smoke.py --output-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ams.config import (
    CheckpointConfig,
    DataConfig,
    LoopConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrackingConfig,
    UFOConfig,
)
from ams.data.collate import WaveformBatch
from ams.distributed import cleanup, init_distributed
from ams.model import EATModel
from ams.objective import UFOObjective
from ams.seeding import seed_everything
from ams.tracking import LocalArtifacts
from ams.train import Trainer


class _Waveforms(Dataset[torch.Tensor]):
    def __init__(self, count: int = 4, samples: int = 4_000) -> None:
        generator = torch.Generator().manual_seed(123)
        self.values = torch.randn(count, samples, generator=generator)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


def _collate(items: list[torch.Tensor]) -> WaveformBatch:
    waveforms = torch.stack(items)
    return WaveformBatch(waveforms, torch.ones_like(waveforms, dtype=torch.bool))


def _config(precision: str) -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            d_model=24,
            depth=2,
            num_heads=3,
            target_length=32,
            max_time_patches=2,
            decoder_depth=1,
        ),
        ufo=UFOConfig(
            num_masks=1,
            average_top_k_layers=2,
            ema_anneal_steps=2,
        ),
        data=DataConfig(
            dataset_dir="synthetic",
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
        ),
        optim=OptimConfig(lr=1e-3, warmup_steps=0, total_steps=2),
        loop=LoopConfig(
            max_epochs=1,
            precision=precision,
            log_every_n_steps=1,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_epochs=1,
            save_every_n_steps=1,
        ),
        tracking=TrackingConfig(
            campaign_id="gpu-smoke",
            trial_id="two-rank",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = init_distributed()
    if context.device.type != "cuda":
        raise RuntimeError("gpu_ddp_smoke requires CUDA")
    seed_everything(0)
    config = _config(args.precision)
    dataset = _Waveforms()
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=False,
        drop_last=True,
    )
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=_collate)
    model = EATModel(config.model)
    objective = UFOObjective(
        model,
        config.ufo,
        generator=torch.Generator(device=context.device).manual_seed(0),
    )
    artifacts = LocalArtifacts(
        args.output_dir,
        {"campaign_id": "gpu-smoke", "trial_id": "two-rank"},
    )
    state = Trainer(
        model=model,
        objective=objective,
        train_loader=loader,
        config=config,
        context=context,
        output_dir=args.output_dir,
        callbacks=[artifacts],
        resume=args.resume,
    ).fit()
    norm = torch.tensor(state.last_grad_norm, device=context.device)
    gathered = [torch.zeros_like(norm) for _ in range(context.world_size)]
    dist.all_gather(gathered, norm)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise AssertionError(f"gradient norms differ across ranks: {gathered}")
    dist.barrier(device_ids=[context.local_rank])
    if context.rank == 0:
        resume = args.output_dir / "checkpoints" / "resume.json"
        status = args.output_dir / "status.json"
        if not resume.is_file() or json.loads(status.read_text())["status"] != "complete":
            raise AssertionError("rank-zero runner artifacts are incomplete")
        report = {
            "world_size": context.world_size,
            "optimizer_step": state.optimizer_step,
            "grad_norms": [float(value) for value in gathered],
            "devices": [f"cuda:{index}" for index in range(context.world_size)],
            "precision": args.precision,
        }
        (args.output_dir / "ddp_smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    cleanup(context)


if __name__ == "__main__":
    main()
