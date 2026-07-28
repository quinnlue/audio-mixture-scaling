"""Pretrain EAT with the UFO objective under torchrun."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ams.config import dump_config, load_config
from ams.data import build_dataset, build_sampler, train_collate
from ams.distributed import cleanup, init_distributed, is_main
from ams.model import EATModel
from ams.objective import UFOObjective
from ams.seeding import make_generator, seed_everything, worker_init_fn
from ams.tracking.artifacts import LocalArtifacts
from ams.tracking.hub import HubUpload
from ams.tracking.wandb import WandBTracking
from ams.train import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def _shutdown_loader(loader: DataLoader[Any]) -> None:
    """Stop persistent workers that otherwise hang at interpreter exit.

    PyTorch has no public shutdown hook for persistent DataLoader workers, so
    this intentionally wraps the narrow private call in one documented place.
    """
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None


def _build_evaluator(config: Any, output_dir: Path) -> Any:
    if not config.eval.enabled:
        return None
    if not config.eval.dataset_dir:
        raise ValueError("eval.enabled requires eval.dataset_dir")

    def evaluate(model: torch.nn.Module, state: Any) -> dict[str, float]:
        from ams.eval.hear import HearEvalConfig, HearEvaluator

        hear_config = HearEvalConfig(
            data_root=Path(config.eval.dataset_dir),
            results_dir=output_dir / "results" / "hear",
            tasks=config.eval.tasks or None,
            model_id=f"step_{state.optimizer_step:08d}",
            seed=config.runtime.seed,
            device=str(state.context.device),
        )
        return HearEvaluator(config=hear_config).evaluate(model, state)

    return evaluate


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, args.overrides)
    context = init_distributed()
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_main():
        dump_config(config, args.output_dir / "config.yaml")

    dataset = build_dataset(config.data)
    sampler = build_sampler(
        dataset,
        config.data,
        rank=context.rank,
        world=context.world_size,
        mixture_cfg=config.mixture,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        drop_last=config.data.drop_last,
        num_workers=config.data.num_workers,
        collate_fn=train_collate,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        worker_init_fn=partial(worker_init_fn, base_seed=config.runtime.seed),
        generator=make_generator(config.runtime.seed),
    )
    model = EATModel(config.model)
    mask_device = context.device if context.device.type == "cuda" else torch.device("cpu")
    objective = UFOObjective(
        model,
        config.ufo,
        generator=torch.Generator(device=mask_device).manual_seed(config.runtime.seed),
    )
    if config.loop.compile and hasattr(objective, "compile"):
        objective.compile(dynamic=False)
    metadata = {
        "campaign_id": config.tracking.campaign_id,
        "trial_id": config.tracking.trial_id,
        "code_revision": config.tracking.code_revision,
        "dataset_revision": config.data.revision,
        "dist_id": config.mixture.dist_id,
        "seed": config.runtime.seed,
    }
    callbacks: list[Any] = [
        LocalArtifacts(args.output_dir, metadata),
        WandBTracking(config.tracking, config, enabled=is_main()),
        HubUpload(
            args.output_dir,
            config.tracking.hub_repo_id,
            private=config.tracking.hub_private,
            enabled=is_main(),
        ),
    ]
    trainer = Trainer(
        model=model,
        objective=objective,
        train_loader=loader,
        config=config,
        context=context,
        output_dir=args.output_dir,
        callbacks=callbacks,
        resume=args.resume,
        evaluator=_build_evaluator(config, args.output_dir),
    )
    try:
        trainer.fit()
    finally:
        _shutdown_loader(loader)
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        cleanup(context)


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
