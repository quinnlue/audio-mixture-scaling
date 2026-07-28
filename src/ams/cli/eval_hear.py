"""Evaluate an EAT checkpoint with the local HEAR protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from ams.config import load_config
from ams.eval.hear import run_hear_eval
from ams.eval.hear.presets import build_config
from ams.model import EATModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="eat")
    parser.add_argument("--preset", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def _load_model(path: Path, model: EATModel) -> None:
    if path.suffix == ".safetensors":
        state = load_file(path)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload)
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_config = load_config(args.config, args.overrides)
    model = EATModel(run_config.model)
    _load_model(args.checkpoint, model)
    config = build_config(
        args.preset,
        data_root=args.data_root,
        results_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        tasks=tuple(args.tasks) if args.tasks else None,
        limit=args.limit,
        seed=run_config.runtime.seed,
    )
    result = run_hear_eval(model, config)
    print(f"HEAR evaluation complete: {len(result.task_results)} task(s)")


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
