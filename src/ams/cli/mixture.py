"""RegMix distribution generation and inventory commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ams.mixture.distributions import (
    generate_distributions,
    generate_regmix_paper_distributions,
    validate_distributions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--cluster-stats", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument(
        "--method",
        choices=("concentration-sweep", "regmix-paper"),
        default="concentration-sweep",
    )
    generate.add_argument("--num-distributions", type=int, default=64)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--proxy-budget", type=int)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("inputs", nargs="*", type=Path)
    inventory.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        source = (
            args.cluster_stats / "cluster_stats.json"
            if args.cluster_stats.is_dir()
            else args.cluster_stats
        )
        stats = json.loads(source.read_text(encoding="utf-8"))
        if args.method == "regmix-paper":
            if args.proxy_budget is None:
                raise SystemExit("--proxy-budget is required with --method regmix-paper")
            payload = generate_regmix_paper_distributions(
                np.asarray(stats["proportions"]),
                num_distributions=args.num_distributions,
                proxy_budget=args.proxy_budget,
                seed=42 if args.seed is None else args.seed,
            )
            payload["cluster_source"] = stats.get("source")
            payload["cluster_source_revision"] = stats.get("source_revision")
            payload["cluster_stats_schema_version"] = stats.get("schema_version")
        else:
            payload = generate_distributions(
                np.asarray(stats["proportions"]),
                num_distributions=args.num_distributions,
                seed=0 if args.seed is None else args.seed,
            )
        validate_distributions(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": "training_complete",
            "trial_records": [str(path) for path in args.inputs],
            "num_trials": len(args.inputs),
            "note": "Mixture-to-quality regression is outside this repository.",
        }
        (args.output_dir / "inventory.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
