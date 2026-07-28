"""Generate reproducible data mixtures, including paper-faithful RegMix candidates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def generate_distributions(
    proportions: np.ndarray,
    *,
    num_distributions: int = 64,
    seed: int = 0,
    min_concentration: float = 0.1,
    max_concentration: float = 10_000.0,
) -> dict:
    p = np.asarray(proportions, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or np.any(p <= 0) or not np.all(np.isfinite(p)):
        raise ValueError("proportions must be finite and strictly positive")
    if num_distributions < 2 or seed < 0 or not 0 < min_concentration <= max_concentration:
        raise ValueError("invalid generation parameters")
    p /= p.sum()
    rng = np.random.default_rng(seed)
    rows = [
        {"dist_id": 0, "kind": "natural", "concentration": None, "weights": p.tolist()},
        {
            "dist_id": 1,
            "kind": "uniform",
            "concentration": None,
            "weights": np.full(len(p), 1 / len(p)).tolist(),
        },
    ]
    for dist_id, concentration in enumerate(
        np.geomspace(max_concentration, min_concentration, num=num_distributions - 2), 2
    ):
        weights = rng.dirichlet(concentration * p)
        rows.append(
            {
                "dist_id": dist_id,
                "kind": "size_anchored_dirichlet",
                "concentration": float(concentration),
                "weights": (weights / weights.sum()).tolist(),
            }
        )
    return {
        "schema_version": 1,
        "method": "size_anchored_dirichlet_concentration_sweep",
        "seed": seed,
        "num_clusters": len(p),
        "num_distributions": num_distributions,
        "min_concentration": min_concentration,
        "max_concentration": max_concentration,
        "distributions": rows,
    }


def _sort_and_deduplicate(
    weights: list[np.ndarray], concentrations: list[float], *, threshold: float = 1e-5
) -> tuple[list[np.ndarray], list[float]]:
    """Match the released RegMix lexicographic, adjacent-L1 deduplication."""
    matrix = np.asarray(weights)
    order = np.lexsort(matrix.T)
    kept_weights = [matrix[order[0]]]
    kept_concentrations = [concentrations[int(order[0])]]
    for index in order[1:]:
        row = matrix[index]
        if np.abs(row - kept_weights[-1]).sum() > threshold:
            kept_weights.append(row)
            kept_concentrations.append(concentrations[int(index)])
    return kept_weights, kept_concentrations


def generate_regmix_paper_distributions(
    proportions: np.ndarray,
    *,
    num_distributions: int = 64,
    proxy_budget: int,
    seed: int = 42,
    min_concentration: float = 0.1,
    max_concentration: float = 5.0,
    num_concentrations: int = 15,
    temperature: float = 0.5,
    sample_multiplier: int = 100,
    max_usage: float = 15.0,
    minimum_examples: int = 100,
) -> dict:
    """Adapt the RegMix paper/released generator to fixed audio clusters.

    The paper samples ``Dirichlet(lambda * natural_proportions)`` for
    ``lambda in [0.1, 5]``. The released generator additionally smooths the
    prior, rejects mixtures that over-use a data group, quantizes tiny weights,
    oversamples, and deduplicates. We preserve those details while scaling the
    minimum active mass to ``minimum_examples / proxy_budget``.
    """
    p = np.asarray(proportions, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or np.any(p <= 0) or not np.all(np.isfinite(p)):
        raise ValueError("proportions must be finite and strictly positive")
    if (
        num_distributions < 1
        or proxy_budget < 1
        or seed < 0
        or not 0 < min_concentration <= max_concentration
        or num_concentrations < 1
        or not 0 < temperature <= 1
        or sample_multiplier < 1
        or max_usage <= 0
        or not 1 <= minimum_examples <= proxy_budget
    ):
        raise ValueError("invalid paper RegMix generation arguments")

    p /= p.sum()
    sampling_prior = p**temperature
    sampling_prior /= sampling_prior.sum()
    upper_bounds = np.minimum(max_usage * p, 1.0)
    minimum_weight = minimum_examples / proxy_budget
    concentrations = np.geomspace(
        min_concentration, max_concentration, num=num_concentrations
    )
    numpy_rng = np.random.RandomState(seed)
    python_rng = random.Random(seed)
    accepted_weights: list[np.ndarray] = []
    accepted_concentrations: list[float] = []
    proposal_count = num_distributions * sample_multiplier

    # The released implementation draws once at every strength and then chooses
    # one strength uniformly, which we retain for exact RNG semantics.
    for _ in range(proposal_count):
        draws = [
            numpy_rng.dirichlet(sampling_prior * concentration)
            for concentration in concentrations
        ]
        strength_index = python_rng.randrange(num_concentrations)
        weights = draws[strength_index]
        if np.any(weights > upper_bounds):
            continue
        weights = np.where(weights < minimum_weight, 0.0, weights)
        if weights.sum() <= 0:
            continue
        weights /= weights.sum()
        weights = np.round(weights / minimum_weight) * minimum_weight
        # The local JSON contract requires an exact simplex. The released YAML
        # relies on downstream normalization after rounding.
        weights /= weights.sum()
        accepted_weights.append(weights)
        accepted_concentrations.append(float(concentrations[strength_index]))

    if not accepted_weights:
        raise RuntimeError("paper RegMix rejection sampling accepted no candidates")
    deduplicated_weights, deduplicated_concentrations = _sort_and_deduplicate(
        accepted_weights, accepted_concentrations
    )
    if len(deduplicated_weights) < num_distributions:
        raise RuntimeError(
            f"only {len(deduplicated_weights)} unique candidates available; "
            f"need {num_distributions}"
        )
    selected = python_rng.sample(range(len(deduplicated_weights)), num_distributions)
    rows = [
        {
            "dist_id": dist_id,
            "kind": "regmix_proxy_candidate",
            "concentration": deduplicated_concentrations[index],
            "weights": deduplicated_weights[index].tolist(),
        }
        for dist_id, index in enumerate(selected)
    ]
    return {
        "schema_version": 1,
        "method": "regmix_paper_dirichlet_rejection_v1",
        "paper": "Liu et al., RegMix, arXiv:2407.01492",
        "reference_implementation": (
            "https://github.com/sail-sg/regmix/blob/main/"
            "mixture_config/synthesize_mixture.py"
        ),
        "seed": seed,
        "num_clusters": len(p),
        "num_distributions": num_distributions,
        "proxy_budget": proxy_budget,
        "natural_proportions": p.tolist(),
        "sampling_prior": sampling_prior.tolist(),
        "temperature": temperature,
        "min_concentration": min_concentration,
        "max_concentration": max_concentration,
        "num_concentrations": num_concentrations,
        "concentration_grid": concentrations.tolist(),
        "sample_multiplier": sample_multiplier,
        "proposal_count": proposal_count,
        "accepted_count": len(accepted_weights),
        "deduplicated_count": len(deduplicated_weights),
        "max_usage": max_usage,
        "upper_bounds": upper_bounds.tolist(),
        "minimum_examples": minimum_examples,
        "minimum_weight": minimum_weight,
        "rounding": "nearest minimum_weight, then renormalize exactly",
        "adaptations": [
            f"num_distributions set to {num_distributions}",
            "natural domains replaced by fixed embedding clusters",
            "minimum active mass scaled to minimum_examples / proxy_budget",
        ],
        "distributions": rows,
    }


def validate_distributions(payload: dict) -> None:
    rows, n = payload.get("distributions"), int(payload["num_clusters"])
    if (
        not isinstance(rows, list)
        or len(rows) != payload.get("num_distributions")
        or [r.get("dist_id") for r in rows] != list(range(len(rows)))
    ):
        raise ValueError("invalid distribution ids")
    for row in rows:
        w = np.asarray(row.get("weights"), dtype=np.float64)
        if (
            w.shape != (n,)
            or not np.all(np.isfinite(w))
            or np.any(w < 0)
            or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError(f"invalid weights for dist {row['dist_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-stats", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--method",
        choices=("concentration-sweep", "regmix-paper"),
        default="concentration-sweep",
    )
    parser.add_argument("--num-distributions", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--proxy-budget", type=int)
    parser.add_argument("--min-concentration", type=float, default=0.1)
    parser.add_argument("--max-concentration", type=float, default=10_000.0)
    args = parser.parse_args()
    path = (
        args.cluster_stats / "cluster_stats.json"
        if args.cluster_stats.is_dir()
        else args.cluster_stats
    )
    proportions = np.asarray(json.loads(path.read_text())["proportions"])
    if args.method == "regmix-paper":
        if args.proxy_budget is None:
            parser.error("--proxy-budget is required with --method regmix-paper")
        payload = generate_regmix_paper_distributions(
            proportions,
            num_distributions=args.num_distributions,
            proxy_budget=args.proxy_budget,
            seed=42 if args.seed is None else args.seed,
        )
    else:
        payload = generate_distributions(
            proportions,
            num_distributions=args.num_distributions,
            seed=0 if args.seed is None else args.seed,
            min_concentration=args.min_concentration,
            max_concentration=args.max_concentration,
        )
    validate_distributions(payload)
    output = args.output or path.parent / "distributions.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
