"""Write the deliberately analysis-free RegMix campaign inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "training_complete",
                "trial_records": [str(p) for p in args.inputs],
                "num_trials": len(args.inputs),
                "note": "Mixture-to-quality regression is outside this campaign phase.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
