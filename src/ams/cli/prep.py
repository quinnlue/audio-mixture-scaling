"""Dispatch offline-only data preparation commands."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cluster", "blob", "hear-parquet"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    if parsed.command == "cluster":
        from ams.prep.cluster import main as command
    elif parsed.command == "blob":
        from ams.prep.build_blob import main as command
    else:
        from ams.prep.build_hear_parquet import main as command
    previous = sys.argv
    try:
        sys.argv = [f"ams-prep {parsed.command}", *parsed.args]
        command()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()


__all__ = ["main"]
