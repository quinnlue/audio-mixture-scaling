#!/usr/bin/env bash
# Bootstrap a runner image exactly once; CUDA packages are provided by the base image.
set -euo pipefail

repo_dir="${1:-$PWD}"
cd "$repo_dir"
python -m pip install --upgrade pip
python -m pip install -e '.[eval,prep,dev]'
python -c 'import ams; from ams.eval.hear import HearEvalConfig; print("ams bootstrap complete")'
