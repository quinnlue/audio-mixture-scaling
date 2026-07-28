#!/usr/bin/env bash
# Clone or update the repo, install it into the image venv, and report host fitness.
set -euo pipefail

AMS_REPO="${AMS_REPO:-/workspace/audio-mixture-scaling}"
AMS_GIT_URL="${AMS_GIT_URL:-https://github.com/quinnlue/audio-mixture-scaling.git}"
AMS_GIT_REF="${AMS_GIT_REF:-main}"

if [[ ! -d "$AMS_REPO/.git" ]]; then
  git clone "$AMS_GIT_URL" "$AMS_REPO"
fi
cd "$AMS_REPO"
git fetch --depth 1 origin "$AMS_GIT_REF"
git checkout -q FETCH_HEAD

# shellcheck disable=SC1091
source "$AMS_REPO/deploy/vast_env.sh"
ams_activate

# The image supplies torch/torchcodec; only the pure-python deps are missing.
if command -v uv >/dev/null; then
  uv pip install -e .
else
  python -m pip install -e .
fi

chmod +x deploy/*.sh
mkdir -p "$AMS_DATA" "$AMS_LOG_DIR" "$AMS_RUN_ROOT"

python - "$AMS_DISTRIBUTIONS" <<'PY'
import json
import sys
from pathlib import Path

import torch

import ams  # noqa: F401
from ams.mixture.distributions import validate_distributions

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_distributions(payload)
device_count = torch.cuda.device_count()
if device_count:
    torch.empty(8, device="cuda").normal_()
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={device_count}")
print(f"distributions {payload['num_distributions']} clusters {payload['num_clusters']}")
PY

gpus="$(ams_gpu_count)"
free_gb="$(ams_free_gb /workspace)"
echo
echo "repo        $AMS_REPO @ $(git rev-parse --short HEAD)"
echo "gpus        $gpus x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "vcpu        $(nproc)"
echo "ram         $(free -g | awk '/^Mem:/{print $2}') GB"
echo "free disk   ${free_gb} GB on /workspace"
echo "workers/run $(ams_num_workers "${gpus:-1}") at ${gpus:-1} concurrent runs"

if (( free_gb < 130 )); then
  echo "WARNING: under 130 GB free; set AMS_KEEP_PARQUET=0 to drop the snapshot after packing"
fi
