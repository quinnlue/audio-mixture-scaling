#!/usr/bin/env bash
# Find the loader-worker count where a full-width campaign stops being CPU-bound.
#
# Runs every GPU at once, exactly as the campaign will, and reports steady-state
# steps/s plus GPU utilization per worker count. Rising throughput means decode
# is still the bottleneck; a plateau at high GPU utilization means the card is.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

gpus="$(ams_gpu_count)"
steps="${AMS_TUNE_STEPS:-300}"
candidates="${AMS_TUNE_WORKERS:-16 32 48}"
root="${AMS_TUNE_ROOT:-/workspace/runs/worker-tuning}"

[[ -f "$AMS_READY" ]] || { echo "blob not ready at $AMS_BLOB_DIR" >&2; exit 1; }
cd "$AMS_REPO"

echo "tuning on $gpus GPUs, $(nproc) vCPU, $steps steps per point"
printf '%-10s%-14s%-14s%-10s\n' workers steps/s/run aggregate gpu%

for workers in $candidates; do
  rm -rf "$root"
  for (( gpu = 0; gpu < gpus; gpu++ )); do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node=1 -m ams.cli.pretrain \
      --config "$AMS_CONFIG" \
      --set "data.blob_dir=$AMS_BLOB_DIR" \
      --set "data.batch_size=$AMS_BATCH_SIZE" \
      --set "data.num_workers=$workers" \
      --set "mixture.distributions=$AMS_BLOB_DIR/distributions.json" \
      --set "mixture.dist_id=$gpu" \
      --set "mixture.budget=$(( AMS_BATCH_SIZE * steps ))" \
      --set optim.warmup_steps=10 \
      --set "optim.total_steps=$steps" \
      --set loop.log_every_n_steps=50 \
      --set tracking.wandb_mode=disabled \
      --output-dir "$root/w$workers-gpu$gpu" >/dev/null 2>&1 &
  done

  # Sample utilization mid-run rather than during warmup or teardown.
  sleep 45
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | awk '{total += $1; n += 1} END {if (n) printf "%.0f", total / n}')"
  wait

  python - "$root" "$workers" "$util" <<'PY'
import json
import sys
from pathlib import Path

root, workers, util = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rates = []
for run in sorted(root.glob(f"w{workers}-gpu*")):
    events = (run / "events.jsonl").read_text().splitlines()
    metrics = [json.loads(l) for l in events if '"metrics"' in l]
    steady = [m["metrics"]["train/steps_per_second"] for m in metrics][1:]
    if steady:
        rates.append(sum(steady) / len(steady))
if rates:
    per_run = sum(rates) / len(rates)
    print(f"{workers:<10}{per_run:<14.2f}{per_run * len(rates):<14.2f}{util + '%':<10}")
else:
    print(f"{workers:<10}{'FAILED':<14}{'-':<14}{util + '%':<10}")
PY
done

rm -rf "$root"
echo
echo "Pick the smallest worker count at the plateau; set it with AMS_NUM_WORKERS."
