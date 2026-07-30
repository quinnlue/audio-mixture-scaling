#!/usr/bin/env bash
# One proxy run per GPU, exporting a model snapshot every AMS_EXPORT_FRACTION of
# training so the mixture ranking can be recomputed at each depth afterwards.
#
# Everything except the export cadence matches the original dist-N proxy runs
# (same seeds, same budget), so the final-depth point is directly comparable to
# the scores already in results_full/.
set -euo pipefail

: "${GPU_ID:?GPU_ID is required}"
: "${DIST_ID:?DIST_ID is required}"

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

# Replicates of a single mixture need distinct output dirs, so the trial label is
# separable from the mixture it trains on.
label="${AMS_TRIAL_LABEL:-dist-$DIST_ID}"
out="$AMS_RUN_ROOT/$label"
mkdir -p "$out" "$AMS_LOG_DIR"
exec > >(tee -a "$AMS_LOG_DIR/depth-$label.log") 2>&1

num_workers="${AMS_NUM_WORKERS:-$(ams_num_workers "$(ams_gpu_count)")}"

# total_steps lives in the config; deriving the cadence from it keeps the export
# points at exact multiples of the fraction even if the budget changes.
total_steps="$(python - "$AMS_REPO/$AMS_CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["optim"]["total_steps"])
PY
)"
export_every="${AMS_EXPORT_EVERY:-$(( total_steps / ${AMS_EXPORT_DIVISOR:-10} ))}"

while [[ ! -f "$AMS_READY" ]]; do
  echo "$label waiting for the shared RegMix blob"
  sleep 15
done

if [[ -f "$out/status.json" ]] && grep -q '"status": "complete"' "$out/status.json"; then
  echo "$label is already complete."
  exit 0
fi

resume=()
if [[ -s "$out/checkpoints/resume.pt" ]]; then
  resume=(--resume "$out/checkpoints/resume.pt")
  echo "Resuming $label on GPU $GPU_ID."
else
  echo "Starting $label (dist $DIST_ID) on GPU $GPU_ID, exporting every $export_every of $total_steps steps."
fi

wandb=(--set "tracking.wandb_mode=$AMS_WANDB")
if [[ "$AMS_WANDB" != "disabled" ]]; then
  wandb+=(
    --set "tracking.wandb_project=$AMS_WANDB_PROJECT"
    --set "tracking.wandb_run_id=$AMS_CAMPAIGN-$label"
  )
fi

cd "$AMS_REPO"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=1 -m ams.cli.pretrain \
  --config "$AMS_CONFIG" \
  --set "data.blob_dir=$AMS_BLOB_DIR" \
  --set "data.batch_size=$AMS_BATCH_SIZE" \
  --set "data.num_workers=$num_workers" \
  --set "data.seed=0" \
  --set "runtime.seed=0" \
  --set "checkpoint.export_every_n_steps=$export_every" \
  --set "mixture.distributions=$AMS_BLOB_DIR/distributions.json" \
  --set "mixture.dist_id=$DIST_ID" \
  --set "tracking.campaign_id=$AMS_CAMPAIGN" \
  --set "tracking.trial_id=$label" \
  "${wandb[@]}" \
  --output-dir "$out" \
  "${resume[@]}" \
  2>&1 | tee -a "$out/console.log"

echo "$label finished"
