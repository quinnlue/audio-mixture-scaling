#!/usr/bin/env bash
# Drain a queue of RegMix proxy runs on one physical GPU, one run at a time.
# DIST_IDS is a comma-separated list; DIST_ID stays accepted for single-run use.
set -euo pipefail

: "${GPU_ID:?GPU_ID is required}"
DIST_IDS="${DIST_IDS:-${DIST_ID:?DIST_IDS or DIST_ID is required}}"

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

mkdir -p "$AMS_RUN_ROOT" "$AMS_LOG_DIR"
exec > >(tee -a "$AMS_LOG_DIR/gpu-$GPU_ID.log") 2>&1

num_workers="${AMS_NUM_WORKERS:-$(ams_num_workers "$(ams_gpu_count)")}"

waited=0
while [[ ! -f "$AMS_READY" ]]; do
  (( waited % 20 == 0 )) && echo "gpu-$GPU_ID waiting for the shared RegMix blob"
  waited=$(( waited + 1 ))
  sleep 15
done

for dist_id in ${DIST_IDS//,/ }; do
  out="$AMS_RUN_ROOT/dist-$dist_id"
  mkdir -p "$out"
  if [[ -f "$out/status.json" ]] && grep -q '"status": "complete"' "$out/status.json"; then
    echo "dist-$dist_id is already complete."
    continue
  fi

  resume=()
  if [[ -s "$out/checkpoints/resume.pt" ]]; then
    resume=(--resume "$out/checkpoints/resume.pt")
    echo "Resuming dist-$dist_id on physical GPU $GPU_ID."
  else
    echo "Starting dist-$dist_id on physical GPU $GPU_ID."
  fi

  wandb=(--set "tracking.wandb_mode=$AMS_WANDB")
  if [[ "$AMS_WANDB" != "disabled" ]]; then
    wandb+=(
      --set "tracking.wandb_project=$AMS_WANDB_PROJECT"
      --set "tracking.wandb_run_id=$AMS_CAMPAIGN-dist-$dist_id"
    )
  fi

  cd "$AMS_REPO"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONUNBUFFERED=1 \
  torchrun \
    --standalone \
    --nproc_per_node=1 \
    -m ams.cli.pretrain \
    --config "$AMS_CONFIG" \
    --set "data.blob_dir=$AMS_BLOB_DIR" \
    --set "data.batch_size=$AMS_BATCH_SIZE" \
    --set "data.num_workers=$num_workers" \
    --set "mixture.distributions=$AMS_BLOB_DIR/distributions.json" \
    --set "mixture.dist_id=$dist_id" \
    --set "tracking.campaign_id=$AMS_CAMPAIGN" \
    --set "tracking.trial_id=dist-$dist_id" \
    "${wandb[@]}" \
    --output-dir "$out" \
    "${resume[@]}" \
    2>&1 | tee -a "$out/console.log"
done

echo "gpu-$GPU_ID drained: $DIST_IDS"
