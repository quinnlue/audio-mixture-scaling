#!/usr/bin/env bash
# Shared campaign layout and host probing. Source, do not execute.
#
# Precedence is explicit environment, then the campaign state file written by
# vast_campaign.sh, then these defaults; every consumer therefore agrees on
# paths without repeating them.

AMS_STATE="${AMS_STATE:-/workspace/ams-campaign.env}"
if [[ -f "$AMS_STATE" ]]; then
  # shellcheck disable=SC1090
  source "$AMS_STATE"
fi

AMS_REPO="${AMS_REPO:-/workspace/audio-mixture-scaling}"
AMS_DATA="${AMS_DATA:-/workspace/data}"
AMS_AUDIO_DIR="${AMS_AUDIO_DIR:-$AMS_DATA/audioset-opus}"
AMS_BLOB_DIR="${AMS_BLOB_DIR:-$AMS_DATA/audioset-regmix-blob}"
AMS_CAMPAIGN="${AMS_CAMPAIGN:-regmix-proxy}"
AMS_RUN_ROOT="${AMS_RUN_ROOT:-/workspace/runs/$AMS_CAMPAIGN}"
AMS_LOG_DIR="${AMS_LOG_DIR:-/workspace/ams-logs}"

AMS_CLUSTERING="${AMS_CLUSTERING:-audioset-dasheng-0.6b-k20-r31d6389}"
AMS_ARTIFACTS="${AMS_ARTIFACTS:-$AMS_REPO/artifacts/$AMS_CLUSTERING}"
AMS_DISTRIBUTIONS="${AMS_DISTRIBUTIONS:-$AMS_ARTIFACTS/regmix-paper-64/distributions.json}"
AMS_CLUSTER_LABELS="${AMS_CLUSTER_LABELS:-$AMS_ARTIFACTS/cluster_labels.parquet}"

AMS_DATASET="${AMS_DATASET:-danjacobellis/audioset_opus_24kbps}"
AMS_REVISION="${AMS_REVISION:-a725d7cf1fea563c6eb9f6127dbd1d75b294668d}"
AMS_SHARD_COUNT="${AMS_SHARD_COUNT:-96}"

AMS_CONFIG="${AMS_CONFIG:-configs/regmix_proxy.yaml}"
AMS_BATCH_SIZE="${AMS_BATCH_SIZE:-48}"
AMS_WANDB="${AMS_WANDB:-disabled}"
AMS_WANDB_PROJECT="${AMS_WANDB_PROJECT:-eat-regmix-proxy}"
AMS_KEEP_PARQUET="${AMS_KEEP_PARQUET:-1}"

AMS_READY="$AMS_BLOB_DIR/.ready"

ams_activate() {
  if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
  elif [[ -n "${VIRTUAL_ENV:-}" && -f "$VIRTUAL_ENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VIRTUAL_ENV/bin/activate"
  fi
  command -v python >/dev/null || { echo "no python on PATH" >&2; return 1; }
}

ams_gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

# Opus decode is the only CPU cost, and it is what starves the GPU: four runs at
# 16 workers on 88 vCPU measured 6.86 steps/s against 10.28 for a single run on
# the same card. Give each run its share of the box, minus a couple of cores for
# the driver process, rather than a fixed ceiling.
AMS_MAX_WORKERS="${AMS_MAX_WORKERS:-48}"

ams_num_workers() {
  local concurrent="$1" cpus
  cpus="$(nproc)"
  local workers=$(( cpus / concurrent - 2 ))
  (( workers < 2 )) && workers=2
  (( workers > AMS_MAX_WORKERS )) && workers="$AMS_MAX_WORKERS"
  echo "$workers"
}

ams_free_gb() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}
