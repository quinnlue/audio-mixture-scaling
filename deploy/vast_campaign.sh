#!/usr/bin/env bash
# Generate and install the supervisor campaign: one prep job plus one worker per GPU.
# Usage: deploy/vast_campaign.sh [num_runs]   (or AMS_DIST_IDS="0,3,7" deploy/vast_campaign.sh)
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"

gpus="$(ams_gpu_count)"
(( gpus > 0 )) || { echo "no GPUs visible" >&2; exit 1; }

if [[ -n "${AMS_DIST_IDS:-}" ]]; then
  read -ra dist_ids <<<"${AMS_DIST_IDS//,/ }"
else
  num_runs="${1:-$gpus}"
  mapfile -t dist_ids < <(seq 0 $(( num_runs - 1 )))
fi
(( ${#dist_ids[@]} > 0 )) || { echo "no dist ids requested" >&2; exit 1; }

conf="${AMS_SUPERVISOR_CONF:-/etc/supervisor/conf.d/ams-campaign.conf}"
script_dir="$(dirname "$(readlink -f "$0")")"

# Fewer runs than GPUs means fewer concurrent loaders, so size workers by what
# actually runs at once rather than by the card count.
concurrent="$gpus"
(( ${#dist_ids[@]} < concurrent )) && concurrent="${#dist_ids[@]}"
workers="${AMS_NUM_WORKERS:-$(ams_num_workers "$concurrent")}"

# Pin the resolved campaign so workers, prep, and status share one view.
{
  for name in AMS_REPO AMS_DATA AMS_AUDIO_DIR AMS_BLOB_DIR AMS_CAMPAIGN AMS_RUN_ROOT \
    AMS_LOG_DIR AMS_ARTIFACTS AMS_DISTRIBUTIONS AMS_CLUSTER_LABELS AMS_DATASET \
    AMS_REVISION AMS_SHARD_COUNT AMS_CONFIG AMS_BATCH_SIZE AMS_WANDB AMS_WANDB_PROJECT \
    AMS_KEEP_PARQUET; do
    printf 'export %s="${%s:-%s}"\n' "$name" "$name" "${!name}"
  done
  printf 'export AMS_NUM_WORKERS="${AMS_NUM_WORKERS:-%s}"\n' "$workers"
} >"$AMS_STATE"

{
  echo "[program:ams-prep]"
  echo "command=$script_dir/vast_prepare_regmix.sh"
  echo "environment=AMS_STATE=\"$AMS_STATE\""
  echo "autostart=true"
  echo "autorestart=unexpected"
  echo "startretries=3"
  echo "startsecs=0"
  echo "exitcodes=0"
  echo "stopasgroup=true"
  echo "killasgroup=true"
  echo "stdout_logfile=/dev/stdout"
  echo "redirect_stderr=true"
  echo "stdout_logfile_maxbytes=0"
  echo

  for (( gpu = 0; gpu < gpus; gpu++ )); do
    queue=""
    for (( i = gpu; i < ${#dist_ids[@]}; i += gpus )); do
      queue="${queue:+$queue,}${dist_ids[i]}"
    done
    [[ -n "$queue" ]] || continue
    echo "[program:ams-gpu-$gpu]"
    echo "command=$script_dir/vast_run_proxy.sh"
    echo "environment=GPU_ID=\"$gpu\",DIST_IDS=\"$queue\",AMS_STATE=\"$AMS_STATE\""
    echo "autostart=true"
    echo "autorestart=unexpected"
    echo "startretries=100"
    echo "startsecs=10"
    echo "exitcodes=0"
    echo "stopasgroup=true"
    echo "killasgroup=true"
    echo "stdout_logfile=/dev/stdout"
    echo "redirect_stderr=true"
    echo "stdout_logfile_maxbytes=0"
    echo
  done
} >"$conf"

echo "campaign  $AMS_CAMPAIGN"
echo "runs      ${#dist_ids[@]} across $gpus GPUs, $workers loader workers each"
echo "outputs   $AMS_RUN_ROOT/dist-*"
echo "state     $AMS_STATE"
echo "conf      $conf"

if command -v supervisorctl >/dev/null; then
  supervisorctl reread
  supervisorctl update
  supervisorctl status | grep '^ams-' || true
else
  echo "supervisor not found; start workers manually with GPU_ID/DIST_IDS"
fi
