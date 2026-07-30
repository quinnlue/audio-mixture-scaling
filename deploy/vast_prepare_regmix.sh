#!/usr/bin/env bash
# Materialize the shared mmap Opus blob once per node. Idempotent; workers wait on .ready.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

mkdir -p "$AMS_AUDIO_DIR" "$AMS_BLOB_DIR" "$AMS_LOG_DIR"
exec > >(tee -a "$AMS_LOG_DIR/prep.log") 2>&1

if [[ -f "$AMS_READY" ]]; then
  echo "RegMix blob is already ready."
  exit 0
fi

needed=$(( AMS_KEEP_PARQUET == 1 ? 130 : 70 ))
free_gb="$(ams_free_gb "$AMS_DATA")"
if (( free_gb < needed )); then
  echo "Need ${needed} GB free under $AMS_DATA, found ${free_gb} GB." >&2
  exit 1
fi

echo "Downloading the pinned AudioSet Opus Parquet snapshot."
"$(dirname "$(readlink -f "$0")")/vast_fetch_snapshot.sh"

shard_count="$(find "$AMS_AUDIO_DIR/data" -maxdepth 1 -name '*.parquet' -type f | wc -l)"
if [[ "$shard_count" -ne "$AMS_SHARD_COUNT" ]]; then
  echo "Expected $AMS_SHARD_COUNT Parquet shards, found $shard_count." >&2
  exit 1
fi

required=(
  "$AMS_BLOB_DIR/opus_blob.bin"
  "$AMS_BLOB_DIR/opus_offsets.npy"
  "$AMS_BLOB_DIR/cluster_index.npy"
  "$AMS_BLOB_DIR/preprocess_metadata.json"
)
complete=true
overwrite=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || complete=false
  [[ -e "$path" ]] && overwrite=(--overwrite)
done

if [[ "$complete" != true ]]; then
  echo "Packing the mmap-ready Opus blob."
  cd "$AMS_REPO"
  python -m ams.cli.prep blob \
    --audio-source "$AMS_AUDIO_DIR" \
    --audio-revision "$AMS_REVISION" \
    --cluster-labels "$AMS_CLUSTER_LABELS" \
    --output-dir "$AMS_BLOB_DIR" \
    --batch-size 2048 \
    "${overwrite[@]}"
fi

cp "$AMS_DISTRIBUTIONS" "$AMS_BLOB_DIR/distributions.json"

cd "$AMS_REPO"
python -m ams.data.validate --dataset-dir "$AMS_BLOB_DIR"

if [[ "$AMS_KEEP_PARQUET" != "1" ]]; then
  echo "Dropping the Parquet snapshot; the blob is self-contained."
  rm -rf "${AMS_AUDIO_DIR:?}/data" "$AMS_AUDIO_DIR/.cache"
fi

touch "$AMS_READY"
echo "RegMix data preparation is complete."
