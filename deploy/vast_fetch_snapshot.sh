#!/usr/bin/env bash
# Fetch the pinned Parquet snapshot with a stall watchdog and byte-exact verification.
#
# `hf download` resumes but has no timeout, so a stalled transfer hangs forever. This
# restarts the transfer whenever on-disk bytes stop growing, and only reports success
# when every shard matches the size the Hub reports.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

manifest="$AMS_AUDIO_DIR/.manifest.json"
stall_seconds="${AMS_STALL_SECONDS:-180}"
poll_seconds="${AMS_POLL_SECONDS:-20}"
max_attempts="${AMS_FETCH_ATTEMPTS:-40}"

mkdir -p "$AMS_AUDIO_DIR"

if [[ ! -s "$manifest" ]]; then
  python - "$AMS_DATASET" "$AMS_REVISION" "$manifest" <<'PY'
import json
import sys

from huggingface_hub import HfApi

dataset, revision, out = sys.argv[1:4]
info = HfApi().repo_info(dataset, repo_type="dataset", revision=revision, files_metadata=True)
files = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".parquet")}
if not files or any(size is None for size in files.values()):
    raise SystemExit("hub did not report sizes for every shard")
json.dump({"sha": info.sha, "files": files}, open(out, "w"), indent=2, sort_keys=True)
print(f"manifest: {len(files)} shards, {sum(files.values()) / 1e9:.2f} GB")
PY
fi

verify() {
  python - "$manifest" "$AMS_AUDIO_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest, root = json.load(open(sys.argv[1])), Path(sys.argv[2])
missing = partial = 0
have = 0
for name, size in manifest["files"].items():
    path = root / name
    actual = path.stat().st_size if path.is_file() else 0
    have += actual
    if actual == 0:
        missing += 1
    elif actual != size:
        partial += 1
total = sum(manifest["files"].values())
print(f"{have / 1e9:.2f}/{total / 1e9:.2f} GB  missing={missing} partial={partial}")
raise SystemExit(0 if missing == 0 and partial == 0 else 1)
PY
}

on_disk_bytes() {
  du -sb "$AMS_AUDIO_DIR" 2>/dev/null | cut -f1
}

for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
  if verify; then
    echo "snapshot complete and byte-exact."
    exit 0
  fi

  # The xet transport is the usual stall suspect; fall back to plain HTTPS after
  # two failed attempts so a bad path cannot wedge the campaign indefinitely.
  if (( attempt > 2 )); then
    export HF_HUB_DISABLE_XET=1
  fi

  echo "fetch attempt $attempt (xet=${HF_HUB_DISABLE_XET:-on})"
  hf download "$AMS_DATASET" \
    --repo-type dataset \
    --revision "$AMS_REVISION" \
    --include "data/*.parquet" \
    --local-dir "$AMS_AUDIO_DIR" \
    --max-workers "${AMS_FETCH_WORKERS:-8}" &
  fetch_pid=$!

  last_bytes="$(on_disk_bytes)"
  idle=0
  while kill -0 "$fetch_pid" 2>/dev/null; do
    sleep "$poll_seconds"
    current="$(on_disk_bytes)"
    if [[ "$current" == "$last_bytes" ]]; then
      idle=$(( idle + poll_seconds ))
      if (( idle >= stall_seconds )); then
        echo "stalled for ${idle}s at $(( current / 1000000 )) MB; restarting transfer"
        pkill -P "$fetch_pid" 2>/dev/null || true
        kill -9 "$fetch_pid" 2>/dev/null || true
        break
      fi
    else
      rate=$(( (current - last_bytes) / poll_seconds / 1000000 ))
      echo "  $(( current / 1000000 )) MB  (~${rate} MB/s)"
      idle=0
      last_bytes="$current"
    fi
  done
  wait "$fetch_pid" 2>/dev/null || true
done

echo "snapshot incomplete after $max_attempts attempts." >&2
verify || true
exit 1
