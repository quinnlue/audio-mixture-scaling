#!/usr/bin/env bash
# Measure real throughput from the Hugging Face CDN before committing to 58 GB.
# Rented nodes vary by three orders of magnitude; this takes 30 seconds and has
# caught a node that would have needed 12 days for the snapshot.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"

seconds="${AMS_NETCHECK_SECONDS:-30}"
url="https://huggingface.co/datasets/$AMS_DATASET/resolve/$AMS_REVISION/data/part_00-00000-of-00001.parquet"

# curl's -w output has no trailing newline; without one `read` returns EOF and
# `set -e` would abort before anything is reported.
read -r code bytes speed < <(
  curl -sS -L --max-time "$seconds" -r 0-536870911 -o /dev/null \
    -w '%{http_code} %{size_download} %{speed_download}\n' "$url" || true
) || true

# curl reports these as floats, and can emit a trailing CR; bash arithmetic takes
# integers only, so truncate at the first non-digit.
speed="${speed:-0}"
bytes="${bytes:-0}"
speed="${speed%%[!0-9]*}"
bytes="${bytes%%[!0-9]*}"
mb_s=$(( ${speed:-0} / 1000000 ))
echo "http=${code:-none} downloaded=$(( ${bytes:-0} / 1000000 )) MB rate=${mb_s} MB/s"

if (( mb_s < 1 )); then
  echo "FAIL: under 1 MB/s. The 58 GB snapshot would take over a day. Use another node." >&2
  exit 1
fi
eta_minutes=$(( 58000 / mb_s / 60 ))
if (( mb_s < 10 )); then
  echo "WARNING: ${mb_s} MB/s implies about ${eta_minutes} minutes for the snapshot." >&2
  exit 2
fi
echo "OK: snapshot should land in about ${eta_minutes} minutes."
