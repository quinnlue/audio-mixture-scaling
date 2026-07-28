#!/usr/bin/env bash
# Prove the node on one shard before committing to the full 58 GB snapshot.
# Downloads one Parquet shard, packs a throwaway blob, and trains 200 steps.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

smoke="${AMS_SMOKE_DIR:-$AMS_DATA/smoke}"
shard="${AMS_SMOKE_SHARD:-data/part_00-00000-of-00001.parquet}"
mkdir -p "$smoke"
cd "$AMS_REPO"

hf download "$AMS_DATASET" \
  --repo-type dataset \
  --revision "$AMS_REVISION" \
  --include "$shard" \
  --local-dir "$smoke/audio" \
  --max-workers 16

# The packer requires every label to appear in the source, so restrict the
# label table to this shard's paths.
python - "$smoke" "$shard" "$AMS_CLUSTER_LABELS" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

smoke, shard, labels_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
paths = set(pq.read_table(smoke / "audio" / shard, columns=["path"]).column("path").to_pylist())
labels = pq.read_table(labels_path)
subset = labels.filter(pa.array([p in paths for p in labels.column("path").to_pylist()]))
if not subset.num_rows:
    raise SystemExit("shard paths do not intersect the cluster labels")
pq.write_table(subset, smoke / "cluster_labels.parquet")
print(f"smoke labels {subset.num_rows} rows, {len(set(subset.column('cluster').to_pylist()))} clusters")
PY

python -m ams.cli.prep blob \
  --audio-source "$smoke/audio" \
  --audio-revision "$AMS_REVISION" \
  --cluster-labels "$smoke/cluster_labels.parquet" \
  --output-dir "$smoke/blob" \
  --batch-size 2048 \
  --overwrite
cp "$AMS_DISTRIBUTIONS" "$smoke/blob/distributions.json"
python -m ams.data.validate --dataset-dir "$smoke/blob"

rm -rf "$smoke/run"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" PYTHONUNBUFFERED=1 torchrun \
  --standalone \
  --nproc_per_node=1 \
  -m ams.cli.pretrain \
  --config "$AMS_CONFIG" \
  --set "data.blob_dir=$smoke/blob" \
  --set "data.batch_size=$AMS_BATCH_SIZE" \
  --set "data.num_workers=${AMS_NUM_WORKERS:-$(ams_num_workers "$(ams_gpu_count)")}" \
  --set "mixture.distributions=$smoke/blob/distributions.json" \
  --set mixture.dist_id=0 \
  --set "mixture.budget=$(( AMS_BATCH_SIZE * 200 ))" \
  --set optim.warmup_steps=10 \
  --set optim.total_steps=200 \
  --set loop.log_every_n_steps=50 \
  --set tracking.wandb_mode=disabled \
  --output-dir "$smoke/run"

python - "$smoke/run" "$AMS_BATCH_SIZE" <<'PY'
import json
import sys
from pathlib import Path

CORPUS_ROWS = 1_912_024

events = [json.loads(l) for l in (Path(sys.argv[1]) / "events.jsonl").read_text().splitlines()]
rates = [e["metrics"]["train/steps_per_second"] for e in events if e["event"] == "metrics"]
status = json.loads((Path(sys.argv[1]) / "status.json").read_text())
steady = sum(rates[1:] or rates) / len(rates[1:] or rates)
steps = CORPUS_ROWS // int(sys.argv[2])
print(f"\nsmoke {status['status']} at step {status['optimizer_step']}, {steady:.2f} steps/s")
print(f"projected full proxy run: {steps} steps, {steps / steady / 3600:.2f} h per 1-epoch trial")
PY
