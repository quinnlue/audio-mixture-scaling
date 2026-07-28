#!/usr/bin/env bash
# One-screen campaign state: per-run progress, throughput, ETA, and GPU load.
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/vast_env.sh"
ams_activate

python - "$AMS_RUN_ROOT" "$AMS_READY" <<'PY'
import json
import sys
import time
from pathlib import Path

run_root, ready = Path(sys.argv[1]), Path(sys.argv[2])
print(f"blob      {'ready' if ready.is_file() else 'NOT READY (prep still running)'}")

rows = []
for run in sorted(run_root.glob("dist-*"), key=lambda p: int(p.name.split("-")[1])):
    status_path, events_path = run / "status.json", run / "events.jsonl"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    step, rate, loss, age = status.get("optimizer_step", 0), None, status.get("last_loss"), None
    if events_path.is_file():
        for line in reversed(events_path.read_text().splitlines()):
            event = json.loads(line)
            if event["event"] == "metrics":
                step = event["optimizer_step"]
                rate = event["metrics"]["train/steps_per_second"]
                loss = event["metrics"]["train/loss"]
                age = time.time() - event["time_unix"]
                break
    rows.append((run.name, status.get("status", "pending"), step, rate, loss, age))

total = 0
for path in run_root.glob("dist-*/config.yaml"):
    import yaml

    config = yaml.safe_load(path.read_text())
    total = config["optim"]["total_steps"] or (
        config["mixture"]["budget"] // config["data"]["batch_size"] * config["loop"]["max_epochs"]
    )
    break

print(f"campaign  {run_root}  total_steps={total or '?'}")
print(f"{'run':<10}{'status':<10}{'step':>8}{'pct':>7}{'steps/s':>9}{'loss':>9}{'eta':>9}  stale")
for name, status, step, rate, loss, age in rows:
    pct = f"{100 * step / total:.0f}%" if total else "-"
    eta = f"{(total - step) / rate / 3600:.1f}h" if rate and total and step < total else "-"
    stale = f"{age:.0f}s" if age is not None else "-"
    rate_s = f"{rate:.2f}" if rate else "-"
    loss_s = f"{loss:.4f}" if loss else "-"
    print(f"{name:<10}{status:<10}{step:>8}{pct:>7}{rate_s:>9}{loss_s:>9}{eta:>9}  {stale}")

done = sum(1 for row in rows if row[1] == "complete")
failed = [row[0] for row in rows if row[1] == "failed"]
print(f"\n{done}/{len(rows)} complete" + (f", FAILED: {', '.join(failed)}" if failed else ""))
PY

echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
if command -v supervisorctl >/dev/null; then
  echo
  supervisorctl status | grep '^ams-' || true
fi
