# VM Runbook: RegMix Proxy Campaigns

How to take a freshly rented GPU node from nothing to N parallel RegMix proxy runs.

Every command here was executed on a live Vast.ai 2xA100-40GB node before this document was
written. Numbers marked *(measured)* came from that node; numbers marked *(extrapolated)*
are per-shard measurements scaled to the full corpus.

## Prompt an agent with this

> Here is a VM: `ssh -p PORT root@HOST`. Follow `docs/VM_RUNBOOK.md` and run 16 proxy runs.

That is the whole prompt. The agent needs nothing else: the runbook resolves GPU count,
loader-worker count, dist-id-to-GPU assignment, and restart policy on its own.

Add qualifiers only to change intent, not mechanics:

> ...run dist ids 16-31 instead. / ...use W&B project `eat-regmix-proxy`. /
> ...the node only has 200 GB of disk.

## Four commands

```bash
ssh -p PORT root@HOST
```

```bash
curl -fsSL https://raw.githubusercontent.com/quinnlue/audio-mixture-scaling/main/deploy/vast_bootstrap.sh | bash
```

```bash
/workspace/audio-mixture-scaling/deploy/vast_netcheck.sh && /workspace/audio-mixture-scaling/deploy/vast_smoke.sh
```

```bash
/workspace/audio-mixture-scaling/deploy/vast_campaign.sh 16
```

Then poll:

```bash
/workspace/audio-mixture-scaling/deploy/vast_status.sh
```

Everything below explains what those do, what they assume, and what to do when one fails.

## What a proxy run is

One RegMix proxy trains the 4-layer / 144-dim EAT proxy with the UFO objective for exactly
one epoch over a fixed budget of 1,912,024 sampled examples.

| Property | Value |
| --- | --- |
| Config | `configs/regmix_proxy.yaml` |
| Optimizer steps | 39,833 (1,912,024 / batch 48, `drop_last`) |
| Precision | bfloat16 |
| VRAM | 7.5 GB at batch 48 *(measured)* |
| Throughput | 9.5 steps/s on one A100-40GB *(measured)* |
| Wall clock | ~1.2 h per run on A100 *(measured rate x 39,833 steps)* |
| Parallelism | one run per GPU; **never DDP** |
| Distributions | 64 candidates, dist ids 0-63 |

Proxy runs are independent single-GPU jobs. A node with G GPUs runs G at a time and queues
the rest. Sixteen runs on 8 GPUs is two sequential waves, not a 16-way job.

Each run reads the same read-only mmap blob, so data is prepared **once per node** and
shared by every worker.

## Host requirements

| Resource | Minimum | Why |
| --- | --- | --- |
| GPU | any CUDA GPU with >= 10 GB | 7.5 GB per run |
| Disk | 130 GB free | 58 GB snapshot + 58 GB blob + runs |
| Disk (tight nodes) | 70 GB free with `AMS_KEEP_PARQUET=0` | snapshot deleted after packing |
| vCPU | >= 8 per concurrent run | Opus decode is the only CPU cost |
| RAM | >= 32 GB | page cache over the mmap blob |
| Storage class | NVMe | ~13 MB/s of random 30 KB reads per run |

`vast_bootstrap.sh` prints all of these and warns when disk is short.

**CPU is the constraint that actually bites.** Throughput per run scales with loader workers
until the GPU saturates:

| Loader workers | steps/s *(measured, A100)* |
| --- | --- |
| 8 | 8.69 |
| 16 | 9.06 |
| 32 | 9.35 |
| 48 | 9.42 |

Eight workers already reach 92% of the ceiling, so **8 vCPU per concurrent run is the
practical floor**. Two concurrent runs on this node measured 9.42 and 8.92 steps/s, i.e. no
meaningful contention. GPU utilization sits at 70-80% *(measured)*, so a faster card
(5090/4090) becomes loader-bound sooner: on an 8xGPU node with 64 vCPU expect roughly
7 workers and ~8 steps/s per run rather than the card's ceiling.

`ams_num_workers` in `deploy/vast_env.sh` encodes this as `nproc / concurrent_runs - 1`,
clamped to `[2, 16]`.

## Step 1: connect and orient

```bash
ssh -p PORT root@HOST
```

On a Vast.ai node, read `/etc/vast-agents-guide.md` first — it documents supervisor, the
Caddy auth edge, port mapping, and what survives a recycle. Two facts matter here:

- **`/workspace` is not automatically persistent.** Check with
  `vast-capabilities | jq '.instance.workspace_is_volume'`. When it is `false`, a
  *recycle* or *destroy* wipes the blob and every checkpoint. Stop/start is safe.
- **Long-running work belongs to supervisor**, not to a bare `&` or a detached shell.
  Supervisor restarts crashed runs; a background shell does not.

The image ships torch and torchcodec in `/venv/main`. Do not install torch, and do not
touch the NVIDIA driver.

```bash
source /venv/main/bin/activate && python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

Verified image state on the test node: Python 3.12.13, torch 2.12.0+cu130,
torchcodec 0.12.0+cu130, ffmpeg present, 256 vCPU, 503 GB RAM.

## Step 2: bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/quinnlue/audio-mixture-scaling/main/deploy/vast_bootstrap.sh | bash
```

Or, if the repo is already on the box, `deploy/vast_bootstrap.sh`. It is idempotent:

1. clones or fast-forwards `AMS_GIT_URL` at `AMS_GIT_REF` into `/workspace/audio-mixture-scaling`;
2. installs the package with `uv pip install -e .` — this adds only pyarrow, soxr, wandb,
   safetensors, and requests, and **leaves the image torch untouched** (verified with
   `uv pip install -e . --dry-run`);
3. validates `distributions.json` and runs a real CUDA allocation;
4. prints the host report used for the sizing decisions above.

Uncommitted local work will not reach the node this way. To ship a dirty tree instead:

```bash
tar czf - deploy src configs | ssh -p PORT root@HOST 'cd /workspace/audio-mixture-scaling && tar xzf -'
```

### Never hold a long job in a foreground SSH session

These sessions drop. A download or training run started in the foreground of
`ssh host 'command'` dies with the connection and leaves a partial transfer behind. Every
long step here runs under supervisor, or detached:

```bash
ssh -p PORT root@HOST 'nohup deploy/vast_smoke.sh > /workspace/smoke.log 2>&1 &'
ssh -p PORT root@HOST 'tail -20 /workspace/smoke.log'
```

## Step 3a: gate on download bandwidth

```bash
deploy/vast_netcheck.sh
```

30 seconds, and it is the single highest-value check in this runbook. Rented nodes vary by
three orders of magnitude on the route to the Hugging Face CDN. Measured on two nodes the
same afternoon: **77 MB/s** on one, **0.056 MB/s** on another. At the latter rate the 58 GB
snapshot needs about twelve days, and no amount of retry logic helps.

```text
http=206 downloaded=512 MB rate=77 MB/s
OK: snapshot should land in about 12 minutes.
```

Exit codes: `0` healthy, `2` under 10 MB/s (usable, but budget the time), `1` under
1 MB/s — **abandon the node and rent another**. A slow CDN route is not correlated with GPU
quality or price; it is worth re-rolling the instance rather than waiting.

## Step 3: smoke test before spending 58 GB

```bash
deploy/vast_smoke.sh
```

45 seconds *(measured)*. It downloads one Parquet shard, restricts the cluster-label table
to that shard, packs a throwaway blob, trains 200 real steps, and prints the projected
full-run wall clock for this specific node:

```text
smoke labels 20000 rows, 20 clusters
smoke complete at step 200, 9.52 steps/s
projected full proxy run: 39833 steps, 1.16 h per 1-epoch trial
```

This exercises every component the campaign needs — HF download, the Opus packer, mmap
reads, torchcodec decode, the fbank/EAT/UFO forward, checkpointing, and the artifact
contract — for 0.6 GB instead of 58 GB. Run it before every campaign on a new node. If the
projected wall clock is wildly worse than ~1.2 h, the node is CPU-starved; fix workers
before launching 16 runs.

## Step 4: launch the campaign

```bash
deploy/vast_campaign.sh 16
```

`vast_campaign.sh` is the only command that needs a decision, and the decision is just
"how many runs". It:

1. detects GPU count G;
2. assigns dist ids round-robin so GPU `g` owns `g, g+G, g+2G, ...`;
3. computes loader workers from `nproc` and G;
4. writes `/workspace/ams-campaign.env`, the single resolved view of the campaign that
   prep, workers, and status all source;
5. writes `/etc/supervisor/conf.d/ams-campaign.conf` and calls
   `supervisorctl reread && supervisorctl update`.

On an 8-GPU node, `vast_campaign.sh 16` produces:

```text
ams-prep     -> downloads + packs the shared blob, then exits 0
ams-gpu-0    -> DIST_IDS="0,8"
ams-gpu-1    -> DIST_IDS="1,9"
...
ams-gpu-7    -> DIST_IDS="7,15"
```

Selecting non-contiguous ids, for a second wave or a retry:

```bash
AMS_DIST_IDS="16,17,18,19,20,21,22,23" deploy/vast_campaign.sh
```

The workers start immediately and block on `$AMS_BLOB_DIR/.ready`, so launching the
campaign before data exists is the normal path — not a race. Prep runs once; all GPU
workers wait, then start together.

### Data preparation, in detail

`ams-prep` runs `deploy/vast_prepare_regmix.sh`:

1. guards free disk (130 GB, or 70 GB when `AMS_KEEP_PARQUET=0`);
2. calls `deploy/vast_fetch_snapshot.sh` (below) to pull `data/*.parquet` from
   `danjacobellis/audioset_opus_24kbps` at revision
   `a725d7cf1fea563c6eb9f6127dbd1d75b294668d` into `/workspace/data/audioset-opus`;
3. asserts exactly 96 shards;
4. `python -m ams.cli.prep blob` packs source-order Opus payloads plus the cluster index
   into `/workspace/data/audioset-regmix-blob`;
5. copies `artifacts/audioset-dasheng-0.6b-k20-r31d6389/regmix-paper-64/distributions.json`
   next to the blob;
6. runs `python -m ams.data.validate --dataset-dir` (decodes head/middle/tail);
7. touches `.ready`.

Facts worth trusting, all checked against the pinned revision:

- 96 shards, 58.23 GB, 1,912,024 rows (95 shards of 20,000 + a final 12,024).
- `cluster_labels.parquet` covers all 1,912,024 rows exactly — delta 0. This matters
  because the packer aborts on any source row without a label.
- The packed blob is ~58 GB; one shard packed at 90 MB/s *(measured)*, so expect
  ~11 minutes for the full pack *(extrapolated)*.
- Download measured 77 MB/s on a single shard; the full snapshot typically lands in
  10-25 minutes depending on host bandwidth *(extrapolated)*.
- `distributions.json` in the artifacts root is byte-identical to the `regmix-paper-64`
  copy; the nested path is the canonical one.

Output layout:

```text
/workspace/data/audioset-regmix-blob/
  opus_blob.bin            ~58 GB, read-only mmap
  opus_offsets.npy         int64 start offsets
  cluster_index.npy        int8 cluster per row
  distributions.json       64 candidate mixtures over 20 clusters
  preprocess_metadata.json rows, num_clusters, blob_bytes, source revision
  .ready                   the barrier every GPU worker waits on
```

Tight disk: `AMS_KEEP_PARQUET=0 deploy/vast_campaign.sh 16` deletes the 58 GB snapshot once
the blob validates, taking the steady-state footprint to ~58 GB.

### Surviving a flaky download

`hf download` resumes interrupted transfers but has **no timeout**, so a stalled connection
hangs forever and the campaign waits behind it. `deploy/vast_fetch_snapshot.sh` wraps it:

- builds a manifest of all 96 shard sizes from the Hub API once;
- runs the transfer detached and polls on-disk bytes every 20 s, printing the running total
  and rate;
- kills and restarts the transfer after `AMS_STALL_SECONDS` (default 180) with no growth —
  the restart resumes, so no bytes are lost;
- sets `HF_HUB_DISABLE_XET=1` from the third attempt, falling back from the xet transport to
  plain HTTPS;
- succeeds only when **every shard matches its Hub-reported size byte for byte**, not merely
  when the file count is right.

Watch it with `tail -f /workspace/ams-logs/prep.log`. Occasional restarts are normal on a
flaky route; more than three, or a rate that collapses and stays down, means the node's
network is the problem rather than the transfer. Re-run `vast_netcheck.sh` to confirm, and
move nodes if it fails.

## Step 5: monitor

```bash
deploy/vast_status.sh
```

```text
blob      ready
campaign  /workspace/runs/regmix-proxy  total_steps=39833
run       status        step    pct  steps/s     loss      eta  stale
dist-0    running      12400    31%     9.52   1.2044     0.8h  2s
dist-1    running      12350    31%     9.48   1.2101     0.8h  1s
dist-8    pending          0     0%        -        -        -  -
...
2/16 complete
```

`stale` is the age of the last metric event. A growing `stale` on a `running` row means the
process died without writing a failure — check `supervisorctl status` and the GPU log.

`status.json` only updates at start, checkpoint, and end, so its `optimizer_step` lags;
`events.jsonl` is the live source and is what this table reads.

Other views:

```bash
supervisorctl status | grep ams-          # per-GPU worker state
tail -f /workspace/ams-logs/gpu-3.log     # one GPU's queue, across its runs
tail -f /workspace/ams-logs/prep.log      # download and packing
nvidia-smi                                # expect ~7.5 GB and 70-80% per run
```

## Failure and resume semantics

This was verified live: a trainer killed with `SIGKILL` mid-run was restarted by supervisor
and resumed from its last checkpoint with no repeated optimizer steps (metrics ran
`... 350, 400` then `450, 500, 550` after the kill at ~step 400).

- Checkpoints land every 9,958 steps (4 per epoch), so a crash costs <= ~17 minutes.
- The worker passes `--resume` automatically when `checkpoints/resume.pt` exists.
- A run whose `status.json` says `complete` is skipped, so re-running the campaign is
  free and idempotent.
- `autorestart=unexpected` with `exitcodes=0` means a worker that drains its queue exits
  cleanly and stays `EXITED`; only failures restart.
- Resume rejects a changed world size or any config change other than extending
  `loop.max_epochs`. Do not edit configs mid-campaign.

To restart one GPU's queue: `supervisorctl restart ams-gpu-3`.
To abandon a run and redo it: `rm -rf /workspace/runs/regmix-proxy/dist-7` then restart its
worker.
To stop everything: `supervisorctl stop 'ams-gpu-*'`.

## Artifacts

Each run writes to `/workspace/runs/$AMS_CAMPAIGN/dist-<id>/`:

```text
config.yaml                  fully resolved config
run.json                     campaign/trial ids, config fingerprint
events.jsonl                 metrics, checkpoint, status events
status.json                  running | complete | failed
console.log                  this run's stdout
checkpoints/resume.pt        latest, with resume.json (sha256, step, world size)
checkpoints/final.pt         on completion
exports/step_*/model.safetensors
results/hear_metrics.jsonl   inventory-only record for proxies
```

Collect a finished campaign off the node before destroying it — on a non-volume
`/workspace`, destroy wipes everything:

```bash
tar czf - -C /workspace/runs regmix-proxy --exclude='*.pt' | ssh you@host 'cat > regmix-proxy.tar.gz'
```

Exclude `*.pt` unless the checkpoints are wanted; `exports/` carries the safetensors.

## Knobs

All are environment variables read by `deploy/vast_env.sh`. Precedence is explicit
environment, then `/workspace/ams-campaign.env`, then defaults.

| Variable | Default | Use |
| --- | --- | --- |
| `AMS_CAMPAIGN` | `regmix-proxy` | names the run root and W&B run ids |
| `AMS_DIST_IDS` | — | explicit id list, overrides the count argument |
| `AMS_RUN_ROOT` | `/workspace/runs/$AMS_CAMPAIGN` | output root |
| `AMS_BLOB_DIR` | `/workspace/data/audioset-regmix-blob` | shared blob |
| `AMS_KEEP_PARQUET` | `1` | `0` deletes the snapshot after packing |
| `AMS_NUM_WORKERS` | auto | override loader workers per run |
| `AMS_BATCH_SIZE` | `48` | lower only if VRAM-constrained |
| `AMS_CONFIG` | `configs/regmix_proxy.yaml` | alternate config path |
| `AMS_WANDB` | `disabled` | `online` also needs `WANDB_API_KEY` |
| `AMS_STALL_SECONDS` | `180` | seconds of zero progress before restarting the transfer |
| `AMS_FETCH_WORKERS` | `8` | parallel shard downloads |
| `AMS_FETCH_ATTEMPTS` | `40` | transfer restarts before giving up |
| `AMS_DISTRIBUTIONS` | `artifacts/.../regmix-paper-64/distributions.json` | candidate set |
| `AMS_GIT_REF` | `main` | branch or SHA to deploy |

Changing batch size changes steps per epoch; `optim.total_steps` and `warmup_steps` in
`configs/regmix_proxy.yaml` are anchored to batch 48, so leave it alone unless the run is
being redefined.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Download crawls at KB/s | bad CDN route on this node, not a client bug | `vast_netcheck.sh`; if it fails, rent another node |
| Download hangs with no progress | stalled xet transfer | the fetcher restarts it automatically; check `prep.log` for restart lines |
| A long command dies mid-way | SSH session dropped | rerun detached under supervisor or `nohup`; never foreground long jobs |
| `no kernel image is available` | wheel predates the GPU arch (Blackwell needs CUDA >= 12.8) | use the image torch; never pin an old `--index-url` |
| Workers idle, `blob NOT READY` | prep still running or failed | `tail /workspace/ams-logs/prep.log` |
| `Expected 96 Parquet shards, found N` | interrupted download | rerun; `hf download` resumes |
| `missing/duplicate path` from the packer | wrong `cluster_labels.parquet` or a non-pinned revision | confirm `AMS_REVISION` and the artifacts dir |
| `Need 130 GB free` | disk too small | `AMS_KEEP_PARQUET=0`, or rent more disk |
| steps/s far below ~9 | too few loader workers | raise `AMS_NUM_WORKERS`, or run fewer GPUs at once |
| `distribution assigns mass to empty clusters` | blob and distributions disagree | repack, or point `AMS_DISTRIBUTIONS` at the matching set |
| worker `FATAL` in supervisor | 100 failed starts | read `/workspace/ams-logs/gpu-N.log`; the underlying error is at the top |
| everything vanished after a restart | instance was recycled, `/workspace` is not a volume | check `workspace_is_volume`; re-bootstrap |

## Without supervisor

On a non-Vast box, the same worker script runs under `nohup` or `tmux`:

```bash
export AMS_STATE=/workspace/ams-campaign.env
for gpu in 0 1 2 3; do
  GPU_ID=$gpu DIST_IDS="$gpu,$((gpu+4)),$((gpu+8)),$((gpu+12))" \
    nohup deploy/vast_run_proxy.sh >/dev/null 2>&1 &
done
```

Run `deploy/vast_prepare_regmix.sh` first, or the workers simply wait for `.ready`. The
loss versus supervisor is automatic restart, so check `vast_status.sh` more often.

## A single run, by hand

For debugging one distribution, skip the campaign layer entirely:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m ams.cli.pretrain \
  --config configs/regmix_proxy.yaml \
  --set data.blob_dir=/workspace/data/audioset-regmix-blob \
  --set data.num_workers=16 \
  --set mixture.distributions=/workspace/data/audioset-regmix-blob/distributions.json \
  --set mixture.dist_id=7 \
  --set tracking.wandb_mode=disabled \
  --output-dir /workspace/runs/manual/dist-7
```

`torchrun --standalone` binds a random free rendezvous port, so many concurrent
single-GPU launches on one node do not collide.
