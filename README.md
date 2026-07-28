# Audio Mixture Scaling

This is the canonical training repository for the RegMix scale-invariance study in audio
self-supervised learning. It contains one pretraining path—EAT with the UFO objective—plus
HEAR evaluation, RegMix sampling, and offline data preparation. It does not import or depend
on the older `encoder` repository.

The repository is intentionally small: configuration is frozen dataclasses composed from
YAML, distributed execution is `torchrun`, and the training loop has one concrete
model/objective pair. Training never imports HEAR, mixture analysis, or preparation code.

## Install

The target image supplies its CUDA-compatible `torch` and `torchcodec` builds:

```bash
python -m pip install -e .
```

For local evaluation, preparation, and development:

```bash
python -m pip install -e ".[eval,prep,dev]"
```

`deploy/provision.sh` is the single standalone bootstrap. Runner campaigns instead install
the exactly pinned, non-CUDA dependencies shown in
`deploy/campaign-regmix.toml.example`.

## Pretrain

Full AudioSet EAT-base training:

```bash
torchrun --standalone --nproc_per_node=4 -m ams.cli.pretrain \
  --config configs/pretrain_as2m.yaml \
  --set data.dataset_dir=/data/audioset \
  --output-dir /runs/eat-base
```

One RegMix proxy:

```bash
torchrun --standalone --nproc_per_node=1 -m ams.cli.pretrain \
  --config configs/regmix_proxy.yaml \
  --set data.blob_dir=/data/audioset_blob \
  --set mixture.distributions=/data/audioset_blob/distributions.json \
  --set mixture.dist_id=7 \
  --output-dir /runs/regmix/dist-7
```

Multiple `--config` files compose left-to-right. Repeated `--set key=value` overrides are
parsed with YAML typing. Every run writes its fully resolved `config.yaml`.

`ams.cli.pretrain` is an ordinary importable Python module and honors `--resume PATH`. It
reads `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` from `torchrun`; there is no runner-specific
branch.

## Fidelity decision

Teacher targets follow the released
`cwx-worst-one/EAT/config/pretraining_AS2M.yaml` order:

1. select the top 12 layers for EAT-base (all four for the proxy);
2. apply per-layer instance normalization on `(B, C, T)`;
3. average layers;
4. apply post-average layer normalization.

In particular, the released config has both `instance_norm_target_layer: true` and
`layer_norm_targets: true`. This corrects the old repository, which averaged raw hidden
states. Previous training baselines are therefore not comparable and are intentionally not
reported as references here. The new proxy baseline remains to be recorded after the first
live target-VM campaign.

The utterance target remains the mean of teacher patch tokens, not teacher CLS.

## Data

Full pretraining reads the revision-pinned
`danjacobellis/audioset_opus_24kbps` Parquet snapshot. `RowGroupShuffleSampler` shuffles
files, row groups, then rows with `seed + epoch`, and assigns whole global batches
round-robin across DDP ranks.

Proxy runs use:

```text
opus_blob.bin
opus_offsets.npy
cluster_index.npy
distributions.json
```

The blob is read-only mmap data. `MixtureSampler` samples example `i` from cluster `c` with
probability `mixture[c] / count[c]`, so requested weights control cluster mass rather than
being biased by cluster size.

Both paths decode with `torchcodec`, average channels, resample through `soxr`, and pad or
truncate to 160,000 samples. The EAT fbank runs in the model on the accelerator.

Validate a runner-materialized Parquet snapshot or blob directory with:

```bash
python -m ams.data.validate --dataset-dir /data/materialized
```

Offline preparation is explicitly separated:

```bash
python -m ams.cli.prep cluster --source OWNER/embeddings --output-dir /data/clusters
python -m ams.cli.prep blob --audio-source /data/audioset \
  --cluster-labels /data/clusters --output-dir /data/audioset_blob
python -m ams.cli.mixture generate --cluster-stats /data/clusters \
  --output /data/audioset_blob/distributions.json
```

## HEAR

HEAR archives are self-contained Parquet files built with:

```bash
python -m ams.cli.prep hear-parquet --data-root data/HEAR --out-dir data/hear_parquet
```

Evaluate a checkpoint:

```bash
python -m ams.cli.eval_hear \
  --config configs/eat_base.yaml \
  --checkpoint /runs/eat-base/exports/step_00398330/model.safetensors \
  --data-root data/hear_parquet \
  --output-dir results/hear \
  --model-id eat-base
```

Scene-task kNN and linear-probe protocols and frame-level event-task MLP protocols are
implemented. Event evaluation includes target rasterization, post-processing, onset/offset
F-measures, and segment error rate. Loading `quinnlue/eat-base-k16-ema-10ep-v1` should
reproduce `val/hear/score ≈ 0.7917`; that live anchor requires the external checkpoint and
HEAR dataset.

## Runner artifact contract

Rank zero writes:

```text
config.yaml
run.json
events.jsonl
status.json
checkpoints/resume.pt
checkpoints/resume.json
checkpoints/final.pt
results/hear_metrics.jsonl
exports/step_*/model.safetensors
```

`resume.json` includes the resume checkpoint SHA-256, completed epoch, world size, optimizer
step, and resume-compatible config fingerprint. Resume rejects world-size changes and config
changes other than extending `loop.max_epochs`. Mixture proxies still emit an inventory-only
record to the literal `results/hear_metrics.jsonl` runner path.

W&B is rank-zero only, resumes with a stable caller-supplied run ID, and uses
`optimizer_step` as the shared metric axis. Standalone runs can enable background Hub export
uploads with `tracking.hub_repo_id`; this remains off in runner recipes.

## Verification

```bash
python -m pytest -q
ruff check .
mypy src
```

The suite covers configuration and fingerprints, schedule anchors, masking, target
normalization, fbank shape, model/objective forward execution, Parquet/DDP sampling, RegMix
cluster mass, runner artifact parsing, and interrupted/resumed training equality.

The pre-normalization port was run side-by-side with `C:\ml\encoder` for 20 fp32 optimizer
steps. Every loss and every final trainable parameter matched exactly; the dependency-free
trace is committed at `tests/fixtures/legacy_proxy_loss_trace.json`.

The CUDA smoke was also run under bf16 `torchrun` on two RTX 3090s. Both ranks completed the
real fbank/EAT/UFO path with equal gradient norms, rank zero produced the runner artifact
tree, and a one-step checkpoint resumed to a final safetensors file bit-identical to the
uninterrupted two-rank run. A runner-shaped single-GPU invocation on a valid mmap Opus blob
likewise resumed exactly, and `train-runner` accepted its campaign manifest and resume
metadata.

The remaining external-data gates are the released-checkpoint HEAR anchor, the full proxy
baseline under corrected target normalization, and target-campaign throughput.
