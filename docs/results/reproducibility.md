# Does the same mixture score the same twice?

**Verdict: partly established.** Training noise is the same size as the effect
under study — this is supported by two independent measurements. The specific
claim that *nothing but kernel nondeterminism* separates two runs is
**unresolved**: the cleanest test was launched but not completed.

This question gates everything else. If one mixture trained twice lands further
apart than two different mixtures, no ranking built from single runs means
anything.

## Measurement 1 — varying the training seed (clean)

Four proxy runs of dist-0 differing in `runtime.seed` and `data.seed` (both
varied, since holding either fixed understates the variance a real re-run sees).
Same node, same blob, same code, same libraries.

**Training-seed sd = 0.00925** on the 13-task aggregate under the `full` recipe.
(The same runs scored under `fast` gave 0.01898, most of which was eval noise.)

Against between-mixture sd of 0.00963 across the 12 paired mixtures, this is a
signal-to-noise ratio of almost exactly **1.0**.

This measurement has no cross-node confound. It is the load-bearing number.

Artifact: `quinnlue/eat-regmix-hear-eval` → `recipe_snr/snr_results.json`

## Measurement 2 — retraining with the *same* seed (confounded)

The four depth runs used seed 0, identical to the original dist-0…3 proxy runs,
making them nominal replicates.

| dist | original | rerun | Δ |
|---|---|---|---|
| 0 | 0.5795 | 0.5913 | +0.0118 |
| 1 | 0.5910 | 0.5926 | +0.0016 |
| 2 | 0.5829 | 0.5980 | +0.0151 |
| 3 | 0.5905 | 0.5876 | −0.0029 |

Mean |Δ| = 0.00786, max 0.01505, against a between-mixture sd of 0.0043–0.0057
for these four. The induced rankings are uncorrelated (ρ = 0.000).

Two cautions on reading this. **ρ=0.000 at n=4 is close to uninformative** —
Spearman on 4 items takes 6 values and has enormous sampling variance. The
informative quantity is the magnitude of Δ, not the correlation. And **all four
reruns scored higher than their originals** (mean +0.0064), which looks like a
systematic offset rather than symmetric noise; n=4 cannot distinguish a real
offset from chance.

### What was verified about these being replicates

| Checked | Result |
|---|---|
| Training config | Identical except `export_every_n_steps` and tracking labels |
| Seeds | `runtime.seed=0`, `data.seed=0` in both |
| `num_workers` | 32 in both — no dataloader-sharding difference |
| Dataset revision | Same pin, `a725d7cf…` |
| Blob ordering | Built from `sorted(glob("*.parquet"))`, so row order is deterministic and seed 0 draws the same examples |
| Eval path | The two comparisons used different config files; both load to byte-identical `ModelConfig` dataclasses. Same preset, eval seed, data root |

### What could not be verified

- **Library and driver versions on the original node.** That instance was
  destroyed. A different torch build can change kernel selection, matmul/TF32
  defaults, and fused-op behaviour — plausibly more than scheduling
  nondeterminism.
- **Exact code revision.** `run.json` records `code_revision: "local"`, not a
  commit SHA.
- **Blob byte-identity across nodes.** Deterministic by construction, but neither
  blob was hashed; `preprocess_metadata.json` records only `verified_samples: 3`.

So this measurement is **corroborating, not clean**. It is consistent with
Measurement 1 in magnitude (0.0079 vs 0.00925), which is the main reason to
believe it is measuring training noise rather than an environment difference —
but that is an argument, not a control.

## The experiment that would have settled it

Four runs of dist-0, identical seed, on one node, same blob, same code, same
torch — isolating run-to-run nondeterminism with everything else pinned. Combined
with Measurement 1 (same node, *varying* seed) it would have decomposed training
noise into a seed component and a nondeterminism component.

It was launched and abandoned: the host dropped SSH about two minutes after the
runs started, and the work was not resumed. **No result.** The scripts are in
place ([vast_depth_runs.sh](../../deploy/vast_depth_runs.sh) with
`AMS_TRIAL_LABEL`, commit `ac2fc26`) and it costs about one hour on 4 GPUs.

Until it is run, the split between "seed" and "kernel nondeterminism" inside that
0.00925 is unknown. Note this does not affect the practical conclusion — total
training noise is what matters for experiment design, and that is measured.

## What this establishes

- **Training noise ≈ between-mixture signal at proxy scale.** Two independent
  measurements agree (0.00925 seed-varying, 0.0079 same-seed retrain). Single
  runs cannot reliably rank these mixtures.
- **`runtime.deterministic: false` is set** in the proxy config, so bitwise
  reproducibility was never expected.

## What this does not establish

- **The nondeterminism-only floor.** Not measured.
- **Whether the +0.0064 offset between run sets is real.** Could be node,
  library, or chance.
- **That noise is uniform across mixtures.** All replicate measurements used
  dist-0.

## Practical implications

1. **Report proxy scores as means over ≥3 runs**, or state explicitly that a
   single-run score carries ±0.009 uncertainty — comparable to the entire spread
   between candidate mixtures.
2. **Record the environment.** `run.json` should capture the git SHA, `torch.__version__`,
   `torch.version.cuda`, and the driver version. Its `code_revision: "local"` made
   a checkable question uncheckable.
3. **Hash the blob.** A cheap content hash in `preprocess_metadata.json` would
   let any two nodes prove they trained on identical bytes.
