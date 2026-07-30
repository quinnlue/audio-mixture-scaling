# How much of the signal was the instrument?

**Verdict: established.** The probe recipe originally used for evaluation
injected more variance than the mixtures under study produced. This is the most
solid finding of the campaign, and it invalidates the first pass at every other
question.

## Why this was investigated

The first scale-transfer pass gave a weak, non-significant correlation
(ρ=+0.462, p=0.131, n=12). A weak correlation has two explanations that look
identical from the outside: the effect is absent, or the measurement is too noisy
to see it. Distinguishing them requires measuring the instrument directly, which
means scoring *the same weights* repeatedly and watching the answer move.

## Experiment 1 — eval-seed noise on fixed weights

One proxy checkpoint, held constant. Seven probe recipes. Four evaluation seeds
each. The embedding cache was shared across all runs, so the encoder forward pass
happened once and only the probe-fitting stage repeated — any spread observed is
probe noise, not encoder noise.

Recipes varied three knobs: cross-validation folds (`max_folds` 1 vs all), probe
training length (`probe_epochs` 10/20/40), and which predictors ran (linear, kNN,
or both).

**Result:** the `fast` preset in use at the time — a single fold, 10 probe
epochs — produced an eval-seed standard deviation of **0.02182** in the aggregate
score. The best recipe reduced this roughly **8×**.

For scale, the spread across all 64 different mixtures under that same `fast`
preset was **0.01730**. The measurement was noisier than the thing being
measured: **the eval recipe alone accounted for about 83% of the apparent spread
between mixtures.**

Under the chosen low-noise recipe (`full`: all folds, 20 probe epochs),
eval-seed sd falls to **0.00171**.

Artifact: `quinnlue/eat-regmix-hear-eval` → `recipe_variance/recipe_results.json`

## Experiment 2 — low variance is not the goal

A recipe can be stable and useless: reporting a constant has zero variance. The
quantity that matters is the ratio of between-mixture spread to noise. Three
recipes were re-run measuring both halves — spread over checkpoints trained on
*different* mixtures, and spread over checkpoints differing only in *training
seed*.

| Recipe | mixture sd / training-seed sd |
|---|---|
| R1 `fast` baseline | 1.37 |
| R3 all folds, 20 epochs | 0.98 |
| R6 all folds, kNN only | 1.09 |

**This table is easy to misread.** The `fast` preset scores highest, but not
because it separates mixtures better — it is the ratio's denominator that moved.
Reducing eval noise exposed how large the *training-seed* noise underneath it
had always been. The honest reading: once eval noise is removed, between-mixture
differences and training-seed differences are roughly the same size, at every
recipe tested.

Artifact: `quinnlue/eat-regmix-hear-eval` → `recipe_snr/snr_results.json`

## The noise budget

Assembled from separate measurements, all on the aggregate over 13 tasks. These
come from different experiments and are not all equally precise — see caveats.

| Source | sd | Measured how |
|---|---|---|
| Eval seed, `full` recipe | 0.00171 | 1 checkpoint, 4 eval seeds |
| Eval seed, `fast` recipe | 0.02182 | 1 checkpoint, 4 eval seeds |
| Training seed, `full` recipe | 0.00925 | 4 seeds, same mixture, same node |
| Training seed, `fast` recipe | 0.01898 | same runs, `fast` scoring |
| Between mixtures, 12 pairs (proxy) | 0.00963 | 12 mixtures, `full` |
| Between mixtures, 12 pairs (base) | 0.00666 | 12 mixtures, `full` |
| Between mixtures, dists 0–3 only | 0.0043 | 4 mixtures, `full` |

The headline: **under the good recipe, training-seed noise (0.00925) is about
the same size as the between-mixture spread (0.00963).** A single proxy run is
roughly a coin flip's worth of evidence about its mixture.

## What this establishes

- **The `fast` preset was unfit for ranking mixtures.** Directly measured, large
  effect, replicated across 7 recipes. High confidence.
- **Eval noise is now well below the effect of interest.** 0.00171 against
  0.00963 is a 5.6× margin. High confidence.
- **Training-seed noise is now the binding constraint.** Measured at 0.00925 on
  4 replicates.

## What this does not establish

- **The exact training-seed sd.** It rests on 4 replicates of a single mixture.
  A standard deviation from n=4 has roughly ±40% uncertainty, and it may differ
  by mixture. Treat 0.00925 as an order of magnitude, not a constant.
- **That `full` is optimal.** It was the best of seven candidates on one
  checkpoint. No claim that a better recipe does not exist.
- **That the 13-task unweighted mean is the right aggregate.** It was chosen for
  simplicity. The per-task tables show task-level noise varies widely, so a
  reliability-weighted aggregate would likely do better and was never tried.

## Consequence for experiment design

With between-mixture sd ≈ training-seed sd, averaging *k* runs per mixture
improves the ratio by √k. To get the mixture effect to twice its standard error
needs roughly 4 runs per mixture. For a 64-mixture RegMix regression that is 256
proxy runs — about 64 GPU-hours at the measured ~1 h/run on a 5090, which is
affordable and was not budgeted for in the original design.

The alternative is to widen the signal rather than shrink the noise: select
candidate mixtures that are far apart in mixture space instead of sampling 64
Dirichlet draws that mostly resemble each other.
