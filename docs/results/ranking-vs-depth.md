# Does the mixture ranking at 10% of training match the ranking at 100%?

**Verdict: inconclusive by construction.** The experiment ran cleanly and the
machinery works, but the four mixtures selected differ by less than the
run-to-run noise, so there was no stable ranking for an early checkpoint to
agree with. The depth curve below measures when a run settles into *its own*
outcome, which is a different and much less useful quantity.

## Motivation

If a mixture ranking is already correct at 20% of the training budget, RegMix
candidate screening gets 5× cheaper. The question is whether ranking quality
saturates early or keeps improving to the end.

## Setup

Four mixtures (dists 0–3), one per GPU, retrained from scratch with a model
snapshot exported every 3,983 steps — every 10% of the 39,833-step budget. Seeds
and budget identical to the original proxy runs, so the 100% point is comparable
to existing numbers. 44 snapshots, all evaluated with the `full` preset at eval
seed 0.

Implementation note: `CheckpointConfig.export_every_n_steps` already existed in
the codebase but nothing read it. It is now wired to an `on_export` callback
(commit `4f9dc26`), so export cadence is independent of resume cadence.

Artifacts: `quinnlue/eat-regmix-hear-eval` → `depth/`;
`quinnlue/eat-regmix-proxy-64` → `depth-snapshots/`.

## Aggregate score by depth

| depth | dist-0 | dist-1 | dist-2 | dist-3 | ranking |
|---|---|---|---|---|---|
| 10% | 0.5128 | 0.4971 | 0.4961 | 0.4940 | d0 > d1 > d2 > d3 |
| 20% | 0.5575 | 0.5331 | 0.5582 | 0.5470 | d2 > d0 > d3 > d1 |
| 30% | 0.5483 | 0.5715 | 0.5672 | 0.5687 | d1 > d3 > d2 > d0 |
| 40% | 0.5657 | 0.5814 | 0.5837 | 0.5763 | d2 > d1 > d3 > d0 |
| 50% | 0.5811 | 0.5857 | 0.5889 | 0.5784 | d2 > d1 > d0 > d3 |
| 60% | 0.5856 | 0.5882 | 0.5967 | 0.5873 | d2 > d1 > d3 > d0 |
| 70% | 0.5887 | 0.5897 | 0.5970 | 0.5905 | d2 > d3 > d1 > d0 |
| 80% | 0.5904 | 0.5914 | 0.5973 | 0.5898 | d2 > d1 > d0 > d3 |
| 90% | 0.5934 | 0.5903 | 0.5973 | 0.5876 | d2 > d0 > d1 > d3 |
| 100% | 0.5913 | 0.5926 | 0.5980 | 0.5876 | d2 > d1 > d0 > d3 |
| base | 0.7395 | 0.7288 | 0.7314 | 0.7375 | d0 > d3 > d2 > d1 |

The ordering churns through 8 distinct permutations across 10 depths and never
holds stably before the last snapshot.

## Ranking agreement with the final proxy depth

With only 4 mixtures an aggregate Spearman takes just 6 distinct values, so the
per-task column — ranking the 4 mixtures within each of the 13 tasks separately
and pooling — carries the real information.

| depth | ρ aggregate | ρ z-normalised | per-task ρ (mean ± sd) |
|---|---|---|---|
| 10% | +0.200 | +0.400 | 0.222 ± 0.610 |
| 20% | +0.400 | +0.600 | 0.278 ± 0.629 |
| 30% | 0.000 | 0.000 | 0.368 ± 0.622 |
| 40% | +0.800 | −0.800 | 0.650 ± 0.427 |
| 50% | +1.000 | +0.200 | 0.743 ± 0.286 |
| 60% | +0.800 | +0.400 | 0.559 ± 0.566 |
| 70% | +0.400 | +0.400 | 0.738 ± 0.359 |
| 80% | +1.000 | +0.200 | 0.919 ± 0.223 |
| 90% | +0.800 | +0.800 | 0.938 ± 0.222 |

The per-task column rises fairly smoothly from ~0.22 to ~0.94. Taken at face
value that reads as "usable from 40–50%, settled by 80%".

Note the 40% row: aggregate ρ=+0.800 while z-normalised ρ=−0.800 on identical
data. At n=4 these statistics are unstable enough to contradict each other, which
is itself a warning about reading any single row.

## Agreement with the base models

This is the question the proxy actually has to answer, and the answer is nothing:

| depth | ρ aggregate | per-task ρ (mean ± sd) |
|---|---|---|
| 10% | +0.200 | 0.031 ± 0.663 |
| 30% | −0.800 | −0.010 ± 0.707 |
| 50% | −0.600 | −0.082 ± 0.711 |
| 70% | −0.400 | −0.119 ± 0.626 |
| 90% | 0.000 | 0.031 ± 0.679 |
| 100% | −0.600 | 0.131 ± 0.698 |

Per-task pooled agreement sits within ±0.13 of zero at every depth. No depth
predicts the base ranking for these four mixtures — including the final one.

## Why the experiment could not have worked

Two controls explain the result.

**The eval pipeline is fine.** The last two snapshots are 3 optimizer steps
apart — near-identical weights, scored independently. Their aggregate scores
differ by 0.00034 on average (max 0.00092). Scoring is not the problem.

**The mixtures are too close together.** Spread across the four at final depth is
0.0104 (range) / 0.0043 (sd). Independently measured training-seed sd for this
model and recipe is 0.00925. The between-mixture signal is roughly *half* the
run-to-run noise. There is no stable ranking of these four mixtures to detect,
at any depth.

This is confirmed directly in [reproducibility.md](reproducibility.md): retraining
dists 0–3 produced a ranking uncorrelated with the original (ρ=0.000), with
per-run shifts up to 0.0151.

## What this establishes

- **The instrumentation works.** Export cadence, 44-snapshot evaluation, and the
  analysis pipeline all ran end to end, and the 3-step control confirms the
  scoring stage is stable to ±0.0003.
- **Dists 0–3 are not separable at proxy scale.** Three independent lines agree:
  low between-mixture spread, uncorrelated retrain, and inverted base agreement.

## What this does not establish

- **Anything about ranking stability over depth.** The headline curve measures
  convergence to a run-specific outcome, not to a transferable ranking. It should
  not be cited as "the ranking settles by 80%".
- **That early checkpoints are or are not usable.** Untested, because the test as
  built had no signal to detect.

## How to run it properly

1. **Choose separable mixtures.** Pick 4–6 from the *tails* of the 64 by base
   score rather than the first four by index. The 12 paired mixtures span sd
   0.00963 against 0.0043 for dists 0–3 — more than double the signal, for free.
2. **Replicate.** 2–3 seeds per mixture per depth, so each depth's score is a
   mean rather than a single draw.
3. **Keep the 3-step control.** It cost nothing and cleanly separated eval noise
   from training noise.

Estimated cost at the measured ~1 h per proxy run on a 5090: 6 mixtures × 3 seeds
= 18 runs ≈ 4.5 h on 4 GPUs, plus ~1 h to evaluate 198 snapshots.
