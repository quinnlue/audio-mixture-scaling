# Does the proxy ranking predict the base ranking?

**Verdict: suggestive but underpowered.** The best estimate is ρ=+0.594
(p=0.042, n=12), which is positive, of a plausible magnitude, and consistent
with the RegMix hypothesis. It is also analysis-dependent, not pre-registered,
and not robust to subsetting. It should be treated as motivation for a properly
powered test, not as evidence that transfer works.

## Setup

Twelve mixtures were trained at both scales:

- **Proxy** — 4 layers, 144 dim, 39,833 optimizer steps, batch 48
- **Base** — 12 layers, 768 dim, same token budget, global batch 48

Mixture ids 0–3 and 42–49. Both families evaluated with the same HEAR protocol,
`full` preset, eval seed 0, aggregate over 13 tasks.

Artifacts: `quinnlue/eat-regmix-hear-eval` → `full_preset/` (24 summaries);
`artifacts/proxy-base-correlation/` (earlier `fast`-preset pass, per-task).

## Results

| Aggregation | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|
| z-normalised per task | **+0.594** | **0.042** | +0.676 | 0.016 |
| raw unweighted mean | +0.168 | 0.602 | — | — |

Under the earlier `fast` preset the same 12 pairs gave z-normalised ρ=+0.462
(p=0.131) and raw ρ=+0.308. The critical |ρ| for p<0.05 at n=12 is ≈0.587, so
the headline result clears the threshold by 0.007.

Spreads under `full`: proxy aggregate sd 0.00963, base aggregate sd 0.00666,
against eval-seed noise of 0.00171 — so both families do separate mixtures well
above *eval* noise.

## Why z-normalisation, and why that matters

The raw mean gives every task equal weight in score units. Tasks with wide score
ranges (several small ones here) therefore dominate. Z-normalising each task
across the 12 mixtures before averaging equalises their influence.

This is a defensible choice, and it is standard. But it was **not fixed in
advance**, and it is the difference between ρ=+0.168 (p=0.602) and ρ=+0.594
(p=0.042) on the same data. Two aggregation rules × two eval recipes gives four
analysis paths that were all computed; one of them produced p<0.05. That is a
garden-of-forking-paths problem and it is the single biggest reason not to lean
on this result.

In partial defence: the recipe change was decided on *separate* data (a fixed
checkpoint and the seed replicates, see [measurement-noise.md](measurement-noise.md)),
on a variance criterion, without reference to the transfer correlation. So the
recipe choice is not circular. The aggregation choice is the exposed one.

## The subsetting problem

The 12 pairs split into two groups by construction: dists 0–3 and dists 42–49.
Computed separately on dists 0–3 alone, the proxy ranking is *inverted* relative
to base:

| Comparison | ρ |
|---|---|
| Original proxy vs base, dists 0–3 only | **−0.800** |
| All 12 pairs, z-normalised | +0.594 |

So the positive result is carried entirely by dists 42–49. A correlation that
flips sign on a third of the sample is fragile. This is expected when the effect
is small relative to noise — subsets of 4 are nearly uninformative — but it means
the n=12 estimate has much wider real uncertainty than its nominal p-value
implies.

## Per-task view, and multiple comparisons

From the earlier `fast`-preset pass over 15 tasks
(`artifacts/proxy-base-correlation/correlation-statistics.csv`):

| Task | Spearman ρ | raw p | BH q |
|---|---|---|---|
| GTZAN | +0.594 | 0.042 | 0.444 |
| NSynth Pitch | +0.472 | 0.122 | 0.444 |
| Beijing Opera | +0.423 | 0.171 | 0.444 |
| ESC-50 | +0.402 | 0.195 | 0.444 |
| GTZAN Music/Speech | −0.174 | 0.588 | 0.749 |
| LibriCount | −0.035 | 0.913 | 0.913 |

**No task survives Benjamini–Hochberg correction** — the smallest q is 0.444.
Two tasks (`dcase2016_task2`, `maestro`) produced no paired values at all, and
`gunshot_triangulation` was constant, leaving 12 usable.

The per-task correlations are mostly positive, which is mildly encouraging as a
consistency check — if transfer were absent you would expect them scattered
around zero, and **10 of 12 usable tasks are positive** (only LibriCount and
GTZAN Music/Speech are negative). But none is individually significant after
correction, and the 12 task correlations are not independent of each other, so
this is weaker than a 10-of-12 sign test would suggest.

## What this establishes

- **The sign is positive across most tasks and both aggregation rules.** Weak
  evidence, but it is consistent rather than contradictory.
- **Both scales separate mixtures well above eval noise.** The measurement is
  not the limiting factor any more.

## What this does not establish

- **That proxy ranking predicts base ranking.** p=0.042 at n=12, on a
  post-hoc-selected aggregation, that inverts on a subset, is not a result to
  build on. A pre-registered replication could easily return nothing.
- **Any effect size.** The confidence interval on ρ at n=12 spans roughly 0.02
  to 0.87. That is compatible with "barely useful" and "very useful" alike.
- **That the proxy is the right size.** No proxy-capacity sweep was run, so
  nothing here says a 4-layer model is a good or bad choice of proxy.

## What would settle it

The binding constraint is not more mixture pairs at n=1 each — it is that a
single run is a noisy estimate of its own mixture (training-seed sd 0.00925 vs
between-mixture sd 0.00963). Averaging ~4 proxy runs per mixture would halve the
noise on the x-axis. Combined with the existing 12 base runs that would give a
genuinely powered test at a cost of roughly 48 additional proxy runs.

Fixing the aggregation rule in writing *before* running it is free and would
remove the main objection to the current number.
