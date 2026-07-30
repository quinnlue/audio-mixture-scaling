# RegMix scale-invariance study — results index

Four questions were asked of a 64-mixture RegMix campaign in audio SSL. This
directory records what each experiment actually showed, separating what the data
establishes from what it merely suggests.

| Document | Question | Verdict |
|---|---|---|
| [measurement-noise.md](measurement-noise.md) | How much of the score spread is measurement artifact? | **Established.** The original probe recipe contributed most of the apparent spread between mixtures. |
| [reproducibility.md](reproducibility.md) | Does the same mixture score the same twice? | **Partly established.** Training noise is the same size as the mixture effect. The cleanest test was not completed. |
| [scale-transfer.md](scale-transfer.md) | Does the proxy ranking predict the base ranking? | **Suggestive, underpowered.** ρ=+0.594 at n=12, but analysis-dependent and not robust to subsetting. |
| [ranking-vs-depth.md](ranking-vs-depth.md) | Does the ranking at 10% depth match the ranking at 100%? | **Inconclusive by construction.** The four mixtures chosen were closer together than the noise floor. |

## Confidence vocabulary

Used consistently throughout these documents:

- **Established** — directly measured, with a control that rules out the obvious
  alternative explanation.
- **Suggestive** — measured, but underpowered, analysis-dependent, or lacking a
  control. Would not survive a pre-registration.
- **Unresolved** — measured but confounded, or never measured.

## The one-paragraph summary

The campaign set out to test whether a small proxy model can rank data mixtures
the way a larger model would. It ended up mostly measuring its own instruments.
The strongest, most reproducible finding is methodological: the evaluation recipe
originally in use injected more variance than the mixtures themselves produced,
and fixing it changed the headline correlation from non-significant to
marginally significant. The scale-transfer result that emerged is real but weak
and rests on analysis choices made after seeing the data. The depth experiment
could not answer its question because the mixtures selected for it differ by less
than the run-to-run noise. Nothing here refutes the RegMix hypothesis; nothing
here establishes it either. The main deliverable is a much better calibrated
sense of how many runs a real test would need.

## What was actually run

| Campaign | Scale | Count | Artifact |
|---|---|---|---|
| RegMix proxy sweep | 4-layer, 144-dim | 64 mixtures | `quinnlue/eat-regmix-proxy-64` |
| Base pairs | 12-layer, 768-dim | 12 mixtures | `quinnlue/eat-regmix-base-64` |
| Training-seed replicates | proxy | 4 seeds, dist-0 | `quinnlue/eat-regmix-proxy-64` |
| Depth snapshots | proxy | 4 mixtures × 11 depths | `.../depth-snapshots/` |
| HEAR evaluations | — | 76 + 24 + 44 checkpoints | `quinnlue/eat-regmix-hear-eval` |

All training used `danjacobellis/audioset_opus_24kbps` pinned at revision
`a725d7cf1fea563c6eb9f6127dbd1d75b294668d`, with 20 embedding clusters from
`artifacts/audioset-dasheng-0.6b-k20-r31d6389/`.

## A note on the evaluation suite

HEAR ships 15 tasks here, but only 13 enter any aggregate in these documents:

- `dcase2016_task2` and `maestro` are event tasks whose results are written under
  `event_mlp` rather than `predictors`, and both return exactly 0.0 onset
  F-measure for all 76 checkpoints. They are excluded everywhere.
- `gunshot_triangulation` is retained in the 13 but is near-degenerate; it was
  reported as `constant input; correlation undefined` in the earlier per-task
  correlation pass.

Every "aggregate score" in these documents is an unweighted mean over those 13
task scores, each task contributing its primary metric (linear probe preferred
over kNN; `top1_acc`, then `mAP`, then `aucroc`).
