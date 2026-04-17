# Current Findings

## Scope of This Summary

This summary focuses on the current finalized research setup:

- the `24,000`-row scripted hybrid dataset: `datasets/social_media_sentiment_dataset_v4.csv`
- the separated untouched real holdout: `datasets/real_holdout.csv`
- the final `600`-row manual stress benchmark: `datasets/real_holdout_stress_600.csv`
- the latest robustness-oriented training run: `runs/20260417_232537_dataset_v4_30epochs_gated_invariant`

## Final Model Direction

The final combined model does not use a naive fusion setup. It includes robustness-oriented changes intended to reduce over-reliance on the emoji branch:

- emoji branch dropout
- reduced emoji influence in fusion
- learned confidence gating for emoji features
- stronger emoji corruption during training
- invariance training against emoji swaps and emoji removals

This model direction was adopted because the earlier combined model still relied too heavily on emoji shortcuts.

## Training Setup

The latest run used:

- `30` epochs
- local script-based training instead of the notebook workflow
- GPU acceleration in the local `dl312` environment
- grouped splitting by `phrase_key`
- the `24,000`-row hybrid dataset as the main training source

## Main Quantitative Results

### Internal Grouped Test Split

- `text`: accuracy `0.9909`, weighted F1 `0.9909`
- `emoji`: accuracy `0.5639`, weighted F1 `0.5627`
- `combined`: accuracy `0.9954`, weighted F1 `0.9954`

### Untouched Real Holdout

- `text`: accuracy `0.9700`, weighted F1 `0.9698`
- `emoji`: accuracy `1.0000`, weighted F1 `1.0000`
- `combined`: accuracy `0.9967`, weighted F1 `0.9967`

### Manual Stress-600 Benchmark

- `text`: accuracy `0.9700`, weighted F1 `0.9698`
- `emoji`: accuracy `0.1633`, weighted F1 `0.1634`
- `combined`: accuracy `0.9500`, weighted F1 `0.9508`

## What These Results Mean

### Finding 1

The untouched real holdout is still too emoji-coded for emoji-only evaluation to be trusted as a robustness result.

This is shown by:

- `emoji-only` scoring `1.0000` weighted F1 on the untouched holdout
- `emoji-only` collapsing to `0.1634` weighted F1 on the paired manual stress benchmark

This is one of the strongest findings in the current research.

### Finding 2

`text-only` is the most robust pipeline under emoji disruption.

Even after the combined model was strengthened, the `text` pipeline still performs slightly better than the final combined model on `stress_600`.

This means the combined model still picks up some misleading signal from the emoji channel under stress conditions.

### Finding 3

The final combined model improved meaningfully compared with earlier fusion attempts, but it still does not surpass text-only robustness.

This is an important result because it supports an honest claim:

`better multimodal fusion helps, but text-only sentiment analysis can remain more reliable when emoji information becomes ambiguous or contradictory`

### Finding 4

The redesigned `24,000`-row hybrid dataset is doing useful work.

Compared with the earlier synthetic setup:

- the emoji-only branch is no longer unrealistically strong on the grouped internal split
- the stress benchmark exposes shortcut behavior clearly
- the curves and evaluation behavior are more believable

## Stress-600 Confusion Pattern

For the final combined model on `stress_600`, the confusion matrix is:

```text
[[292,   0,  27],
 [  0, 121,   0],
 [  3,   0, 157]]
```

This shows:

- `Neutral` is handled perfectly in this benchmark
- most remaining errors are `Negative -> Positive`
- the combined model still gets pulled by misleading emoji in a subset of negative cases

By comparison, the `text` model remains more stable on the same benchmark.

## Research Interpretation

The current study now supports a publishable central interpretation:

`emoji-only sentiment analysis can look very strong on emoji-coded data, but its apparent strength is highly sensitive to dataset design. When emoji shortcuts are weakened, text-based models remain more robust, and robust multimodal fusion helps but does not automatically outperform text-only modeling.`

## What This Means for the Paper

The paper should now be written as:

- a controlled comparison of `text`, `emoji`, and `text+emoji`
- a study with a redesigned hybrid dataset strategy
- a study that separates untouched real evaluation from training generation
- a robustness study using a paired manual stress benchmark

It should not be written as a paper claiming broad real-world superiority for emoji-only sentiment analysis.

## Current Bottom Line

At the current stage, the research has strong enough findings to support paper writing.

The most important outcome is not that one model won every benchmark. The most important outcome is that the redesigned data and evaluation setup exposed the limits of emoji-only sentiment analysis and produced a much more defensible comparison.
