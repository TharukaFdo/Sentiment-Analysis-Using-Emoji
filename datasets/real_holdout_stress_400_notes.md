# Real Holdout Stress 400

This benchmark is the union of:

- [real_holdout_stress_200.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200.csv:1>)
- [real_holdout_stress_200_part2.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200_part2.csv:1>)

## Purpose

The untouched `real_holdout.csv` remains highly emoji-coded, so emoji-only models can perform unrealistically well on it. This paired stress benchmark weakens the shortcut from emoji to sentiment while keeping the underlying text label unchanged.

## Construction

- Every stress row is manually derived from a unique row in `datasets/real_holdout.csv`.
- The two parts together cover `400` unique source rows.
- The final label distribution matches the existing `real_holdout_400.csv` benchmark:
  - Negative: `212`
  - Neutral: `81`
  - Positive: `107`

## Stress Types

- `emoji_removed`
- `ambiguous_swap`
- `contradictory_swap`

Final counts:

- `ambiguous_swap`: `218`
- `contradictory_swap`: `114`
- `emoji_removed`: `68`

## Evaluation Role

Use this as a robustness benchmark alongside:

- untouched `real_holdout.csv`
- stratified `real_holdout_400.csv`
- external `emoji_sentiment_dataset.csv`
