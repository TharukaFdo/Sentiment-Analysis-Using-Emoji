# Real Holdout Stress 600

This benchmark is the full paired stress version of the untouched real holdout.

Files:

- [real_holdout.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout.csv:1>)
- [real_holdout_stress_600.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_600.csv:1>)

The final stress CSV is the union of:

- [real_holdout_stress_200.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200.csv:1>)
- [real_holdout_stress_200_part2.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200_part2.csv:1>)
- [real_holdout_stress_200_part3.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200_part3.csv:1>)

## Purpose

The untouched real holdout is highly emoji-coded, so emoji-only models can appear unrealistically strong. This paired stress benchmark reduces the shortcut from emoji to label while keeping the underlying sentiment unchanged.

## Construction Rules

- Each stress row maps to one unique row in `real_holdout.csv`.
- The sentiment labels remain unchanged.
- `Sarcastic` source rows remain mapped to `Negative` with `sarcasm_flag=1`.
- The benchmark stays test-only.

## Final Composition

- Rows: `600`
- Negative: `319`
- Neutral: `121`
- Positive: `160`

Stress-type counts:

- `ambiguous_swap`: `329`
- `contradictory_swap`: `164`
- `emoji_removed`: `107`

## Evaluation Role

Use this as the primary robustness benchmark alongside:

- the untouched real holdout
- the grouped internal split
- `emoji_sentiment_dataset.csv`
