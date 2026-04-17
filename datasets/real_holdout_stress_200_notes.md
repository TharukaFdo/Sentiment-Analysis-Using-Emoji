# Real Holdout Stress 200

This file documents the manually curated stress benchmark in
[real_holdout_stress_200.csv](</H:/docs/Demo/Sentiment-Analysis-Using-Emoji/datasets/real_holdout_stress_200.csv:1>).

## Purpose

The untouched real holdout from `social_media_sentiment_test.csv` is still highly emoji-coded. As a result, the emoji-only pipeline achieves unrealistically perfect scores on that benchmark. This stress set was created to test whether the models still work when the shortcut from emoji to label is reduced.

## Construction Rules

- Every row is derived from an untouched row in `datasets/real_holdout.csv`.
- The main sentiment labels remain `Negative`, `Neutral`, and `Positive`.
- `Sarcastic` source rows stay mapped to `Negative` with `sarcasm_flag=1`.
- The edit changes the emoji signal, not the underlying textual sentiment.
- Each row keeps its source seed id and original text for traceability.

## Stress Types

- `emoji_removed`: removes the sentiment-coded emoji so the model must rely more on text.
- `ambiguous_swap`: replaces the original emoji with a shared or ambiguous emoji.
- `contradictory_swap`: replaces the original emoji with an opposing-polarity emoji while keeping the text unchanged.

## Composition

- Total rows: `200`
- Negative: `70`
- Neutral: `65`
- Positive: `65`
- `emoji_removed`: `42`
- `ambiguous_swap`: `92`
- `contradictory_swap`: `66`

## Evaluation Role

This is a curated robustness benchmark, not a new training source. It should be used as an additional evaluation set alongside:

- the untouched real holdout
- the grouped internal split
- `emoji_sentiment_dataset.csv`
