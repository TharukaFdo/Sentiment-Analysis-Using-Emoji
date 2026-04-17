# Dataset Construction and Reasoning

## Files in Scope

This current research stage is centered on the following files:

- `social_media_sentiment_test.csv`
- `datasets/real_generation_seed.csv`
- `datasets/real_holdout.csv`
- `datasets/social_media_sentiment_dataset_v4.csv`
- `datasets/social_media_sentiment_dataset_v4_manifest.json`
- `datasets/real_holdout_stress_600.csv`
- `datasets/real_holdout_stress_600_notes.md`

Intermediate `200`-row and `400`-row stress sets were only stepping stones during development. The final manual stress benchmark for the current research is the `600`-row version.

## Real Source Data

The only real source file available for this project is `social_media_sentiment_test.csv`, which contains `2,000` short social-media style sentiment examples.

This real source was treated as the anchor for the entire redesigned dataset strategy.

## Label Policy

The study keeps a strict 3-class sentiment scheme:

- `Negative`
- `Neutral`
- `Positive`

Rows originally labeled `Sarcastic` in the real source are not discarded and are not turned into a new sentiment class. Instead, they are mapped as follows:

- `Sarcastic -> Negative`
- `sarcasm_flag = 1`

The reasoning is that sarcasm in this dataset usually carries negative underlying sentiment, while the main task must remain a 3-class sentiment problem.

## Separation of the Real Dataset

The `2,000` real rows were separated before generation into two disjoint parts:

- `1,400` rows for generation and training support
- `600` rows for untouched evaluation

These are stored as:

- `datasets/real_generation_seed.csv`
- `datasets/real_holdout.csv`

The holdout rule is strict:

- no holdout row appears in the training corpus
- no generated child of a holdout row appears in the training corpus
- no exact overlap exists between the training corpus and the holdout

According to `datasets/social_media_sentiment_dataset_v4_manifest.json`, the exact overlap count between the final hybrid training dataset and the untouched real holdout is `0`.

### Real Partition Counts by Original Label

- `Negative`: `380` generation seed, `163` holdout, `543` total
- `Neutral`: `281` generation seed, `121` holdout, `402` total
- `Positive`: `375` generation seed, `160` holdout, `535` total
- `Sarcastic`: `364` generation seed, `156` holdout, `520` total

## Scripted Hybrid Training Dataset

The main training corpus is `datasets/social_media_sentiment_dataset_v4.csv`.

### Size and Balance

- total rows: `24,000`
- `Negative`: `8,000`
- `Neutral`: `8,000`
- `Positive`: `8,000`
- unique comments: `24,000`

### Source Types

- `1,400` real seed rows copied from the generation partition
- `22,600` controlled augmentations built from the real seed partition

### Important Columns

The final scripted dataset includes:

- `id`
- `parent_id`
- `comment`
- `sentiment`
- `original_seed_label`
- `sarcasm_flag`
- `hard_case_flag`
- `source_type`
- `generation_signal`
- `phrase_family`
- `split_group`
- `recommended_split`

These columns were kept so the dataset would be traceable, auditable, and usable for grouped splitting.

## Generation Reasoning

The scripted dataset was not built from arbitrary templates. It was built from controlled generation signals designed to address known weaknesses in the earlier synthetic dataset.

### Signals and Why They Were Added

`real_observation`

- keeps direct examples from the real seed source
- anchors the training data in real slang, noise, and emoji usage

`lexical_swap`

- changes wording within the same sentiment family
- reduces dependence on one exact phrase

`orthographic_noise`

- adds repeated letters and small misspellings
- reflects the noisy style already present in real social-media posts

`mixed_resolution`

- creates posts where the sentiment becomes clear late in the sentence
- makes the text channel less trivial and more realistic

`hedged_neutral`

- creates low-affect neutral language such as "just okay" or "kinda normal"
- stops neutral from becoming a weak version of positive

`balanced_neutral`

- creates mixed low-intensity reactions such as "not bad not great"
- improves realism for neutral sentiment

`shared_emoji_overlap`

- uses emojis shared across positive, neutral, and negative examples
- reduces direct emoji-to-label shortcuts

`short_phrase_overlap`

- reuses short social-media phrase shells across different classes
- prevents one phrase family from belonging to only one label

`contradictory_emoji`

- pairs sentiment-bearing text with ambiguous or conflicting emojis
- forces the model to rely on more than emoji identity

`sarcasm_negative`

- preserves sarcastic negative intent under the 3-class setup
- keeps sarcasm as metadata instead of introducing a fourth label

## Final Signal Counts in the 24,000-Row Dataset

- `real_observation`: `1,400`
- `shared_emoji_overlap`: `5,147`
- `orthographic_noise`: `4,271`
- `contradictory_emoji`: `4,712`
- `lexical_swap`: `2,264`
- `short_phrase_overlap`: `2,548`
- `mixed_resolution`: `1,533`
- `sarcasm_negative`: `1,159`
- `hedged_neutral`: `624`
- `balanced_neutral`: `342`

The high count of overlap, contradiction, and hard-case signals was intentional. The earlier dataset failed mainly because it was too easy and too class-coded.

## Manual Stress Benchmark

The final manual benchmark is `datasets/real_holdout_stress_600.csv`.

This file is a paired stress version of the untouched `600`-row real holdout.

### Construction Rules

- each stress row maps to one row in `datasets/real_holdout.csv`
- the sentiment label stays unchanged
- sarcasm mapping remains unchanged
- the benchmark remains test-only
- the edits are manual, not generated by the scripted builder

### Stress Types

- `ambiguous_swap`: `329`
- `contradictory_swap`: `164`
- `emoji_removed`: `107`

### Final Label Counts

- `Negative`: `319`
- `Neutral`: `121`
- `Positive`: `160`

## Why the Manual Stress Set Was Needed

The untouched real holdout is itself highly emoji-coded. That means an emoji-only model can appear unrealistically strong even when there is no train-test leakage.

The stress benchmark was therefore created to answer a different question:

`What happens when the same real source rows are edited so that emoji is removed, made ambiguous, or made contradictory while the underlying sentiment stays the same?`

That benchmark is central to the current research direction because it tests robustness rather than only in-source fit.

## Final Dataset Logic

The overall dataset design is now:

- `social_media_sentiment_test.csv` as the only real source
- `real_generation_seed.csv` as the real partition used for controlled generation
- `social_media_sentiment_dataset_v4.csv` as the main hybrid training corpus
- `real_holdout.csv` as the untouched real benchmark
- `real_holdout_stress_600.csv` as the paired robustness benchmark

This structure is the main reason the current study is much stronger than the original notebook-based setup.
