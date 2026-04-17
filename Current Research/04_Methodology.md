# Methodology

## 1. Research Design

This study was designed as a controlled comparative experiment in three-class sentiment analysis. The goal was to compare how well three input settings perform under different levels of emoji reliability:

- `text-only`
- `emoji-only`
- `text+emoji`

The methodology was deliberately redesigned from the initial notebook-based setup because the original large synthetic dataset produced unrealistically strong internal results. The revised methodology therefore emphasizes:

- controlled dataset construction rather than raw scale
- strict separation between training-generation data and real test data
- robustness testing under weakened emoji shortcuts
- comparison of unimodal and multimodal models under the same training protocol

The final system is implemented as a local Python training pipeline rather than a notebook-only workflow.

## 2. Data Sources and Dataset Strategy

### 2.1 Real Source Dataset

The only real source dataset available for this study was `social_media_sentiment_test.csv`, which contains `2,000` short social-media style posts. This file served as the anchor for the entire data construction process.

The original real dataset contains the labels:

- `Negative`
- `Neutral`
- `Positive`
- `Sarcastic`

To preserve a strict three-class sentiment setup, the label policy was defined as follows:

- `Negative -> Negative`
- `Neutral -> Neutral`
- `Positive -> Positive`
- `Sarcastic -> Negative` with `sarcasm_flag = 1`

This mapping was adopted because the sarcastic samples in the source file generally express negative pragmatic intent even when the surface wording is positive.

### 2.2 Separation of Real Data

Before any augmentation or dataset generation, the `2,000` real rows were partitioned into two disjoint subsets:

- `1,400` rows for controlled generation and training support
- `600` rows reserved as untouched real evaluation data

These were stored as:

- `datasets/real_generation_seed.csv`
- `datasets/real_holdout.csv`

The separation rule was strict:

- no row from the untouched holdout was used to generate training data
- no derived child of a holdout row was allowed in the hybrid training dataset
- overlap between the final hybrid dataset and the untouched holdout was verified to be `0`

This separation was necessary because earlier experiments showed that using one real source without isolation would inflate evaluation results.

### 2.3 Hybrid Training Corpus

The main training dataset was `datasets/social_media_sentiment_dataset_v4.csv`, a `24,000`-row hybrid corpus built from the `1,400` real generation-seed rows.

The dataset was class-balanced:

- `8,000` `Negative`
- `8,000` `Neutral`
- `8,000` `Positive`

It contains:

- `1,400` direct real-seed observations
- `22,600` controlled augmentations

Each row includes traceability metadata such as:

- `parent_id`
- `original_seed_label`
- `sarcasm_flag`
- `generation_signal`
- `phrase_family`
- `split_group`
- `recommended_split`

The training corpus was created to reduce the weaknesses observed in the earlier synthetic data, especially repetitive phrasing and class-coded emoji shortcuts.

### 2.4 Controlled Generation Signals

The hybrid dataset was generated with rule-based signals derived from observed properties of the real source rather than from arbitrary templates.

The final generation signals were:

- `real_observation`
- `lexical_swap`
- `orthographic_noise`
- `mixed_resolution`
- `hedged_neutral`
- `balanced_neutral`
- `shared_emoji_overlap`
- `short_phrase_overlap`
- `contradictory_emoji`
- `sarcasm_negative`

The reasoning behind these signals was as follows.

`real_observation` preserved real examples from the source distribution and anchored the dataset in authentic slang, spelling noise, and emoji usage.

`lexical_swap` reduced dependence on one exact phrase by replacing sentiment-bearing wording with alternatives from the same polarity family.

`orthographic_noise` introduced repeated letters and small misspellings because such noise is common in short social-media posts.

`mixed_resolution` created examples whose sentiment becomes clear later in the sentence, making the task less template-driven.

`hedged_neutral` and `balanced_neutral` were used to make neutral sentiment more realistic by including low-affect and mixed-attitude expressions instead of treating neutral as a weak form of positivity.

`shared_emoji_overlap` explicitly reused emojis across different sentiment labels so that emoji identity alone would not trivially reveal the class.

`short_phrase_overlap` reused short phrase shells across labels so that one phrase family would not collapse into a one-label shortcut.

`contradictory_emoji` paired sentiment-bearing text with ambiguous or conflicting emojis so that models would need to rely on more than the emoji channel.

`sarcasm_negative` preserved sarcastic negative intent inside the three-class setup while retaining sarcasm as metadata.

### 2.5 Final Manual Stress Benchmark

In addition to the untouched real holdout, a second test-only benchmark was created manually: `datasets/real_holdout_stress_600.csv`.

This file is a paired stress version of the untouched `600`-row real holdout. Each row in the stress file corresponds to one row in `datasets/real_holdout.csv`, and the sentiment label is preserved.

The stress benchmark was necessary because the untouched real holdout remained strongly emoji-coded. As a result, an emoji-only model could score unrealistically well on the untouched real set even without train-test leakage.

The manual stress set weakens emoji shortcuts using three edit types:

- `ambiguous_swap`
- `contradictory_swap`
- `emoji_removed`

The final composition of the stress benchmark is:

- total rows: `600`
- `Negative`: `319`
- `Neutral`: `121`
- `Positive`: `160`

Stress-type counts:

- `ambiguous_swap`: `329`
- `contradictory_swap`: `164`
- `emoji_removed`: `107`

This benchmark was treated as the primary robustness test in the final experimental setup.

## 3. Data Preprocessing

### 3.1 Text Processing

For each sample, the raw text was preserved and a text-only version was extracted by removing all emoji characters and emoji joiners. The preprocessing procedure also normalized repeated whitespace. If a sample contained no non-emoji text after removal, the placeholder token `[empty]` was used.

Text tokenization was performed with a lowercase regular-expression tokenizer matching alphanumeric tokens and apostrophes. The model therefore uses lightweight lexical tokenization rather than a pretrained transformer tokenizer.

### 3.2 Emoji Processing

Emoji tokens were extracted directly from the raw text using a Unicode emoji pattern. Emoji joiners and presentation selectors were removed before extraction. If no emoji was present, the sequence was represented using the special token `[NO_EMOJI]`.

### 3.3 Phrase-Key Construction for Grouped Splitting

To reduce leakage from repetitive short social-media openings, a `phrase_key` was created from the text-only representation by tokenizing the text and dropping the first two tokens. This grouped key was used as the main split unit for the synthetic-hybrid corpus.

### 3.4 Vocabulary Construction

Separate vocabularies were built for text tokens and emoji tokens using only the training split. The minimum frequency threshold was `2` for both vocabularies.

Special symbols were used for both vocabularies:

- `[PAD]`
- `[UNK]`

The emoji vocabulary additionally ensured support for `[NO_EMOJI]`.

### 3.5 Sequence Lengths

Input sequences were truncated and padded to fixed maximum lengths:

- text maximum length: `24`
- emoji maximum length: `12`

## 4. Train-Validation-Test Splitting

### 4.1 Hybrid Corpus Split

The `24,000`-row hybrid training corpus was split with grouped partitioning rather than simple random partitioning. Grouped splitting was chosen because random splitting on synthetic-style data can leak reusable phrase families across train, validation, and test.

The final split configuration was:

- training proportion: `0.60`
- validation proportion: `0.20`
- test proportion: `0.20`

The grouping variable was `phrase_key`.

The realized split sizes were:

- train: `15,609`
- validation: `3,642`
- test: `4,749`

The number of unique grouped templates was:

- train: `3,009`
- validation: `1,003`
- test: `1,004`

Observed overlap between grouped partitions was:

- train/validation overlap: `0`
- train/test overlap: `0`
- validation/test overlap: `0`

### 4.2 External Test Sets

Two additional test-only datasets were used in the current research configuration:

- `datasets/real_holdout.csv`
- `datasets/real_holdout_stress_600.csv`

The first was used as an untouched in-source real benchmark. The second was used as a robustness benchmark with identical source coverage but weakened emoji shortcuts.

## 5. Model Architectures

Three sentiment pipelines were trained.

### 5.1 Text-Only Model

The text-only model uses:

- a token embedding layer with dimension `128`
- a bidirectional LSTM encoder with hidden size `128`
- a classifier MLP with structure `256 -> 128 -> 3`
- dropout `0.35`

The final text representation is formed by concatenating the forward and backward hidden states of the biLSTM.

### 5.2 Emoji-Only Model

The emoji-only model uses:

- an emoji embedding layer with dimension `64`
- a bidirectional LSTM encoder with hidden size `96`
- a linear classifier from `192 -> 3`
- dropout `0.30`

This model receives only the emoji sequence extracted from each sample.

### 5.3 Combined Text+Emoji Model

The final multimodal model uses two separate encoders:

- a text biLSTM branch
- an emoji biLSTM branch

The text branch produces a `256`-dimensional representation, which is projected to `192` dimensions. The emoji branch produces a `192`-dimensional representation, which is projected to `64` dimensions.

The final combined model incorporates four robustness-oriented fusion mechanisms:

1. `emoji feature projection`
   This reduces the relative dimensional contribution of the emoji branch.

2. `learned confidence gating`
   A sigmoid gating network receives the concatenated text and emoji features and outputs a scalar confidence value that scales the emoji representation.

3. `emoji branch dropout`
   During training, the projected emoji feature vector is randomly dropped at the branch level with probability `0.35`.

4. `emoji scaling`
   After gating and branch dropout, the emoji representation is multiplied by a fixed scale factor of `0.45` before fusion.

The final fusion classifier uses:

- fused input size `256`
- hidden size `192`
- output size `3`

## 6. Robustness-Oriented Training for the Combined Model

The final combined model was not trained with standard cross-entropy alone.

To reduce over-reliance on the emoji channel, the combined model used additional corruption and invariance objectives during training.

### 6.1 Emoji Corruption

For each combined-model batch, a corrupted emoji view was created using one of two interventions:

- emoji removal with probability `0.20`
- emoji token swap with probability `0.35`

If an emoji sequence was removed, it was replaced with `[NO_EMOJI]`. If it was swapped, valid emoji tokens were sampled from the emoji vocabulary and used to replace the active emoji positions.

### 6.2 Invariance Training

The combined model computed predictions for:

- the clean batch
- the corrupted-emoji batch

Training then used three terms:

- clean cross-entropy loss
- corrupted-view cross-entropy loss
- a symmetric KL divergence consistency loss between clean and corrupted logits

The total loss for the combined model was:

`L = L_clean + 0.40 * L_corrupted + 0.20 * L_invariance`

where the invariance term is the symmetric KL divergence between predictions from the clean and corrupted emoji views.

The text-only and emoji-only models were trained with standard cross-entropy only.

## 7. Training Configuration

All models were trained with the same general optimization settings:

- epochs: `30`
- batch size: `256`
- optimizer: `AdamW`
- learning rate: `0.002`
- weight decay: `0.01`
- label smoothing: `0.10`

Training-time token corruption was also applied at the input level:

- text token dropout: `0.15`
- emoji token dropout: `0.10`

The best checkpoint for each pipeline was selected using validation weighted F1.

## 8. Evaluation Procedure

Each model was evaluated on:

- the grouped internal test split from the hybrid corpus
- the untouched real holdout
- the paired manual stress benchmark

The core evaluation metrics were:

- accuracy
- weighted F1 score
- confusion matrices

Weighted F1 was used as the main model-selection criterion because the study compares behavior across different evaluation settings and class supports.

The untouched real holdout and the stress benchmark were interpreted differently:

- the untouched real holdout measures in-source performance
- the stress benchmark measures robustness when emoji shortcuts are weakened

## 9. Implementation Environment

The final experiments were executed using the local script pipeline with GPU acceleration in the `dl312` Python environment.

The final run used:

- device: `cuda`
- GPU: `NVIDIA GeForce RTX 5070`
- random seed: `42`

This scripted implementation replaced the earlier notebook-centered workflow and improved reproducibility, checkpointing, and figure generation.

## 10. Methodological Rationale

The final methodology was designed around one central concern: a model can appear strong on emoji-rich sentiment data because the dataset itself contains recoverable emoji shortcuts.

For that reason, the study did not rely on one test set alone. Instead, it combined:

- a controlled hybrid training corpus
- an untouched real holdout separated before generation
- a manually curated paired stress benchmark

This methodology makes it possible to distinguish between:

- high performance due to dataset-specific shortcuts
- high performance that remains stable when emoji information becomes unreliable

That distinction is the core methodological contribution of the current study.
