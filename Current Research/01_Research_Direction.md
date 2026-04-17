# Current Research Direction

## Working Title

The study title remains unchanged:

`Emoji-Only Sentiment Analysis Using a Deep Learning Model: A Comparative Study with Text-Enhanced Approaches`

## Initial State of the Research

The project originally started as a notebook-centered experiment in `EmojiPredict_Model.ipynb` using a large synthetic dataset. The early setup had several characteristics:

- training and evaluation were tied to the notebook workflow
- the dataset scale was large, but the language patterns were highly repetitive
- internal scores became unrealistically high very quickly
- loss curves looked too clean for a realistic social-media sentiment task
- the study risked measuring template memorization and emoji shortcuts rather than robust sentiment understanding

In that initial state, the research direction was too close to: "train three pipelines on a large dataset and compare scores." That was not strong enough for publication because the data validity problem would dominate the results.

## Problems Identified

After auditing the notebook and the original synthetic data, the main issues became clear:

- the synthetic dataset was too templated and too easy
- emoji usage was strongly class-coded, which allowed shortcut learning
- near-perfect internal results were not reliable evidence of generalization
- the only real dataset available, `social_media_sentiment_test.csv`, was small and also heavily emoji-coded
- the original workflow did not clearly separate training data, untouched evaluation data, and stress testing

These findings forced a shift in the research direction.

## New Direction

The research was redirected from a simple performance comparison into a controlled robustness study.

The new direction has five main components:

1. Move training out of the notebook and into reproducible Python scripts with local GPU acceleration.
2. Keep the paper title unchanged, but tighten the actual claim of the study.
3. Build a hybrid training corpus anchored in the real seed data instead of relying on a purely synthetic dataset.
4. Separate untouched real evaluation data from the data used for generation.
5. Introduce a manually curated stress benchmark to test what happens when emoji shortcuts are weakened.

## What the Study Now Tries to Show

The current study is no longer trying to prove that emoji-only sentiment analysis is broadly superior.

Instead, it is trying to answer a more defensible question:

`How well do emoji-only, text-only, and text+emoji models perform when the training data is controlled, the real source is separated properly, and emoji shortcuts are explicitly stress-tested?`

This is a stronger and more publishable research direction because it allows the paper to make claims about robustness rather than only reporting high scores.

## Current Scope

The current research scope is:

- a 3-class sentiment setting: `Negative`, `Neutral`, `Positive`
- sarcasm is preserved as metadata, not as a fourth class
- a scripted `24,000`-row hybrid training dataset
- a separated real holdout derived from the original real source
- a manually created `600`-row paired stress benchmark
- comparison of `text`, `emoji`, and `combined` pipelines
- a final combined model that uses robustness-oriented fusion rather than naive fusion

## What We Now Claim

The current project can defensibly claim:

- emoji-only performance can look very strong on emoji-coded datasets
- that strength can collapse when emoji shortcuts are weakened
- text-only models can remain more robust than multimodal fusion under emoji disruption
- multimodal fusion can be improved with robustness training, but it does not automatically become the most robust system

## What We Do Not Claim

The current project should not claim:

- that the training corpus is a natural social-media benchmark
- that one real source proves broad real-world generalization
- that emoji-only sentiment analysis is generally superior across domains
- that the final fusion model is universally better than text-only sentiment analysis

## Current Research Position

The best description of the project at this stage is:

`a controlled, publishable robustness study on emoji-only, text-only, and text+emoji sentiment analysis using a hybrid training corpus, an untouched in-source real holdout, and a manually curated stress benchmark`
