# Neural Research Paper draft

Emoji-Only Sentiment Analysis Using a Deep
## Learning Model: A Comparative Study with Text-

## Enhanced Approaches

Tharushika Hirushani
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Hasindu Thirasara
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Pasan Karunathilaka
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Hasindu Thirasara
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Dhanitha Kolonnage
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Hasindu Thirasara
Department of Industrial Management,
Faculty of Science,
University of Kelaniya,
## Kelaniya, Sri Lanka

Abstract—Biometric authentication is widely used in mobile
devices to improve security beyond traditional methods such as
passwords and PINs. This paper presents a systematic review of
fingerprint, face recognition, and behavioral biometrics,
focusing on their security, performance, and privacy aspects.
Fingerprint and face recognition provide high accuracy and
ease of use but are vulnerable to spoofing attacks. Behavioral
biometrics enable continuous authentication by monitoring
user behavior, though they may have higher error rates and
resource usage. This study also highlights important privacy
concerns, as biometric data cannot be changed once
compromised. Different privacy-preserving techniques, such as
encryption and secure on-device processing, are discussed. The
review identifies key research gaps, including limited real-world
testing and performance challenges. It concludes that
combining multiple biometric methods can improve both
security and usability in mobile devices.

Conventional
sentiment
analysis
approaches
have
predominantly relied on text-based representations,
employing machine learning and deep learning models
such as Long Short-Term Memory (LSTM) networks and
transformer-based architectures like BERT. These models
have demonstrated strong capabilities in capturing
contextual semantics and sequential dependencies [1], [2].
However, they often fail to fully represent the nuances of
modern communication, where emojis significantly
influence sentiment interpretation. Emojis can reinforce,
alter, or even invert the polarity of textual content,
particularly in informal or sarcastic contexts, leading to
potential
misclassification
when
ignored
or
underrepresented.

To address these limitations, recent studies have explored
fusion-based approaches that integrate textual and emoji
information. While these methods improve classification
performance by leveraging additional emotional cues, they
inherently treat emojis as auxiliary features rather than
primary sentiment carriers. Furthermore, existing research
on emoji-only sentiment analysis remains limited and often
constrained to domain-specific datasets or traditional
modeling techniques. Challenges such as contextdependent emoji semantics, sarcasm interpretation, and the
lack of comprehensive evaluation frameworks continue to
hinder progress in this area [3], [4].

Keywords—Biometric Authentication, Fingerprint, Face
Recognition,
Behavioral
Biometrics,
Mobile
Security,
## Privacy, Data Protection

I.
INTRODUCTION

Sentiment analysis has emerged as a critical task in Natural
Language Processing (NLP), enabling the automatic
extraction of opinions, emotions, and attitudes from largescale textual data. With the exponential growth of social
media platforms, user-generated content has evolved into a
highly informal and multimodal form of communication.
In this context, emojis have become an integral component
of digital expression, serving as compact yet powerful
carriers of affective meaning that complement or even
replace textual cues.

In response to these challenges, this study proposes a
unified sentiment analysis framework that systematically
evaluates three distinct modalities: text-only, emoji-only,
and combined text–emoji inputs. The proposed approach
employs hybrid BERT-LSTM architecture to effectively

capture both contextual representations and sequential
dependencies. In addition, a dataset is used, specifically
designed to include emoji-dense content and sarcasm,
enabling a more realistic and challenging evaluation of
sentiment models in contemporary
communication
settings.

The main contributions of this paper are as follows. First, a
unified experimental framework is developed to enable a
fair comparison between text-only, emoji-only, and fusionbased sentiment analysis approaches. Second, the study
investigates the viability of emoji-only sentiment analysis
as an independent modality, addressing a significant gap in
existing literature. Third, a sarcasm-focused and emoji-rich
dataset is constructed to evaluate model robustness under
complex linguistic conditions. Finally, a hybrid BERT-
LSTM architecture is utilized to enhance the modeling of
contextual and semantic relationships within multimodal
data.

The remainder of this paper is organized as follows.
Section II presents the related work. Section III describes
the proposed methodology. Section IV details the
experimental setup and results. Section V discusses the
findings, and Section VI concludes the paper with
directions for future research.

II.
# LITERATURE REVIEW

## A. Text-Based Sentiment Analysis

Text-based sentiment analysis is the traditional approach
for analyzing opinions in online content. Early methods
relied on machine learning algorithms such as Naive
Bayes, Support Vector Machines, and Logistic Regression
with
bag-of-words
and
TF-IDF
features.
With
advancements in deep learning, LSTM and transformerbased models such as BERT have become dominant due to
their ability to capture contextual and sequential
dependencies.

BERT
demonstrates
superior
performance
in
understanding contextual semantics and complex linguistic
patterns [1], while Bi-LSTM effectively models sequential
dependencies across domains such as social media and
software engineering [2].

However, text-based methods fail to fully capture modern
communication, where emojis play a crucial role. Ignoring
emojis leads to loss of emotional information and reduced
performance [1]. Studies further show that incorporating
emoji features improves classification accuracy, indicating
that text-only models overlook important sentiment cues
[5].

Thus, text-based approaches remain limited in emoji-rich
environments, motivating the need for emoji-enhanced and
emoji-only methods.

## B. Fusion-Based Sentiment Analysis (Text + Emoji)

Fusion-based approaches integrate text and emojis to
overcome limitations of text-only models, recognizing
emojis as key carriers of emotional context. Studies show
that combining both modalities improves sentiment
classification by capturing richer user intent [6].

Hybrid ML/DL models demonstrate that emoticons
enhance performance and can dominate sentiment
interpretation [7], while multi-view frameworks treat text
and emojis as complementary inputs [8]. Advanced
architectures such as EMFSA (cross-attention) and
multimodal systems combining BERT, Bi-LSTM, and
ResNet further improve accuracy, with performance
dropping when emoji features are removed [9], [10].

Additional studies confirm that integrating emojis
improves performance across ML and DL models, with
transformer-based approaches achieving the best results
[11]. Fusion methods are also effective in handling sarcasm
and nuanced expressions [12].

However, these approaches rely primarily on text and treat
emojis as supplementary features, without exploring their
standalone capability.

## C. Emoji-Based Sentiment Analysis

Recent research shows that emojis are semantically rich
and capable of independently conveying sentiment. Largescale studies demonstrate that emoji-only models can
achieve performance comparable to text-based models,
with equal accuracy in Logistic Regression (0.75) and
slightly lower performance in BiLSTM (0.74 vs. 0.80),
while requiring up to 40× less data and faster processing
[4].

Deep learning studies confirm strong correlations between
emojis and emotional states, with models like BERT
effectively learning text–emoji relationships. However,
challenges such as class imbalance, sarcasm, and cultural
variation remain [3].

Other works report emoji-only accuracy around 70–85%,
with limitations in handling semantic relationships and
unseen emojis [13]. Nevertheless, emojis can significantly
influence sentiment, even inverting polarity in up to 40%
of cases [14].

Incorporating emojis improves performance across models
(e.g., BERT accuracy from 95.3% to 96.2%) [11], while
traditional ML approaches show improvements up to

26.7% [15], [16]. Rule-based systems like VADER
struggle with sarcasm, and frameworks such as SEntiMoji
show that emojis can act as weak supervision signals for
deep learning [2].

Overall, emojis are powerful sentiment indicators capable
of functioning independently and enhancing sentiment
understanding.

## D. Research Gap

Despite progress, several key gaps remain.

First, most studies focus on fusion approaches, treating
emojis as supplementary, with limited exploration of
emoji-only analysis and reliance on domain-specific
datasets [4], [11], [17].

Second, emoji-only models are mainly based on traditional
ML or simple DL, and even advanced models struggle with
semantic relationships and unseen emojis, highlighting the
need for architectures such as BERT-LSTM [1], [13].

Third, sarcasm and irony are insufficiently addressed, as
few studies use datasets specifically designed for emojionly sarcasm detection [3], [18].

Fourth,
many
approaches
rely
on
static
emoji
representations, limiting context-aware and culturally
adaptive understanding [15], [16].

Finally, there is no unified framework comparing text-only,
emoji-only, and fusion approaches under the same
conditions. No prior work integrates all three pipelines
within a single BERT-LSTM framework evaluated on
sarcasm-heavy, emoji-dense datasets.

III.
METHODOLOGY

This study employed a unified experimental framework to
evaluate deep learning models for sentiment classification
across three input modalities: text-only, emoji-only, and a
hybrid approach. Three pipelines were developed for
comparative analysis.

●
Pipeline A (Text-Only): A pre-trained BERT
model was used as a fixed feature extractor with
frozen layers. A custom Multi-Layer Perceptron
(MLP) classifier was trained on the CLS token
representation for sentiment prediction.
●
Pipeline B (Emoji-Only): A custom model
processed emoji sequences using an Embedding
layer followed by a Bidirectional Long Short-
Term Memory (Bi-LSTM) network.
●
Pipeline
C
(Combined):
A
dual-branch
architecture integrated BERT-based text features
and
Bi-LSTM-based
emoji
features.
The
concatenated representations were passed through
a fusion layer to generate final prediction.

## Data Preparation

Three datasets were used:
Primary dataset (100,000 samples) for training/testing
Two external validation sets:
External Dataset 1 (2,000 samples)
External Dataset 2 (40,000 samples).

Preprocessing included regex-based text cleaning and
emoji extraction using the emoji Python library. Emojis
were converted into textual aliases (e.g., :smiling_face:).
Sentiment labels were mapped to three classes: Negative
(0), Neutral (1), and Positive (2).
title and abstract screening, to full-text eligibility
assessment before final inclusion. For synthesis, the
selected studies were categorized into a taxonomy based on
biometric modality — specifically Fingerprint, Face, and
Behavioral biometrics — and subsequently analyzed
against privacy-preserving frameworks, enabling a critical
evaluation of how each modality interacts with data
protection methods such as secure template storage and
cryptographic frameworks in mobile environments.
## C. Inclusion and Exclusion Criteria

To ensure quality, relevance, and technical depth, strict
selection criteria were applied. Included were peerreviewed, English-language articles published between
2016 and 2026 that focused specifically on mobile or
smartphone hardware architectures. Excluded were short
communications, white papers, and preprints, as well as
studies addressing desktop-only or stationary biometric
systems and any papers lacking quantitative performance
metrics such as Equal Error Rate (EER) or False
Acceptance Rate (FAR).

## D. PRISMA Selection & Synthesis Strategy

Literature filtering followed the PRISMA framework
through a multi-stage screening process, progressing from
initial database retrieval and duplicate removal, through
title and abstract screening, to full-text eligibility
assessment before final inclusion. For synthesis, the

![page3_img1.png](Neural%20Research%20Paper%20draft_images/page3_img1.png)
