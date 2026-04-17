# EMFSA: Emoji-based multifeature fusion sentiment analysis

# PLOS ONE

# RESEARCH ARTICLE
EMFSA: Emoji-based multifeature fusion
sentiment analysis

Hongmei Tang1,2, Wenzhong Tang1*, Dixiongxiao Zhu1, Shuai WangID1*,
## Yanyang Wang3,4, Lihong Wang5

## 1 School of Computer Science and Engineering, Beihang University, Beijing, China, 2 Xinjiang Astronomical
Observatory, Chinese Academy of Sciences, Urumqi, China, 3 School of Aeronautic Science and
Engineering, Beihang University, Beijing, China, 4 Jiangxi Research Institute of Beihang University, Nanchan,
China, 5 National Computer Network Emergency Response Technical Team/Coordination Center of China,
## Beijing, China

a1111111111
a1111111111
a1111111111
a1111111111
a1111111111

* tangwenzhong@buaa.edu.cn (WT); wangshuai@buaa.edu.cn (SW)

Abstract

Short texts on social platforms often suffer from insufficient emotional semantic expressions, sparse features, and polysemy. To enhance the accuracy achieved by sentiment
analysis for short texts, this paper proposes an emoji-based multifeature fusion sentiment
analysis model (EMFSA). The model mines the sentiments of emojis, topics, and text features. Initially, a pretraining method for feature extraction is employed to enhance the
semantic expressions of emotions in text by extracting contextual semantic information from
emojis. Following this, a sentiment- and emoji-masked language model is designed to prioritize the masking of emojis and words with implicit sentiments, focusing on learning the emotional semantics contained in text. Additionally, we proposed a multifeature fusion method
based on a cross-attention mechanism by determining the importance of each word in a text
from a topic perspective. Next, this method is integrated with the original semantic information of emojis and the enhanced text features, attaining improved sentiment representation
accuracy for short texts. Comparative experiments conducted with the state-of-the-art baseline methods on three public datasets demonstrate that the proposed model achieves accuracy improvements of 2.3%, 10.9%, and 2.7%, respectively, validating its effectiveness.

# OPEN ACCESS

Citation: Tang H, Tang W, Zhu D, Wang S, Wang
Y, Wang L (2024) EMFSA: Emoji-based
multifeature fusion sentiment analysis. PLoS ONE
19(9): e0310715. https://doi.org/10.1371/journal.
pone.0310715

Editor: Daniela Moctezuma, Centro de
Investigacion en Ciencias de Informacion
Geoespacial AC (Research Center on Geospatial
## Information Sciences), MEXICO

## Received: February 27, 2024

## Accepted: September 4, 2024

## Published: September 19, 2024

Copyright: © 2024 Tang et al. This is an open
access article distributed under the terms of the
Creative Commons Attribution License, which
permits unrestricted use, distribution, and
reproduction in any medium, provided the original
author and source are credited.

Introduction

An emotion refers to an attitude, thought, or judgment caused by sensations [1]. Emotions are
generally considered to have three polarities: positive, negative, and neutral. Sentiment analysis
involves computationally processing the ideas, emotions, and subjectivity in a text [2]. Currently, social media platforms host vast, diverse, and mixed-modal data. An analysis of platforms such as Twitter reveals an abundance of short text messages, hashtags, and emojis.
Nearly half of the content on Instagram includes emojis [3], and Facebook sees the use of 5 billion emojis every day. Due to the presence of unconstrained language, sparse features, evident
fragmentation, limited vocabularies, high noise levels, colloquial expressions, and implicit
opinions and attitudes in syntactic structures and contextual clues within short text messages

Data Availability Statement: The datasets
generated and/or analyzed during the 513 current
study are available in https://doi.org/10.6084/m9.
figshare.25289353.v1.

Funding: The research work described in this
paper was supported by a grant from the National
Natural Science Foundation of China, under Grant
No.62272022. National Key Research and
Development Program of China, under Grant
## No.210YBXM2024106007, National Key Research

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
1 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

on social platforms, emotional expressions become ambiguous. Although text-based opinion
mining methods have proven highly useful in sentiment analysis tasks, they still face issues
such as domain, topic, and time dependencies [4].
In 1997, emojis appeared in Japan, providing visual representations of emotions through
nonverbal cues. They serve as intentional carriers of emotional states and are widely used in
online communications. And in the context of media communications, Daft et al. [5] suggested that higher information processing efficiency is achieved when multiple cues are available. Approximately 20 billion tweets are posted daily on platforms such as Twitter, and with
each new Unicode version, new emojis are introduced, making them increasingly relevant to
sentiment analysis tasks [6]. Emojis can provide contextual information, enhancing the ability
to process oral information content. To address issues such as insufficient semantic expressions, sparse semantic features, and polysemy in short text sentiments, this paper proposes an
emoji-based multifeature fusion sentiment analysis (EMFSA) model. The model primarily
explores the combination of various features, including emojis, topics, and text, for mining
sentiments across different modalities. It utilizes a cross-attention mechanism to integrate
these features, obtaining complementary and enhanced information between modalities. This
approach facilitates matching similar contexts and better recognizing and understanding the
meanings of contextual representations and attention, ultimately improving the accuracy of
sentiment analysis. Our contributions are as follows:

and Development Program of China, under Grant
No.2022YFB3207700. The funders had a role in
data collection and analysis of the manuscript.

Competing interests: The authors have declared
that no competing interests exist.

(1) An emoji-enhanced text feature extraction and pretraining method is proposed. The
exBERT model is introduced as the base framework, an encoder (emo_exBERT) is designed
for synchronously extracting text and emoji features, and the model is pretrained on a large
corpus containing emojis. This approach aims to extract emoji-based contextual semantic
information, enhancing the overall semantic representation of the given text.

(2) A sentiment- and emoji-masked language model (Senti_MLM) that is suitable for English
text is introduced. This model prioritizes masking emojis and words with implicit sentiments, yielding a higher masking probability. The objective is to focus on learning semantic
representations, thereby enhancing the effectiveness of the subsequent sentiment analysis
process.

(3) A multifeature fusion method based on a cross-attention mechanism is proposed. A biterm
topic model is introduced to further discover the latent semantic relationships (i.e., topics)
implied between documents and words. This process helps determine the importance of
each word in the input text from a topic perspective. Subsequently, these importance scores
are fused with the original semantic information of emojis and the text features enhanced
by emojis, aiming to improve the accuracy of the semantic representations of short-text
sentiments and enhance these sentiments with the topics and emojis in the text.

(4) The experimental results show that the model exhibits outstanding performance in sentiment classification tasks conducted on public datasets and outperforms the existing
methods.

The remaining sections of this paper are organized as follows: Section 2 examines and discusses the related literature. Section 3 provides a detailed introduction to the EMFSA model
proposed in this paper. Section 4 discusses the experimental setups, comparison methods, and
implementation details and presents quantitative and qualitative evaluations of the experimental results. Finally, in Section 5, we summarize the findings of this study and outline our future
work.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
2 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Related work
## Sentiment analysis

Sentiment analysis is performed mainly using lexicon-based methods [7–10] and machine
learning-based methods [11–13] to analyze document-level sentiments. However, the obtained
information is often incomplete, leading to suboptimal accuracy. With the development of
deep learning techniques, Wang et al. [14] proposed an attention-based LSTM combination to
achieve improved model performance. However, LSTM is time-consuming to train and can
encode only unidirectional sequence information. Subsequently, researchers were inspired by
the excellent performance of graph neural networks (GNNs) in terms of managing the complex relational structures in text and preserving the global information contained in feature
embeddings, which led to better classification results [15]. However, neural network-based
approaches still suffer from insufficient model training processes and poor generalization performance. Thereafter, researchers attempted to solve the sentiment analysis problem using two
approaches based on BERT models. First, this approach mainly employs the pretrained model
+ fine-tuning strategy to directly fine-tune the pretrained BERT model according to the sentiment analysis task, and it has produced impressive results in various natural language processing tasks [16–20]. Second, the masked language modeling (MLM) approach is used to further
pretrain on domain-specific datasets and then fine-tune on them for sentiment analysis tasks
[21, 22]. This approach involves retraining BERT again and then fine-tuning it for fine-grained
sentiment analysis and other sentiment-related tasks. However, no special attention is given to
“sentiments” throughout the pretraining process of the above “retraining” approach. Subsequently, multiscale graph attention networks (MSGATs) [23, 24] based on dependency grammars and hybrid models based on topic knowledge [25] were proposed. However, due to the
sparse semantic representations and inadequate semantic expressions of short texts, these
models face challenges when addressing emerging terms on social media networks and realizing high-precision sentiment analysis.

## Sentiment analysis with emojis

Emoji-based sentiment analysis methods are divided into three main categories: dictionarybased methods, machine learning-based methods, and deep learning-based methods. Dictionary-based methods focus on building emoji sentiment dictionaries to support text sentiment
analysis tasks. Kralj et al. created the first emoji sentiment dictionary [26]. Subsequently, the
Emoji2Vec [27] pretrained embedding method was developed, and EmojiNet (a semantic
repository of emojis) was published [28]. Researchers created raw sentiment lexicons [29] and
emoji lexicons using Unicode Consortium classification [30], unsupervised classification [6],
manual compilation [31], and synonym categorization [32, 33] to automate the creation of
combined lexicons [34]. This method lays the foundation for research on emoji sentiment
analysis. Although this method is simple and effective, its rules may not fully cover all possible
situations. Machine learning methods involve training sentiment classifiers based on corpora
to analyze the sentiments of text [35]. However, this method is sensitive to the input data
representation, requires the computation of prior probabilities, and achieves low accuracy in
sentiment analysis tasks.
Deep learning methods excel at effectively utilizing contextual word embeddings to generate dense document representations, thereby significantly improving model accuracy. Examples include attention-based network models [36–38], BiLSTM [39], deep neural networks,
models that combine emoji- and lexicon-based sentiment enhancement with fuzzy inference
[40], document representation models that incorporate word-sentiment associations and topic

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
3 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

models [41], symbiotic graph networks [42] (which learn emoji representations by embedding
them into emoji nodes based on the semantic information derived from an external knowledge
base, such as EmojiNet to learn emoji representations with an up-to-date emoji-text design
baseline), and recurrent neural network models [43, 44]. However, these approaches explore
sentiments only from the textual and emoji perspectives and do not consider the influence of
topics on sentiment analysis. To further explore the potential semantic relationships between
documents and words, topic modeling-based approaches have been used in sentiment analysis
studies [45, 46]. Haque et al. [47] used latent Dirichlet allocation (LDA)-based topic modeling
to identify trending topics in ChatGPT-related tweets and performed a manual labeled sentiment analysis. Taecharungroj [48] explored the capabilities and weaknesses of ChatGPT using
LDA-based topic modeling. These approaches validate the effectiveness of topic embedding in
sentiment analysis tasks but lack the ability to analyze short textual scenarios with sparse
semantics. This study focuses on the short text included in mixed-mode data containing emoticons on social media platforms. To address the problem of sparse word co-occurrence patterns in individual documents, this paper adopts the BTM short text topic model [49] based on
word co-occurrence patterns to capture the entire text passage and the topic vectors of each
token in it; the shared topic distribution is then trained across the corpus.
With the continuous development of pretrained large-scale language models, researchers
have applied these models to sentiment analysis tasks. In 2017, the transformer structures [50]
of deep learning models surpassed 100 million parameters. The parameters of the BERT network model [51] exceeded 300 million for the first time, and the parameter counts of models
such as LLaMA [52], the GLM [53, 54], and GPT-3 [55] exceeded tens of billions. Large-scale
language models are perfect combinations of big data, high computing power, and advanced
algorithms. Studies have shown that they greatly enhance the pretraining and language generation capabilities of large models in general domains. Pradhan et al. [56] proposed semantic
attention optimization, stacking, bidirectional gated recurrent unit with semantic attention
(SRBi-GRU-SA), and multichannel word embedding models to represent a text by combining
sentiment information with semantic information obtained from a natural language model
(BERT). Nusrat et al. [57] proposed a converter-based approach for emoji prediction using
BERT. Talaat et al. [58] proposed four deep learning models that combined BERT with bidirectional long short-term memory (BiLSTM) and bidirectional gated recurrent unit (BiGRU)
algorithms. Yang et al. [59] proposed combining pretrained BERT models with temporal convolutional networks (TCNs) and graphical convolutional networks for short text classification
and emoji prediction. In addition, generative dialog models such as ChatGPT, which can capture complex linguistic patterns, have been investigated in sentiment analysis studies. Mithun
et al. [60] investigated the weaknesses of the ChatGPT model in terms of detecting emojibased hate speech. In cases involving emoji-based hate speech, the model performs poorly
when positive emojis are used in hate posts and fails to accurately label non-English-language
inputs. However, the aforementioned methods only utilize features acquired from text and
emojis without considering the introduction of topic features for sentiment analysis purposes.
Additionally, how to handle the continuous emergence of new network terms on social media
platforms has not been considered.
According to the analysis of a literature review, the current emoji-based sentiment analysis
method mainly adopts pretrained models. A pretrained model based on the transformer architecture utilizes a self-attention mechanism to achieve fast parallelism and increases the network depth to obtain additional global information. For scenarios in which new vocabulary
emerges, to further solve the problems concerning sparse semantic features and insufficient
sentiment expressions in short texts while considering the training cost, model efficiency, and
model performance factors, this study proposes adopting a new neural network model called

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
4 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

exBERT [61] (represented by transformers) as the foundation. This study focuses on extracting
and combining features derived from text, topics, and emoticons and improving the sentiment
accuracy rate achieved by conducting fine-tuning in specific domains.

Methodology
Preliminaries

Given a short textual passage that includes some emojis, such as “Nick has such cute balls

”, sentiment analysis is required to determine whether the sentiment orientation is positive, negative or neutral. Formally speaking, this task can be expressed as a function
F : T 7! P, which takes a short textual passage T as its input and returns P, indicating the
probability that the sentiment tendency of T is positive, negative or neutral.
In mainstream approaches, a pipeline consisting of an encoder E and a classifier C is
employed to solve sentiment analysis task, i.e., F ¼ E  C. The encoder is used to transform a
textual passage T into an embedding E, and the classifier outputs a probability P according to E.
Generally, the embedding E can be a d-dimensional vector that represents all the semantics of T
or a sequence consisting of N d-dimensional vectors that represent each of the N words in T.
The encoder is more significant because of its role in semantic comprehension, and in our
work, the encoder is enhanced with the emoji semantics and topics so that it can yield text
embeddings of higher quality to improve sentiment analysis.

## EMFSA model

The overall architecture of our EMFSA model is shown in Fig 1. The process starts with the
joint feature extraction module (emo_exBERT), which can capture the long-distance contextual dependencies between text and emojis to obtain all the semantic features of the input with
emojis for the subsequent sentiment analysis task. In the domain-specific additional pretraining stage of emo_exBERT, we propose a novel sentiment word priority-based masked language
model (Senti_MLM), which masks emojis and words with richer sentiments, exhibiting a
higher probability of prioritizing the learning of semantic representations and improving the
effect of the downstream sentiment analysis process.
Because textual passages on social platforms are usually short and casual in terms of their
expressions, they contain insufficient semantic information and excessive noise; to address
these issues, we inject two auxiliary pieces of information: descriptions of emojis and descriptions of topics. For the former, we use Emoji2Vec [27], which trains emoji representations on
the basis of text descriptions to embed the emojis; these embeddings are expected to enhance
the semantics of textual passages with emojis. For the latter, a biterm topic model (BTM) [49]
is trained and subsequently used to extract the topic feature of the whole input sentence and
each word in it. By comparing the topic consistency levels of words and their corresponding
sentences, we can learn which words are more significant, and greater emphasis is placed on
the semantics of these words so that the impact of noise can be mitigated. Finally, we design a
multifeature fusion module employing a cross-attention mechanism to fuse the features
extracted by emo_exBERT and the two auxiliary information sources; the output of this module is fed into a classifier composed of a fully connected layer, yielding the ultimate sentiment
analysis result.

## Joint feature extraction module (emo_exBERT)

Abundant research has indicated that high-quality semantic representations of text can be
obtained through pretrained language models (PLMs), thereby enhancing the performance

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
5 / 19

![page5_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page5_img1.jpeg)

![page5_img2.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page5_img2.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Fig 1. The overall architecture of our EMFSA model.

https://doi.org/10.1371/journal.pone.0310715.g001

achieved in various downstream tasks. However, the majority of the current PLMs are incapable of recognizing emojis, which poses a challenge when semantically representing text containing emojis. Inspired by exBERT [61], which extends the pretrained BERT model with
domain-specific vocabulary and has been validated in the biomedical domain, we propose a
joint feature extraction module, emo_exBERT, which represents the semantics of text and
emojis simultaneously. As shown in Fig 2, the module has the same structure as that of
exBERT, which employs a pretrained BERT model as its backbone and introduces two modifications. (1) An extra token embedding layer is added to embed domain-specific tokens (i.e.,
emojis here). (2) An expanded multihead self-attention submodule is added to each encoder
layer, the output of which is the weighted sum of the outputs of the original submodule and
the extended submodule. During the adaptive pretraining phase, only the parameters of the

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
6 / 19

![page6_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page6_img1.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Fig 2. Structure of the joint feature extraction module (emo_exBERT).

https://doi.org/10.1371/journal.pone.0310715.g002

emoji token embedding layer, the extension submodule and the weight generator need to be
updated, while the remaining parameters are frozen.
Different from exBERT, we use a corpus that includes emojis to pretrain our emo_exBERT
module for learning the features of emojis and the interactions between text and emojis. Given
a text passage T = {[CLS], w1, w2,   , wn}, where wi can be a normal word or an emoji, and

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
7 / 19

![page7_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page7_img1.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

[CLS] indicates the beginning of T, emo_exBERT module transforms the passage into a feature
sequence S = {h[CLS], h1, h2,   , hn}.

## Sentiment word priority-masked language model (Senti_MLM)

Masked language models (MLMs) are often used in the pretraining stages of pretrained language models such as BERT. Such a model randomly masks some tokens contained in the
input at a certain proportion and attempts to reconstruct them. Randomness implies equal
treatment for each word; however, in sentiment analysis tasks, the contributions of different
words usually vary. Therefore, we propose a sentiment word priority masked language model
(Senti_MLM), which masks emojis and words possessing richer sentiments with higher probabilities to prioritize learning their semantic representations and thus enhance the performance
achieved in the subsequent sentiment analysis task. The specific steps are as follows:
(1) The SentiWordNet sentiment dictionary [7] is queried, and the overall sentiment weight
of each token is calculated in the vocabulary of our emo_exBERT module. The overall sentiment weight swi of the i-th token wi is shown in Eq (1):

X
L

swi ¼ 1
L

pwi;l þ nwi;l < 1
ð1Þ

l¼1

where L is the number of meanings for wi in the SentiWordNet sentiment dictionary, and pwi,l
and nwi,l are the positive and negative sentiment weights corresponding to the l-th meaning of
wi, respectively. If wi does not exist in SentiWordNet and is not an emoji, swi = 0; if it is an
emoji, swi = 1.
(2) For each batch in the training corpus, its unnormalized masked probability matrix (mpi,j)
is computed, where mpi,j is the element in row i and column j of the matrix, which indicates the
unnormalized masked probability of the j-th token of the i-th textual passage in this batch, as
shown in Eq (2):

8
<

mpi;j ¼
1;
swi;j ¼ 0

:
ð2Þ

ð1 þ swi;jÞ  smr;
swi;j > 0

where swi,j denotes the overall sentiment weight of the j-th token of the i-th textual passage in
the batch, and smr indicates the sentiment word masked coefficient, which is used to increase
the relative masked probabilities of sentiment words.
(3) The unnormalized masked probability matrix is normalized so that the expected number of masked tokens in each batch is proportional to the total number of tokens in that batch,
with the ratio equal to the predefined normalization ratio (set to 0.15), as shown in Eq (3):

!

ðmpi;jÞn ¼ Normalize
mpi;j




¼
#ðmpi;jÞ  g  mpi;j
P

P

ð3Þ

jmpi;j

i

where (mpi,j)n is the normalized masked probability matrix, Normalize() denotes the normalize function, #(mpi,j) indicates the total number of elements in (mpi,j), and γ is the predefined
normalization ratio mentioned above.
An illustrative example is shown in Fig 3. For a batch that includes only one textual passage,
“Nick has such cute balls
”, we first calculate the overall sentiment weight of each token
(assuming that each word is a token; whitespace characters are not illustrated here), i.e., the
“senti_weight” row in the right block. Then, we compute the unnormalized masked probability matrix, the result of which is (1 10.6875 10.625 15.625 1 20 20), as shown in “P(mask)” row,

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
8 / 19

![page8_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page8_img1.jpeg)

![page8_img2.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page8_img2.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Fig 3. An illustrative example of the sentiment word priority-masked language model (Senti_MLM).

https://doi.org/10.1371/journal.pone.0310715.g003

and finally normalize the matrix to (0.013 0.139 0.138 0.203 0.013 0.26 0.26), i.e., the “normalized P(mask)” row.

## Multifeature fusion module

This module is aimed at fuse the joint feature of text and emoji extracted by emo_exBERT and
two additional features yielded by the introduced auxiliary information, named emoji description semantic feature and topic feature, respectively. Given a textual passage with emojis T =
{[CLS], w1, w2,   , wn}, the above two additional features can be obtained as follows:
Emoji description semantic feature. We first extract all the emojis in T and arrange them
in the original order to form a sequence Se ¼ fwF1; . . . ; wFmg, where wFi is an emoji, m is the
number of emojis in T and 1  F1 <    < Fm  n, and then lookup trained emoji embeddings provided by Emoji2Vec [27] to transform Se into an emoji embedding matrix
## Ee ¼ ðeF1; . . . ; eFmÞ

## T. Considering the impact of the order among the emojis contained in T on
semantic, we enhance the emoji embeddings using the parameters in the position embedding
layer of emo_exBERT, as shwon in Eq (4):

E0
e ¼ EeWprj þ Ep
ð4Þ

where E0
e is the enhanced emoji embedding matrix, i.e., emoji description semantic feature,

T is the corresponding position
embedding matrix.
Topic feature. We use biterm topic model (BTM) [49] to capture the topic vectors of the
whole input textual passage and each token in it. BTM is a short text topic model based on
word co-occurrence patterns. In contrast to LDA [62], which trains a separate topic distribution for each document in the given corpus, BTM trains a shared topic distribution for the
entire corpus, thereby resolving the problem concerning the sparse word co-occurrence patterns in individual documents. BTM assumes that the entire corpus is generated through the
following process:

Wprj is a trainable parameter, and Ep ¼ ðpeF1; . . . ; peFmÞ

(1) For each topic tk in the topic set (k = 1,   , K, where K is the number of topics), generate a
prior distribution Ftk  DirðbÞ for its corresponding word distribution.

(2) Generate a prior distribution θ * Dir(α) for the topic distribution of the entire corpus.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
9 / 19

![page9_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page9_img1.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

(3) For each word co-occurrence pair b contained in the corpus, first assign a topic t to it and
then generate the two words it contains bw1 and bw2, where bw1, bw2 * Multi(Ft).

Based on the generation process described above, the likelihood is as follows:

X

PðBÞ ¼ Pðbw1;bw2Þ2B

PðtÞPðbw1jtÞPðbw2jtÞ
ð5Þ

t

where B is the set of word co-occurrence pairs. By using the Gibbs sampling method to maximize the likelihood, the latent variables P(t) and P(w|t) can be obtained by Eqs (6) and (7),
respectively:

PðtÞ ¼
nt þ a
jBj þ Ka
ð6Þ

PðwjtÞ ¼
nwjt þ b
P

wnwjt þ Mb
ð7Þ

where nt denotes the number of word co-occurrence pairs assigned to topic t, nw|t denotes the
number of times word w is assigned to topic t, M is the size of the vocabulary, and α and β are
hyperparameters of the model.
After training the BTM, we calculate the topic vector of the whole textual passage T, i.e., the
sentence-topic distribution denoted by TD, using Eq (8):

TD ¼ ðPðt1jTÞ;    ; PðtKjTÞÞ
ð8Þ

where

!

X

## PðtkjTÞ ¼ normalizeT

PðtkjbÞ

ð9Þ

b2BT

PðtkjbÞ ¼ normalizebðPðtkÞPðbw1jtkÞPðbw2jtkÞÞ
ð10Þ

Here, bw1, bw2 2 b, BT indicates the set of word co-occurrence pairs in T, and normalizeT()

denotes the normalization operation such that P
K

PðtkjTÞ ¼ 1; normalizeb() is similar to that.

k¼1

The topic vectors of each token in T, i.e., the topic embedding matrix denoted by TE, can
be obtained by Eq (11):

TE ¼ ðpi;kÞ ¼ ðPðtkjwiÞÞ
ð11Þ

where

PðtkjwiÞ ¼ normalizewiðPðtkÞPðwijtkÞÞ
ð12Þ

Here, normalizewiðÞ is similar to normalizeT().
The ultimate topic feature TC for denoising is the topic consistency between wi (each token
in T) and T, as shown in Eq (13):

TC ¼ softmaxðTD  TETÞ
ð13Þ

Feature fusion. We fuse the three features of T: the joint feature of text and emojis
extracted by emo_exBERT S = {h[CLS], h1, h2,   , hn}, which can be split into hT
½CLS and a matrix
S¬[CLS] = (h1, h2,   , hn)T; the emoji description semantic feature E0
e; and the topic feature TC.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
10 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

The fusion operation is completed by Eq (14):

F ¼ hT
½CLS þ TC  AttentionðS:½CLSWQ; E0
eWK; E0
eWVÞ
ð14Þ

where WQ, WK and WV are the three weight matrices used in the cross-attention calculation
Attention(, , ), which is shown in Eq (15):

!

## AttentionðQ; K; VÞ ¼ sotfmax QKT

ffiffiffiffiffi
dk
p

V
ð15Þ

where dk is the number of columns of Q and K.

## Sentiment classification

The fusion feature F of the input textual passage T is fed into the classifier with a simple fully
connected layer, which output the relative sentiment distribution P = (pnegtive, pneutral, ppositive),
where pnegtive, pneutral and ppositive indicate the relative probability that the sentiment tendency
of T is negative, neutral and positive, respectively. Finally, the corresponding sentiment orientation of the maximum among these three probabilities is considered as the ultimate classification result.
We use the cross-entropy loss to optimize the parameters of our EMFSA model, as shown
in Eq (16):

X

log epcT
X

# L ¼ 1
jDj

epc
ð16Þ

T2D

c2C

where D denotes the train set, C denotes the label set, which includes three labels: negative,
neutral and positive, cT is the label of T and pc indicates the relative probability that the sentiment tendency of T is c, output by the above classifier.

Experiments

The experiments are divided into pretraining and fine-tuning stages. Sentiment- and topicbased pretraining is performed on a larger unlabeled sentiment corpus containing emoticons,
and fine-tuning is performed on three public datasets. The following three aspects are
described in terms of the utilized datasets and parameter settings: an experimental evaluation,
a model performance comparison, and ablation experiments.

## Datasets and setup

Our dataset is consistent with those of Yuan [42] and Nusrat et al. [57, 63]. EmojifyData-EN
[64] is a large-scale untagged Twitter dataset, and we restructure it to ensure that at least one
emoji is contained in each text of this dataset. SentiWordNet [7] is a lexicon for opinion mining that includes both positive and negative sentiment weights for each lexical sense of each
English word. In the experiments, the EmojifyData-EN dataset is used for model pretraining.
The Multidomain Sentiment Dataset (MSD) [65], Twitter Dataset (TD) and Emotion Recognition Dataset (ERD) [66] are the main sentiment analysis datasets used to validate the model
fine-tuning process. Table 1 provides detailed information about these three datasets.
In the investigations, only emoji data are used for each dataset. It is ensured that each of the
three datasets contains three classifications (negative, neutral and positive), and MSD_Emoji,
TD_Emoji and ERD_Emoji are constructed accordingly. The data of this study are preprocessed using the same procedure as that applied by Padmaja et al. [67]. Comprehensive

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
11 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Table 1. Emoji sentiment analysis fine-tuning dataset information.

Dataset
Number of text
Number of text with emoji
Positive
Neutral
Negative

MSD
60K
2212
1405
180
627

TD
162K
1786
816
679
291

ERD
5K
78
34
38
6

https://doi.org/10.1371/journal.pone.0310715.t001

validation of the model’s effectiveness is conducted through five-fold cross-validation on these
three public datasets, ensuring that each fold maintains a consistent data distribution, with
benign and malignant samples evenly distributed. The architecture is implemented in PyTorch
library on a workstation with an Intel Core i-7 and an NVIDIA GeForce RTX 3090 GPU. The
accuracy, macro-precision, macro-recall, and macro-F1 metrics are used to evaluate the performance of the model.

## Model performance comparison

To evaluate the performance of the proposed model, we select representative baseline methodologies. TextCNN [68] employs two CNNs to address multilabel classification problems. This
approach significantly outperforms the conventional support vector classification (SVC)-
based method. The AttBiLSTM [36] model uses a neural attention mechanism with a bidirectional long short-term memory network (BiLSTM) to capture the most essential semantic
information in a sentence. Based on the Emojis-Attention and BiLSTM models, EA-BiLSTM
[37] was the first method to utilize an attention model to capture the effect of emoticons on
the affective polarity of text. The epistemic symbol-based coattention network (ECN) [69] is
an emoji-based coattention network used to learn the mutual sentiment semantics between
text and emojis on microblogs; this method outperforms numerous baselines in sentiment
analyses of brief social media texts. EmoGraph2vec [42] learns emoji representations by constructing cooccurring graphical networks from social data and expanding the external Emoji-
Net knowledgebase of enriched semantic information to embed emoji nodes. The model
creates cutting-edge networks for emoji-containing texts. Our model is compared to the aforementioned baseline methods in terms of accuracy, and an experimental validation conducted
on publicly available datasets demonstrates that our method is superior for performing sentiment analysis tasks on text data containing emojis. Table 2 details the classification accuracies
attained by the baseline models on the test datasets.
On each of the three benchmark datasets, the EMFSA model obtains the highest level of
precision. The MSD_Emoji dataset yields a performance increase of 2.3%, the TD_Emoji dataset provides a performance increase of 10.9%, and the ERD_Emoji dataset demonstrates an
optimal classification performance increase of 2.7%.

Table 2. Classification accuracy of the baseline model on the datasets.

model
## Classification accuracy

MSD_Emoji
TD_Emoji
ERD_Emoji

TextCNN
0.8258
0.7202
0.6719

Att-BiLSTM
0.8243
0.7198
0.6700

EA-Bi-LSTM
0.8527
0.7470
0.7025

ECN
0.8552
0.7464
0.7256

EmoGraph2vec
0.8815"+2.3%
0.7703"+10.9%
0.7627"+2.7%

EMFSA (ours)
0.9016
0.8543
0.7833

https://doi.org/10.1371/journal.pone.0310715.t002

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
12 / 19

| Dataset | Number of text | Number of text with emoji | Positive | Neutral | Negative |
| --- | --- | --- | --- | --- | --- |
| MSD | 60K | 2212 | 1405 | 180 | 627 |
| TD | 162K | 1786 | 816 | 679 | 291 |
| ERD | 5K | 78 | 34 | 38 | 6 |

| model | Classification accuracy |  |  |
| --- | --- | --- | --- |
|  | MSD Emoji _ | TD Emoji _ | ERD Emoji _ |
| TextCNN | 0.8258 | 0.7202 | 0.6719 |
| Att-BiLSTM | 0.8243 | 0.7198 | 0.6700 |
| EA-Bi-LSTM | 0.8527 | 0.7470 | 0.7025 |
| ECN | 0.8552 | 0.7464 | 0.7256 |
| EmoGraph2vec | 0.8815"+2.3% | 0.7703"+10.9% | 0.7627"+2.7% |
| EMFSA (ours) | 0.9016 | 0.8543 | 0.7833 |

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Among the baseline models, TextCNN and Att-BiLSTM use traditional machine learning
methods such as convolutional neural networks, bidirectional long short-term memory
(BiLSTM), and attention mechanisms to capture the most essential semantic information in a
sentence, but their classification accuracies are poor. Even on the MSD_Emoji dataset, which
has a larger emoji corpus, the highest accuracy is only 82%; on the other two datasets, which
have sparse data resources, the highest accuracy achieved by these methods is only 72%. The
classification accuracies of these two models are not substantially enhanced. These methods
are incapable of capturing context-specific sentiment information in situations where sentiment semantics are inadequately expressed in brief texts. The EA-Bi-LSTM and ECN models,
which utilize bidirectional long short-term memory networks and attention models, are better
able to capture bidirectional semantic dependencies and perceive semantics. Both models
yield improvements of 3 points on the MSD_Emoji dataset, with an accuracy of up to 85%,
and nearly 3 points on the TD_Emoji and ERD_Emoji datasets, with an accuracy of up to
74%, compared to the conventional machine learning methods.
In 2022, the EmoGraph2vec was newly proposed; this model learns emoji representations
by embedding emoji nodes within a co-occurrence graph network and enriching semantic
information using the EmojiNet external knowledge base. The model derives emoji representations using both Emoji2Vec and Unicode characters; it achieves 88% accuracy on the
MSD_Emoji dataset, while the accuracy increases by nearly 3 percentage points to 77% on the
TD_Emoji and ERD_Emoji datasets. However, the classification effect of this method is insufficient for datasets with sparse corpora, and there is still considerable room for improving its
sentiment analysis accuracy when addressing the semantic sparseness of short texts, words
with multiple meanings, and diverse data modalities on social network platforms such as Twitter and Facebook.
The EMFSA model achieves the highest classification accuracy in comparison with the
baseline method when performing sentiment analysis tasks on emoji-containing text data
from the three benchmark datasets. The method designed for this model can be used to effectively mine the contextual semantics of short English texts on Twitter containing emojis, better
realize the complementarity of multiple features, and further improve its ability to recognize
and perceive contextual sentiments by embedding emoji features. A visualization of the outcomes of the accuracy comparison experiment conducted with the baseline models is shown
in Fig 4.

## Ablation experiment

To more rigorous determine the validity of the proposed model, we devise four ablation
schemes, and four evaluation indices areas are utilized: the accuracy, macro-precision, macrorecall, and macro-F1 scores. The ablation experiments are described as follows:

Fig 4. Visualization of the results of the accuracy comparison experiment conducted with the baseline models.

https://doi.org/10.1371/journal.pone.0310715.g004

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
13 / 19

![page13_img1.jpeg](EMFSA%20Emoji%20based%20multifeature%20fusion%20sentimet%20analysis_images/page13_img1.jpeg)

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Table 3. Results of ablation experiments conducted on MSD_Emoji.

Model
MSD_Emoji

Accuracy
Macro-Precision
Macro-Recall
Macro-F1

w/o topic
0.8948 ± 0.0227
0.7240 ± 0.0109
0.6986 ± 0.0046
0.7060 ± 0.0131

w/o fusing
0.8948 ± 0.0218
0.7200 ± 0.0337
0.7062 ± 0.0096
0.7147 ± 0.0097

w/o senti-mlm
0.8948 ± 0.0194
0.7135 ± 0.0269
0.6899 ± 0.0104
0.7103 ± 0.0174

w/o emoji
0.6872 ± 0.0055
0.5528 ± 0.0142
0.4588 ± 0.0170
0.4522 ± 0.0073

EMFSA (ours)
0.9016 ± 0.0283
0.7366 ± 0.0265
0.7199 ± 0.0172
0.7250 ± 0.0120

https://doi.org/10.1371/journal.pone.0310715.t003

• W/o topic: Topic features are not used; i.e., when multiple features are fused, the topic of
each word is set to have the same degree of conformity as that of the overall topic of the text.

• W/o fusion: The multifeature fusion module is not used; i.e., the h[CLS] output from emo_ex-
BERT are directly input into the classifier for sentiment classification.

• W/o senti-mlm: In the domain-specific additional pretraining stage of emo_exBERT, the
original masked language model (MLM) is used instead of our Senti_MLM.

• W/o emoji: Emoji information is not used, as follows: (1) Replace emo_exBERT with the
BERT model and use the EmojifyData-EN dataset with the emojis removed for additional
pretraining. (2) Remove all emojis from the dataset used when training the EMFSA model
via the sentiment analysis task. (3) Do not use the multifeature fusion module (same as w/o
fusing), as the data do not contain emojis at this point, and using this module would only
introduce noise.

• EMFSA (ours): This is the proposed EMFSA model.

Results obtained on the MSD_Emoji dataset.
The results of the experiments conducted
on the MSD_Emoji dataset are detailed in Table 3. Compared to the approach that does not
use emoji symbol information (w/o emoji), the proposed model achieves relative performance
improvements of 31.2%, 33.2%, 56.9%, and 60.3% under the four indices, accuracy, macroprecision, macro-recall, and the macro-F1 score, respectively. The first three experiments also
show that the use of topic features, feature fusion, and sentiment vocabulary-prioritized masking schemes all lead to performance improvements, where the emoji features and topic features are more effective at improving the accuracy of sentiment analysis. This finding verifies
the effectiveness of the sentiment analysis model proposed in this paper.
Results obtained on the ERD_Emoji dataset.
The results of the experiments conducted
on the ERD_Emoji dataset are detailed in Table 4. Compared to the method that does not use
emoji symbol information (w/o emoji), the model proposed in this paper achieves relative

Table 4. Results of ablation experiments conducted on ERD_Emoji.

Model
MSD_Emoji

Accuracy
Macro-Precision
Macro-Recall
Macro-F1

w/o topic
0.7833 ± 0.1333
0.5530 ± 0.0581
0.5250 ± 0.0584
0.5277 ± 0.0610

w/o fusing
0.7833 ± 0.1333
0.5530 ± 0.0581
0.5250 ± 0.0584
0.5277 ± 0.0610

w/o senti-mlm
0.7833 ± 0.1333
0.5530 ± 0.0581
0.5250 ± 0.0584
0.5277 ± 0.0610

w/o emoji
0.5333 ± 0.1333
0.3345 ± 0.0359
0.3750 ± 0.0250
0.3428 ± 0.0361

EMFSA (ours)
0.7833 ± 0.1333
0.5530 ± 0.0581
0.5250 ± 0.0584
0.5277 ± 0.0610

https://doi.org/10.1371/journal.pone.0310715.t004

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
14 / 19

| Model | MSD Emoji _ |  |  |  |
| --- | --- | --- | --- | --- |
|  | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| w/o topic | 0.8948 ± 0.0227 | 0.7240 ± 0.0109 | 0.6986 ± 0.0046 | 0.7060 ± 0.0131 |
| w/o fusing | 0.8948 ± 0.0218 | 0.7200 ± 0.0337 | 0.7062 ± 0.0096 | 0.7147 ± 0.0097 |
| w/o senti-mlm | 0.8948 ± 0.0194 | 0.7135 ± 0.0269 | 0.6899 ± 0.0104 | 0.7103 ± 0.0174 |
| w/o emoji | 0.6872 ± 0.0055 | 0.5528 ± 0.0142 | 0.4588 ± 0.0170 | 0.4522 ± 0.0073 |
| EMFSA (ours) | 0.9016 ± 0.0283 | 0.7366 ± 0.0265 | 0.7199 ± 0.0172 | 0.7250 ± 0.0120 |

| Model | MSD Emoji _ |  |  |  |
| --- | --- | --- | --- | --- |
|  | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| w/o topic | 0.7833 ± 0.1333 | 0.5530 ± 0.0581 | 0.5250 ± 0.0584 | 0.5277 ± 0.0610 |
| w/o fusing | 0.7833 ± 0.1333 | 0.5530 ± 0.0581 | 0.5250 ± 0.0584 | 0.5277 ± 0.0610 |
| w/o senti-mlm | 0.7833 ± 0.1333 | 0.5530 ± 0.0581 | 0.5250 ± 0.0584 | 0.5277 ± 0.0610 |
| w/o emoji | 0.5333 ± 0.1333 | 0.3345 ± 0.0359 | 0.3750 ± 0.0250 | 0.3428 ± 0.0361 |
| EMFSA (ours) | 0.7833 ± 0.1333 | 0.5530 ± 0.0581 | 0.5250 ± 0.0584 | 0.5277 ± 0.0610 |

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

Table 5. Results of ablation experiments conducted on TD_Emoji.

Model
TD_Emoji

Accuracy
Macro-Precision
Macro-Recall
Macro-F1

w/o topic
0.8373 ± 0.0222
0.8167 ± 0.0364
0.8030 ± 0.0213
0.8085 ± 0.0278

w/o fusing
0.8375 ± 0.0252
0.8207 ± 0.0382
0.7953 ± 0.0182
0.8040 ± 0.0247

w/o senti-mlm
0.8522 ± 0.0194
0.8289 ± 0.0311
0.8184 ± 0.0244
0.8229 ± 0.0260

w/o emoji
0.4568 ± 0.0417
0.3204 ± 0.0073
0.3396 ± 0.0117
0.3091 ± 0.0182

EMFSA (ours)
0.8543 ± 0.0280
0.8321 ± 0.0318
0.8131 ± 0.0226
0.8206 ± 0.0308

https://doi.org/10.1371/journal.pone.0310715.t005

performance improvements of 46.9%, 65.3%, 40%, and 53.9% under the four metrics, accuracy, macro-precision, macro-recall, and the macro-F1 score, respectively. However, the
experimental results obtained using the thematic features, feature fusion, and sentiment vocabulary-prioritized masking schemes are consistent with those of the EMFSA model; moreover,
the main reason for this finding is that this dataset is small, and the model is unable to make
full use of its cross-attention mechanism to extract features.
Results obtained on the TD_Emoji dataset.
The results of the experiments conducted on
the TD_Emoji dataset are detailed in Table 5. Compared to those of the method that does not
use emoji symbol information (w/o emoji), the four indicators exhibit relative performance
improvements of 87%, 159%, 139%, and 165%. Compared with the model using thematic features and feature fusion, the relative performance improvement is greater than 2.5% on average. The macro-recall and macro-F1 metrics are slightly lower than those attained without
using senti_mlm. This may be because the proportion of sentiment words included in TD_Emoji is low, so the senti_mlm mechanism is not significantly different from the normal MLM.
The randomness of the actual masking process leads to better model learning of during pretraining when using the MLM.

Conclusion

Emojis provide important information about users’ sentiments when analyzing short informal
texts such as tweets, blogs, or comments. We propose EMFSA, a sentiment analysis model
with multifeature fusion based on emoticons, to address the issues of undirected sentiment
expressions, “multiple meanings of words,” and the low accuracy attained by sentiment analysis methods for brief English texts on social platforms. The developed model uses a combination of multiple features, such as emoticons, themes, and texts, to sentimentally mine various
modes of information and achieve intermodal information enhancement and complementation. Specifically, the model utilizes the semantic and sentiment functions of emoticons to fully
mine contextual semantics, fill in visual cues in the employed text corpus, improve the ability
of the model to process brief text information, and effectively express and enhance the sentiments contained in the text.
In this paper, the EMFSA model achieves optimal performance on all three public benchmark datasets, with relative accuracy improvements of 2.3%, 10.9%, and 2.7%. To further demonstrate the efficacy of emojis for the sentiment analysis task, we devise four scenarios for
ablation experiments. Utilizing the MSD_Emoji dataset for illustrative purposes, the experimental results demonstrate that the EMFSA model achieves relative improvements of 31.2%,
33.2%, 56.9%, and 60.3% over an ablation model using emoji symbol information without
emojis (w/o emoji) in terms of the accuracy, macro-precision, macro-recall, and macro-F1
metrics, respectively. The results of the experiments indicate that emoticons can effectively
enhance the sentiments of text. In addition, the sentiment word priority masking model, the

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
15 / 19

| Model | TD Emoji _ |  |  |  |
| --- | --- | --- | --- | --- |
|  | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| w/o topic | 0.8373 ± 0.0222 | 0.8167 ± 0.0364 | 0.8030 ± 0.0213 | 0.8085 ± 0.0278 |
| w/o fusing | 0.8375 ± 0.0252 | 0.8207 ± 0.0382 | 0.7953 ± 0.0182 | 0.8040 ± 0.0247 |
| w/o senti-mlm | 0.8522 ± 0.0194 | 0.8289 ± 0.0311 | 0.8184 ± 0.0244 | 0.8229 ± 0.0260 |
| w/o emoji | 0.4568 ± 0.0417 | 0.3204 ± 0.0073 | 0.3396 ± 0.0117 | 0.3091 ± 0.0182 |
| EMFSA (ours) | 0.8543 ± 0.0280 | 0.8321 ± 0.0318 | 0.8131 ± 0.0226 | 0.8206 ± 0.0308 |

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

biterm topic model, and the cross-attention fusion mechanism can further improve the
semantic representations of short text sentiments, thereby enhancing the accuracy achieved in
short text sentiment analysis tasks.
In our future work, we will investigate a method for fusing emojis with multimodal data
(video, audio, and images). Detecting sarcasm and irony in text is also a challenge in the field
of natural language processing. We will investigate whether sarcasm and irony can be handled
with the aid of emoticons to increase the accuracy of sentiment analysis models.

## Author Contributions

Conceptualization: Hongmei Tang.

Data curation: Dixiongxiao Zhu.

Formal analysis: Hongmei Tang, Dixiongxiao Zhu.

Funding acquisition: Shuai Wang.

Investigation: Hongmei Tang.

Methodology: Hongmei Tang, Wenzhong Tang.

Project administration: Wenzhong Tang, Shuai Wang, Yanyang Wang, Lihong Wang.

Software: Dixiongxiao Zhu.

Supervision: Wenzhong Tang, Yanyang Wang, Lihong Wang.

Validation: Dixiongxiao Zhu.

Visualization: Shuai Wang.

Writing – original draft: Hongmei Tang.

Writing – review & editing: Shuai Wang.

References

1.
Munezero M, Montero CS, Sutinen E, Pajunen J. Are they different? Affect, feeling, emotion, sentiment,
and opinion detection in text. IEEE transactions on affective computing. 2014; 5(2):101–111. https://doi.
org/10.1109/TAFFC.2014.2317187

2.
Liu B. Sentiment Analysis and Opinion Mining. Morgan & Claypool Publishers; 2012.

3.
Dimson T. Emojineering part 1: Machine learning for emoji trends. Instagram Engineering Blog. 2015;
30.

4.
Al-Azani S, El-Alfy ESM. Early and late fusion of emojis and text to enhance opinion mining. IEEE
Access. 2021; 9:121031–121045. https://doi.org/10.1109/ACCESS.2021.3108502

5.
Daft RL, Lengel RH. Organizational information requirements, media richness and structural design.
Management science. 1986; 32(5):554–571. https://doi.org/10.1287/mnsc.32.5.554

6.
Ferna´ndez-Gavilanes M, Juncal-Martı´nez J, Garcı´a-Me´ndez S, Costa-Montenegro E, Gonza´lez-Castano FJ. Creating emoji lexica from unsupervised sentiment analysis of their descriptions. Expert Systems with Applications. 2018; 103:74–91. https://doi.org/10.1016/j.eswa.2018.02.043

7.
Baccianella S, Esuli A, Sebastiani F, et al. Sentiwordnet 3.0: an enhanced lexical resource for sentiment
analysis and opinion mining. In: Lrec. vol. 10; 2010. p. 2200–2204.

8.
Hu M, Liu B. Mining and summarizing customer reviews. In: Proceedings of the tenth ACM SIGKDD
international conference on Knowledge discovery and data mining; 2004. p. 168–177.

9.
Mai L, Le B. Joint sentence and aspect-level sentiment analysis of product comments. Annals of Operations research. 2021; 300:493–513. https://doi.org/10.1007/s10479-020-03534-7

10.
Subhashini L, Li Y, Zhang J, Atukorale AS, Wu Y. Mining and classifying customer reviews: a survey.
Artificial Intelligence Review. 2021; p. 1–47.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
16 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

11.
Chen L, Li T, Luo H, Yin C. Interactive Attention-Based Convolutional GRU for Aspect Level Sentiment
Analysis. Human-Centric Intelligent Systems. 2021; 1(1-2):25–31. https://doi.org/10.2991/hcis.k.
210704.002

12.
Huang F, Yuan C, Bi Y, Lu J. Exploiting long-term dependency for topic sentiment analysis. IEEE
Access. 2020; 8:221963–221974. https://doi.org/10.1109/ACCESS.2020.3039963

13.
Mahilraj J, Tigistu G, Tumsa S. Text preprocessing method on Twitter sentiment analysis using machine
learning. International Journal of Innovative Technology and Exploring Engineering. 2020; 9(12):233–
240. https://doi.org/10.35940/ijitee.K7771.0991120

14.
Wang Y, Huang M, Zhu X, Zhao L. Attention-based LSTM for aspect-level sentiment classification. In:
Proceedings of the 2016 conference on empirical methods in natural language processing; 2016.
p. 606–615.

15.
Li Y, Li N. Sentiment analysis of Weibo comments based on graph neural network. IEEE Access. 2022;
10:23497–23510. https://doi.org/10.1109/ACCESS.2022.3154107

16.
Bataa E, Wu J. An Investigation of Transfer Learning-Based Sentiment Analysis in Japanese. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics; 2019. p. 4652–
4657.

17.
Gong C, Yu J, Xia R. Unified feature and instance based domain adaptation for aspect-based sentiment
analysis. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP); 2020. p. 7035–7045.

18.
Sun C, Huang L, Qiu X. Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary
Sentence. In: Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers);
2019. p. 380–385.

19.
Li X, Bing L, Zhang W, Lam W. Exploiting BERT for End-to-End Aspect-based Sentiment Analysis. In:
Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019); 2019. p. 34–41.

20.
Song Y, Wang J, Liang Z, Liu Z, Jiang T. Utilizing BERT intermediate layers for aspect based sentiment
analysis and natural language inference. arXiv preprint arXiv:200204815. 2020;.

21.
Xu H, Liu B, Shu L, Philip SY. BERT Post-Training for Review Reading Comprehension and Aspectbased Sentiment Analysis. In: Proceedings of the 2019 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and
Short Papers); 2019. p. 2324–2335.

22.
Rietzler A, Stabinger S, Opitz P, Engl S. Adapt or Get Left Behind: Domain Adaptation through BERT
Language Model Finetuning for Aspect-Target Sentiment Classification. In: Proceedings of the Twelfth
Language Resources and Evaluation Conference; 2020. p. 4933–4941.

23.
Jiang T, Sun W, Wang M. MSGAT-Based Sentiment Analysis for E-Commerce. Information. 2023; 14
(7):416. https://doi.org/10.3390/info14070416

24.
Anggrainingsih R, Hassan GM, Datta A. CE-BERT: Concise And Efficient BERT-based Model for
Detecting Rumours on Twitter. IEEE Access. 2023;. https://doi.org/10.1109/ACCESS.2023.3299858

25.
Zhang X, Ma Y. An ALBERT-based TextCNN-Hatt hybrid model enhanced with topic knowledge for
sentiment analysis of sudden-onset disasters. Engineering Applications of Artificial Intelligence. 2023;
123:106136. https://doi.org/10.1016/j.engappai.2023.106136

26.
Kralj Novak P, Smailović J, Sluban B, Mozetič I. Sentiment of emojis. PloS one. 2015; 10(12):
e0144296. https://doi.org/10.1371/journal.pone.0144296 PMID: 26641093

27.
Eisner B, Rockta¨schel T, Augenstein I, Bosnjak M, Riedel S. emoji2vec: Learning Emoji Representations from their Description. In: Proceedings of the Fourth International Workshop on Natural Language
Processing for Social Media; 2016. p. 48–54.

28.
Wijeratne S, Balasuriya L, Sheth A, Doran D. Emojinet: An open service and api for emoji sense discovery. In: Proceedings of the International AAAI Conference on Web and Social Media. vol. 11; 2017.
p. 437–446.

29.
Wu J, Lu K, Su S, Wang S. Chinese micro-blog sentiment analysis based on multiple sentiment dictionaries and semantic rule sets. IEEE Access. 2019; 7:183924–183939. https://doi.org/10.1109/
ACCESS.2019.2960655

30.
Vora P, Khara M, Kelkar K. Classification of tweets based on emotions using word embedding and random forest classifiers. International Journal of Computer Applications. 2017; 178(3):1–7. https://doi.org/
10.5120/ijca2017915773

31.
Raza AA, Habib A, Ashraf J, Javed M. Semantic orientation based decision making framework for big
data analysis of sporadic news events. Journal of Grid Computing. 2019; 17:367–383. https://doi.org/
10.1007/s10723-018-9466-y

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
17 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

32.
Hauthal E, Burghardt D, Dunkel A. Analyzing and visualizing emotional reactions expressed by emojis
in location-based social media. ISPRS International Journal of Geo-Information. 2019; 8(3):113. https://
doi.org/10.3390/ijgi8030113

33.
Ekman P, Friesen WV. Hand movements. Journal of communication. 1972; 22(4):353–374. https://doi.
org/10.1111/j.1460-2466.1972.tb00163.x

34.
Ferna´ndez-Gavilanes M, Costa-Montenegro E, Garcı´a-Me´ndez S, Gonza´lez-Castaño FJ, Juncal-Martı´-
nez J. Evaluation of online emoji description resources for sentiment analysis purposes. Expert Systems with Applications. 2021; 184:115279. https://doi.org/10.1016/j.eswa.2021.115279

35.
Wang W, Chen L, Thirunarayan K, Sheth AP. Harnessing twitter “big data” for automatic emotion identification. In: 2012 International Conference on Privacy, Security, Risk and Trust and 2012 International
Confernece on Social Computing. IEEE; 2012. p. 587–592.

36.
Zhou P, Shi W, Tian J, Qi Z, Li B, Hao H, et al. Attention-based bidirectional long short-term memory
networks for relation classification. In: Proceedings of the 54th annual meeting of the association for
computational linguistics (volume 2: Short papers); 2016. p. 207–212.

37.
Lou Y, Zhang Y, Li F, Qian T, Ji D. Emoji-based sentiment analysis using attention networks. ACM
Transactions on asian and low-resource language information processing (TALLIP). 2020; 19(5):1–13.
https://doi.org/10.1145/3389035

38.
Tomihira T, Otsuka A, Yamashita A, Satoh T. Multilingual emoji prediction using BERT for sentiment
analysis. International Journal of Web Information Systems. 2020; 16(3):265–280. https://doi.org/10.
1108/IJWIS-09-2019-0042

39.
Li X, Zhang J, Du Y, Zhu J, Fan Y, Chen X. A novel deep learning-based sentiment analysis method
enhanced with Emojis in microblog social networks. Enterprise Information Systems. 2023; 17
(5):2037160. https://doi.org/10.1080/17517575.2022.2037160

40.
Yu Y, Qiu D, Yan R. A multi-modal and multi-scale emotion-enhanced inference model based on fuzzy
recognition. Complex & Intelligent Systems. 2021; p. 1–14.

41.
Hajek P, Barushka A, Munk M. Neural networks with emotion associations, topic modeling and supervised term weighting for sentiment analysis. International journal of neural systems. 2021; 31
(10):2150013. https://doi.org/10.1142/S0129065721500131 PMID: 33573532

42.
Yuan X, Hu J, Zhang X, Lv H. Pay attention to emoji: Feature Fusion Network with EmoGraph2vec
Model for Sentiment Analysis. In: 2022 26th International Conference on Pattern Recognition (ICPR).
IEEE; 2022. p. 1529–1535.

43.
Shaik A, Devi BA, Baskaran R, Bojjawar S, Vidyullatha P, Balaji P. Recurrent neural network with
emperor penguin-based Salp swarm (RNN-EPS2) algorithm for emoji based sentiment analysis. Multimedia Tools and Applications. 2023; p. 1–20.

44.
Venkataraman J, Mohandoss L. FBO-RNN: Fuzzy butterfly optimization-based RNN-LSTM for extracting sentiments from Twitter Emoji database. Concurrency and Computation: Practice and Experience.
2023; 35(12):e7683. https://doi.org/10.1002/cpe.7683

45.
Du X, Zhu R, Zhao F, Zhao F, Han P, Zhu Z. A deceptive detection model based on topic, sentiment,
and sentence structure information. Applied Intelligence. 2020; 50:3868–3881. https://doi.org/10.1007/
s10489-020-01779-0

46.
Tan X, Zhuang M, Lu X, Mao T. An analysis of the emotional evolution of large-scale Internet public
opinion events based on the BERT-LDA hybrid model. IEEE Access. 2021; 9:15860–15871. https://doi.
org/10.1109/ACCESS.2021.3052566

47.
Haque MU, Dharmadasa I, Sworna ZT, Rajapakse RN, Ahmad H. “I think this is the most disruptive
technology”: Exploring Sentiments of ChatGPT Early Adopters using Twitter Data. arXiv preprint
arXiv:221205856. 2022;.

48.
Taecharungroj V. “What Can ChatGPT Do?” Analyzing Early Reactions to the Innovative AI Chatbot on
Twitter. Big Data and Cognitive Computing. 2023; 7(1):35. https://doi.org/10.3390/bdcc7010035

49.
Yan X, Guo J, Lan Y, Cheng X. A biterm topic model for short texts. In: Proceedings of the 22nd international conference on World Wide Web; 2013. p. 1445–1456.

50.
Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need.
Advances in neural information processing systems. 2017; 30.

51.
Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding. In: Proceedings of the 2019 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and
Short Papers); 2019. p. 4171–4186.

52.
Touvron H, Lavril T, Izacard G, Martinet X, Lachaux MA, Lacroix T, et al. Llama: Open and efficient
foundation language models. arXiv preprint arXiv:230213971. 2023;.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
18 / 19

# PLOS ONE
## EMFSA: Emoji-based multifeature fusion sentiment analysis

53.
Zeng A, Liu X, Du Z, Wang Z, Lai H, Ding M, et al. Glm-130b: An open bilingual pre-trained model. arXiv
preprint arXiv:221002414. 2022;.

54.
Du Z, Qian Y, Liu X, Ding M, Qiu J, Yang Z, et al. GLM: General Language Model Pretraining with Autoregressive Blank Infilling. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); 2022. p. 320–335.

55.
Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, et al. Language models are few-shot
learners. Advances in neural information processing systems. 2020; 33:1877–1901.

56.
Pradhan A, Ranjan Senapati M, Sahu PK. A multichannel embedding and arithmetic optimized stacked
Bi-GRU model with semantic attention to detect emotion over text data. Applied Intelligence. 2023; 53
(7):7647–7664. https://doi.org/10.1007/s10489-022-03907-4

57.
Nusrat MO, Habib Z, Alam M. Emoji Prediction using Transformer Models. arXiv preprint
arXiv:230702054. 2023;.

58.
Talaat AS. Sentiment analysis classification system using hybrid BERT models. Journal of Big Data.
2023; 10(1):1–18. https://doi.org/10.1186/s40537-023-00781-w

59.
Yang Z, Ye X, Xu H. TGCN-Bert Emoji Prediction in Information Systems Using TCN and GCN Fusing
Features Based on BERT. International Journal on Semantic Web and Information Systems (IJSWIS).
2023; 19(1):1–16. https://doi.org/10.4018/IJSWIS.337598

60.
Das M, Pandey SK, Mukherjee A. Evaluating ChatGPT’s Performance for Multilingual and Emoji-based
Hate Speech Detection. arXiv preprint arXiv:230513276. 2023;.

61.
Tai W, Kung H, Dong XL, Comiter M, Kuo CF. exBERT: Extending pre-trained models with domain-specific vocabulary under constrained training resources. In: Findings of the Association for Computational
Linguistics: EMNLP 2020; 2020. p. 1433–1439.

62.
Blei DM, Ng AY, Jordan MI. Latent dirichlet allocation. Journal of machine Learning research. 2003; 3
(Jan):993–1022.

63.
Nusrat MO, Habib Z, Alam M, Jamal SA. Emoji Prediction in Tweets using BERT; 2023.

64.
EmojifyData-EN: English tweets, with emojis; 2019. [Online]. Available from: https://www.kaggle.com/
datasets/rexhaif/emojifydata-en/.

65.
Generic Sentiment Multidomain Sentiment Dataset; 2020. [Online]. Available from: https://www.kaggle.
com/datasets/akgeni/generic-sentiment-multidomain-sentiment-dataset/.

66.
Twitter and Reddit Sentimental analysis Dataset; 2019. [Online]. Available from: https://www.kaggle.
com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset/.

67.
Padmaja K, Hegde NP. Twitter sentiment analysis using adaptive neuro-fuzzy inference system with
genetic algorithm. In: 2019 3rd international conference on computing methodologies and communication (ICCMC). IEEE; 2019. p. 498–503.

68.
Kim Y. Convolutional neural networks for sentence classification. arXiv preprint arXiv:14085882. 2014;.

69.
Yuan X, Hu J, Zhang X, Lv H, Liu H. Emoji-Based Co-Attention Network for Microblog Sentiment Analysis. In: Neural Information Processing: 28th International Conference, ICONIP 2021, Sanur, Bali, Indonesia, December 8–12, 2021, Proceedings, Part V 28. Springer; 2021. p. 3–11.

PLOS ONE | https://doi.org/10.1371/journal.pone.0310715
September 19, 2024
19 / 19
