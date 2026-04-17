# Multimodal text-emoji fusion using deep neural networks for text-based emotion detection in online communication

Kusal et al. Journal of Big Data           (2025) 12:32
https://doi.org/10.1186/s40537-025-01062-4
## Journal of Big Data

## Open Access

RESEARCH

Multimodal text‑emoji fusion using deep
neural networks for text‑based emotion
detection in online communication

## Sheetal Kusal1, Shruti Patil1,2* and Ketan Kotecha1,2,3

*Correspondence:
shruti.patil@sitpune.edu.in

Abstract

The task of emotion detection in online social communication has been explored
extensively. However, these studies solely focus on textual cues. Nowadays, emojis
have become increasingly popular, serving as a visual means to express emotions
and ideas succinctly. These emojis can be used supportively or contrastively, even
sarcastically, adding complexity to emotional interpretation. Therefore, incorporating
emoji analysis is crucial for accurately extracting insights from social media content
to support decision-making. This paper aims to investigate to what extent the usage
of emojis can contribute to the automated detection of emotions in text messages
with a focus on online social communication. We propose an emoji-aware hybrid deep
learning framework for multimodal emotion detection. The proposed framework leverages the feature-level fusion of textual and emoji representations, incorporating conventional and recurrent neural networks, to learn the fused modalities. The proposed
approach was extensively evaluated on the GoEmotions dataset with different performance metrics. The experimental results indicate that emoji features can significantly
improve emotion classification accuracy, highlighting their potential for enriching
emotion understanding in online social communication.

## 1 Symbiosis Institute
of Technology, Symbiosis
International (Deemed
University), Lavale, Maharashtra
412115, India
## 2 Symbiosis Centre for Applied
Artificial Intelligence, Symbiosis
International (Deemed
University), Lavale, Maharashtra
412115, India
## 3 Peoples’ Friendship University
of Russia, RUDN University,
Miklukho‑Maklaya Str.6, Moscow,
Russian Federation 117198,
Russia

Keywords:  Multimodal, Fusion, Emoji, Text-based emotion detection, Deep learning,
## Early fusion

Introduction
The era of Web 4.0, a symbiotic or mobile web, symbolises the advancement of internet technologies to their next stage. This advancement will have a symbiotic relationship
between humans and machines, where machines will learn from human interactions
and help humans make decisions. The Internet has become an integral part of human
life, where people generate social content by communicating through social media sites
and reviews by sharing experiences and opinions on topics of interest. Likewise, with the
advent of Artificial Intelligence (AI), people interact with intelligent dialogue machines
in conversational AI systems. Understanding the sentiments and emotions behind this
massive amount of online information has become a necessity. Emotional expressions
and detection have become crucial parts of human–machine interaction [1] People

© The Author(s) 2025. Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you
give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified
the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The
images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a
credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by
statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of
this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.

Page 2 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

express deep and complex emotions with a handful of words, and it has remained a challenge for machines to identify emotions and sentiments.
Human communication involves a variety of senses or modalities. These modalities
work in concert to clarify concepts and emphasize ideas in discussions by resolving
ambiguity. Similarly, nowadays, people are using various modalities in digital communications. People have adopted diversity in writing styles when it comes to communication
with Individuals. One of the popular writing styles people favour these days is emojis
in their interactions. The Oxford Dictionary states [2],” Emoji is a small digital image or
icon used to express an idea or emotion in electronic communication”. The word “emoji”
is borrowed from the Japanese language, and it is partitioned as” e”, meaning”picture”,
and”moji” meaning” letter or character”. Emojis consist of smilies and hearts and are
essentially the subsequent natural development of emoticons such as:) And:D. These
days, emoticon means objects represented by standard character sequences, and emoji
means self-reported labels as a piece of visual information users provide to express emotions in textual interactions to improve interpretability, particularly in short, complex
interactions. Emojis can be found in various genres, such as smiles and people, animals
and nature, food and drinks, travel and places, activities and objects. There are thousands upon thousands of emojis, each with a distinct meaning. This visual language is
currently used as the de facto standard for online communication and can be found on
Facebook, Instagram, WhatsApp, Twitter, and other major websites.
This visual information [2] in written text can express affective states, making it an
appropriate means of expressing sentiment and emotion. These emojis and the text
represent the user’s emotional state more faithfully. Analysing the text without emojis
would lead to the loss of important information that conveys people’s feelings or clarifies the meaning of the text. In some instances, such as sarcastic phrases [3] a user may
convey positive emotions in the text, but by utilising emoticons or emojis, they may convey negative emotions. In these situations, emojis might assist in determining the user’s
actual feelings. Emojis and emoticons are visual cues that aid in understanding natural
language. For example, I’m satisfied with the service!

In contrast, emojis are also not considered a direct expression of the emotions. Emojis
can have several meanings based on the context and other emojis since they are defined
without precise semantics [4]. So, text and emojis can play a complementary role in textbased emotion detection to comprehend people’s emotions. Table 1 presents the sample
text along with the accurate emotion label, the text emotion label, and the emoji emotion
label. Despite their prevalence as language form, emojis and their underlying semantics
have not been widely studied from a natural language processing (NLP) perspective.
The complementary role of text and emojis is not explored in-depth by researchers who
have combined features of text and emojis. So, in this work, the authors propose treating
emojis as a new modality with text to enhance emotion detection. The text with emojis
could accurately communicate the genuine feelings or emotions of the people they are
trying to express.
The usage of deep learning techniques has increased with the introduction of big data
[5]. Specifically, in sentiment analysis and emotion detection research, deep learning
techniques have played significant roles. Deep learning techniques improve accuracy
and performance by automatically learning and extracting information. Techniques like

Page 3 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Table 1  Sample Sentences with emotion label of text from the dataset, emotion label of emoji and
valid sentence label

Convolutional neural network (CNN), Recurrent Neural Networks (RNN), Long Short-
Term Memory (LSTM), Gated Recurrent Units (GRU), and their bi-directional variants,
transformer-based models like BERT have proved highly effective in natural language
processing tasks such as sentiment analysis, emotion detection. Advanced deep learning techniques [6] such as Generative Adversarial Net- works (GAN), Autoencoder (AE),
Graph Neural Networks and Attention mechanisms are progressively used in NLP tasks.
Emojis embedded in the text can help to recognize feelings or emotions; this paper
aims to identify emotions using deep learning techniques from text-based online communication with emojis. This study makes comprehensive use of emojis in text, as well
as the relation between text and emojis. A multimodal deep learning model based on
convolutional and recurrent networks with emojis is proposed for emotion detection in
text-based online communication. Therefore, this multimodal model we develop must
determine which underlying emotion class the input corresponds to given one or more
input modalities. The authors used an emoji-text integrated GoEmotions dataset to propose a multimodal deep-learning model for better emotion detection. The main contributions of this paper are summarized as follows:

1.	 We perform a comprehensive literature review on multimodal studies in natural language processing applications. Similarly, we review sentiment and emotion classification of emojis-related studies in social media analysis.
## 2.	 We propose an emoji-ware multimodal hybrid framework, Multimodal Text- Emoji
Fusion, based on a convolutional and recurrent neural network for emotion detection classification for text-based online communication.
## 3.	 We investigate the role of emojis in the feature-level fusion of text and emojis in
emotion detection and multimodality in text-based emotion detection.
4.	 We conduct an intensive empirical analysis of the proposed model on the GoEmotions dataset using different performance measures.

![page3_img1.png](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page3_img1.png)

Page 4 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

The remainder of this paper is organized as follows. Sect. "Related Work" introduces
the related work on multimodal fusion, unimodal and multimodal emotion detection,
discussing deep learning-based techniques in emotion detection. Sect.  "Multimodal
Fusion" provides an overview of the dataset. A detailed explanation of the emoji-aware
proposed framework is found in Sect.  "Early Fusion". Sect.  "Late Fusion" provides a
detailed description of the experimental setup and the analysis of the results obtained
from the GoEmotions dataset. Sect. "Intermediate Fusion" provides concluding remarks
and presents the future research directions.

Related work
Emotion detection has become increasingly popular in the academic and research communities recently. There are various ways to detect human emotions, such as speech and
facial emotion recognition. Similarly, through EEG signals, emotions can be detected.
However, text-based emotion detection needs to be explored considering the latest
developments in online text communication and the progress to increasing the emotional intelligence of human–computer interaction. As people are using multiple visual
information such as emojis, emotion recognition from text and emojis needs to be studied. This section surveys literature from three perspectives: (1) Multimodal Fusion (2)
Unimodal text-based emotion detection, (3) Multimodal Emotion detection using text
and emojis.

Multimodal fusion
In today’s online communication era, multiple modes are used to communicate and
present ideas and information. Information is disseminated through written text, audio,
and visuals, including images, videos, and emojis. Social networking sites are accountable for the continuous increase in the amount of information offered in visual forms.
The potential for communication that visual representation offers is diverse. According
to [14], the world as it is presented differs from the world as it is narrated or written.
The term”multimodality” describes the presentation of information by combining or’in
a sense fusing’ many media formats. This fused representation of different media like
text and visual or text and audio helps users to describe and understand the context in
a profound manner. Multimodality in natural language processing tasks has remained
in focus in recent years in the form of different modalities. Tasks include caption generation, speech recognition, sentiment analysis, and emotion recognition, where visual
and audio with text modalities have been used. A summary of various representative
multimodality-based NLP task works is provided in Table 2. The authors analyzed these
works based on the modalities used, the kind of fusion, the approach employed, and the
tasks for which it is utilized. Figure 1 shows the basic outline of the multimodal system.
The process of integrating or fusing data from two or more modalities to make a prediction, classification or regression is known as fusion in multimodality. The multimodal
data fusion process is characterized by the level at which the fusion of input modalities
occurs in the network, like early fusion, late fusion, and intermediate fusion. Figure 2
presents the overview of the early and late fusion systems.

Page 5 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Table 2  Literature survey based on types of modalities used, fusion types used, and type task
performed using multimodal fusion

References
Type of modalities
Fusion types
Technique
Task

[7]
Visual, text, acoustic
Early fusion/crossmodal
Deep Learning—Encoder
with an attention mechanism
## Sentiment analysis

[8]
Visual, text, acoustic
Early fusion
Deep learning—LSTM, BERT,
CNN
## Sentiment analysis

[9]
Text, acoustic
Late fusion
Deep Learning—Attention
with auto- encoder, Bi-LSTM
## Emotion recognition

[10]
Visual, text, emoji,
acoustic
Early fusion
Deep Learning—Bi-GRU,
Gated attention mechanism
## Sarcasm detection

[11]
Image, text
Early fusion
Deep Learning—RNN, Pretrained CNN
## Machine translation

[12]
Text, emoji
Late fusion
Deep learning—Bi− GRU
with attention
## Emotion detection

[12]
Text, emoji
Late fusion
SVM, Deep learning—Bi-
# LSTM, CNN
## Sentiment analysis

[13]
Text, acoustic
Late fusion
Deep learning—Bi-RNN, Self-
Attention, BERT
## Emotion recognition

[2]
Text, emoji
Early fusion
Deep Learning—FastText,
Pre-trained FastText
## Emotion classification

## Fig. 1  Multi-modal Outline

Early fusion
It is the traditional way of fusion, also called a feature or data-level fusion. In early
fusion, data fusing involves concatenating original data or extracted features, i.e., data
transformed into features at the input level before input to a unified single model that
accepts all modalities in one model. Data fusion takes place in many ways, typically
pooling or concatenation. A Unified model can be any machine learning algorithm
such as Random forest, SVM or deep learning model like CNN, RNN, etc. In early
fusion, to accommodate multiple data modalities into a unified model, data transformation into a single feature vector is essential [15]. Proposed early data fusion by
concatenating textual, acoustic and visual features in the multimodal stream for sentiment analysis.

![page5_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page5_img1.jpeg)

Page 6 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Fig. 2  Overview of early and late fusion multimodal system

Late fusion
Late fusion is also termed decision-level fusion as it aggregates predictions or results at
the decision level. In late fusion, a number of algorithmic models are trained, typically
one model per modality type. By combining the decisions from multiple models into a
unified decision, a conclusion can be drawn. An ensemble approach inspires late fusion.
In the ensemble approach, multiple models are trained on the same data, whereas in
late fusion, multiple models are trained on different data. In late fusion, models must
attempt to predict the same outcome. In late fusion, the number of models to be trained
depends on the number of data modalities to be fused. To fuse the decisions from different models, different rules are used, such as the Bayes model, max-fusion, average
fusion, etc. Late fusion is simpler than early fusion as compared from an architecture
point of view. Data from different modalities are handled by multiple models individually and dealt with independently, so errors are also handled by models independently or
uncorrelated, so late fusion gives good performance compared to early fusion [16]. Used
an attention-based late fusion mechanism for task adaptation in natural language understanding. The authors developed a late fusion method capable of aggregating representation from intermediate layers of the pre-trained BERT model to adapt to a downstream
NLU task.

Intermediate fusion
It is also called hybrid or joint fusion. In intermediate fusion, multiple models are trained
but in a stepwise manner. The output of some models with input features from different
modalities is given to subsequent models for outcome. Similar to early fusion, interaction effects between the parameters take place. Models are trained in a stepwise fashion so loss from the second model can be propagated back to the first model, updating
weights for both models. Intermediate fusion is based on deep neural networks, allowing
the most flexible fusion of features at different depths of model training. So, it is built
in a very conscientious manner. Intermediate fusion provides significantly improved

![page6_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page6_img1.jpeg)

Page 7 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

performance compared to early fusion. In a deep learning multimodal context, a fusion
of representations of different modalities into a single hidden layer, the model learns a
joint representation of each modality. Different modalities can be fused simultaneously
into a single shared representation layer or progressively using one or more modalities
at a time [17]. Built a system that uses intermediate fusion to combine text, video, and
audio modalities to detect aggression in surveillance systems.

Unimodal text‑based emotion detection
A significant number of literature and research have been developed about text-based
emotion detection. Emotion detection is an expanded version of sentiment analysis. In
order to determine the sentiments of tweets, a BERT architecture incorporating CNN,
RNN, and Bi-LSTM was assessed on the Twitter dataset [17]). The tweets were converted to vectors using Word2Vec. After evaluating various Word2Vec combinations
with baseline models, BERT-Bi-LSTM showed the best results. [18]) curated multilabel emotion dataset from literature sources annotated for fine-grained categories.
The study included thirty-eight emotion labels, with balanced samples in each category.
The dataset was trained and assessed using the F1-Score using semi- supervised techniques like RoBERTa, BERT, and DistilBERT models. In [19], a hierarchical Bi-LSTM
model with Glove embeddings was evaluated on the Twitter dataset. A Hierarchical Bi-
CuDNNLSTM with NVIDIA CUDA deep neural network library used to detect emotions on emotiondatasetforNLP dataset. the authors also assessed the performance of
Glove-CNN-LSTM, GloVe-Bi-LSTM, GloVe-CNN-Bi- LSTM and GloVe-CNN-Bi-
GRU models. The transfer learning approach was used by [20] to capture the contextual meaning of the text and for a better understanding of the emotions. EmotionBERT
model compared with baseline LSTM and Bi-GRU models using MELD dataset. While
the aforementioned text emotion identification techniques improved performance, they
exclusively used features found in text data. Emotional information is abundant in the
visual content, such as emojis of the user data. So, when designing a precise text-based
emotion detection system, it becomes necessary to incorporate text and emoji information to enhance emotion recognition performance by making it a multimodal system.

Multimodal emotion detection using text and emojis
When it comes to bringing more accuracy to text-based emotion detection tasks in
online text communication, text and emojis work well together. Some researchers
combined text and emoji modalities to improve the accuracy of text-based emotion
detection. In one of the works on depression detection [21], multimodality of text and
emojis was used. Benchmark dataset SentiEmoDD labelled with emotion, sentiment
and depression symptoms used on a Multiview ensemble approach to detect the depression. The authors used BERT vectorization for generating text vectors and Emoji2Vec
for emojis. This work used a late fusion of vectors. In machine learning, SVM linear kernel function, decision trees, Naïve Bayes, and KNN were used with a stacking approach.
Similarly, in deep learning, CNN, RNN, LSTM, GRU models and stacking approaches
were used. SVM outperformed the other individual approaches in the machine learning
approach, but the stacking approach produced the best out- comes in the deep learning
approach. The authors [12] created a Bi-LSTM model that combines emojis and text to

Page 8 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

analyze emotions. the authors used the Sina social net- working site for Chinese microblog reviews. To obtain emoji embeddings, the emoji vectorization method was used
in which the authors applied multidimensional emotion and emotional value to transform emojis into equivalent emoji vectors using word2vec. The text was also converted
to vectors using word2vec. Linear SVM, random forest and LSTM models were evaluated on the dataset using macro precision, recall and F1-Score [12]. Presented a SEER
model to capture the emotions of words and used emoticons to enhance the emotion
detection task. The authors used the Bi-GRU model to capture semantic information,
whereas statistical analysis of emoticons from datasets was used to extract emotional
information from emoticons. Different machine learning and deep learning models,
including SVM, GRU, LSTM and its bi-directional variants and CNN-Bi-LSTM, were
assessed on NLPCC2013 and NLPCC2014 datasets using precision, recall and F1-Score.
The authors [22] curated a large dataset of 58 k English Reddit comments with twentyeight emotion classes. It is an emoji-integrated text dataset annotated using principal
preserved component analysis (PCA). The authors used transfer learning based on the
BERT model and evaluated the F1- score, precision, and recall along with Bi-LSTM as
the baseline model [23]. Presents emojis that could contribute to automating sentiment
analysis. The authors created an Arabic-language Twitter tweet dataset. In order to integrate text and emojis at the feature, score, decision, and hybrid levels, the authors suggested early and late fusion. For text feature extraction, TF-IDF, CBOW and Skip-gram
approaches were employed, whereas emoji frequency, lexicon-based, emoji CBOW and
emoji skip-gram methods were utilized for emoji extraction. These features were classified using SVC and LR and evaluated using precision, recall, F1-score and accuracy [24].
Developed a speech act classification to determine the intent of Twitter. The authors
generated a dataset from the speech act on Twitter and named it EmoTA. The dyadic
attention mechanism was introduced to fuse the modalities in the intra-model and intermodel attention ways. The authors used BERT for text feature extraction and emoji2Vec
representation for emoji feature extraction. Bi-directional LSTM was used in a multi-
tasking framework using a fully shared and shared private model. When this multitask,
framework was evaluated based on accuracy and F1-score, the multimodal results outperformed the unimodal ones.

Dataset
In order to train the predictive models in supervised machine learning algorithms,
labelled datasets are needed. A significant amount of human effort is needed to cresate
annotated or labelled datasets. One of the significant challenges in multimodality is
choosing a multimodal dataset. The authors surveyed three datasets suitable for multimodal analysis with text and emojis. The authors studied GoEmotions [22], EmoContext [25], and TweetEval [26] datasets curated from social media sites such as Twitter or
Reddit. After thoroughly looking at the datasets based on the number of records, number of emotions, and sources from where they were curated, the authors preferred to
go with the GoEmotions. The GoEmotions dataset offers the most comprehensive coverage of expressed emotions and minimum overlap between emotions, so the authors
preferred it. It was built on 27 emotion labels with neutral categories. These 28 emotion
labels from the dataset are easily mapped further to the Ekman model of emotions. This

Page 9 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

collection of 58 K records has been compiled from Reddit comments. It is available for
public use*. Data was gathered from the platform from Reddit’s founding in 2005 until
January 2019.
The dataset contains the records with only text and emojis. Some records include simple text, text with one emoji, and text with more than two emojis. Although data contains two modalities, text and emojis, the dataset is annotated only for text information.
It is observed from Fig. 3 that there is a significant discrepancy in terms of emotion frequencies. For example, adoration is 30 times more frequent than grief. Figure 3 presents
the emotion categories with the number of samples. we have analyzed the total number
of entries in the training, validation, and testing sets that include emojis incorporated
in the textual description. It was observed that around 3.51% of the samples had emojis.

Proposed methodology
This section outlines the overall architecture for the multimodal framework using text
and emojis and implementation details. Figure 4 displays the overall architectural diagram of the suggested system. Our approach is based on extracting features from two
distinct modalities: text and emojis. They are described as follows:

## Unimodal feature extraction

Text modality
We have developed two different systems for two different modalities. Both systems consist of two subparts. First, preprocessing, feature extraction, then combining features
and model classifier to predict the emotion.

1.	 Text Preprocessing—The data must be pre-processed in order to extract and identify relevant features. Preprocessing is necessary since the content from the user is
mainly composed of short text, slang terms, incomplete words, etc. Text pre- processing is crucial in preparing natural language data for effective analysis and model-

Fig. 3  Statistics on the total number of samples with single and multiple labels for each emotion category

![page9_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page9_img1.jpeg)

Page 10 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Fig. 4  Overall architectural diagram of the proposed network

ling. Textual data is pre-processed with data cleaning, normalization and vectorization.

(a)	 Data Cleaning: We have performed the following stages to remove noise and
inconsistencies from the text data, typically including:

•	Lowercasing: Converting all text to lowercase eradicates capitalization-based
variations.
•	 Punctuation and digit removal: Omitting punctuation and numerical characters
improves the focus on linguistic features.
•	 Stop word removal: Eliminating common words with minimal semantic content
(e.g.,”the,””a,” and”is”) reduces noise and dimensionality.
•	 Hashtag and HTML tag removal: Removing non-linguistic elements like
hashtags and HTML tags enhances textual clarity.

(b)	 Tokenization/Segmentation: In this step, the text is divided into distinct chunks
known as tokens, which are usually words or morphemes. As a result, the original
text is represented by a series of tokens that may be further processed and analyzed.
(c)	 Normalization: This step improves consistency and facilitates analysis by reducing
morphological variants of words to their regular forms. Stemming and lemmatization are two typical methods of normalizing. We have applied lemmatization to
identify the grammatical root of a word (lemma) based on its context and linguistics.
## 2.	 Textual Feature Extraction—Then, the second step is to extract the features from
pre-processed textual data. Our approach leverages pre-trained NNLM embeddings
to represent words as vectors within a d-dimensional space. These embeddings,
trained on the GoEmotion corpus, provide a sophisticated comprehension of word

![page10_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page10_img1.jpeg)

Page 11 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

meanings and emotions. The NNLM algorithm [27],) introduced the idea of training word embeddings in a statistical neural network language model (NNLM). This
novel methodology aimed to predict the subsequent word in a sequence by leveraging the previous words, effectively capturing the semantic and syntactic connections among them. NNLM embeddings, context-based representations, are tokenbased text embeddings trained on diverse English Google News 200B corpus. In this,
each word (token) is mapped to a 128-dimension fixed-length vector representing its
meaning and relationships with other words, which means capturing semantic and
syntactic relationships between words, even for out-of-vocabulary (OOV) words.

Extracting textual features from the vocabulary of NNLM as V, with cardinality |V |. It
involves the following steps:

•	 The first step is One-Hot Encoding, where each word from preprocessed input w ∈ V
is mapped to a one-hot vector of dimension |V | and given by equation 1:

(1)
hw = (hw1, hw2, ..., hw|V|)

where:

ℎwi: 1 if and only if i = w and 0 otherwise
These vectors are sparse and computationally expensive due to their high
dimensionality.

•	 The Second step, the embedding layer, is a linear transformation represented by a
matrix W ∈ R(d∗|V |), where d is the desired embedding dimension (here 128). The
word embedding for w ∈ V is then obtained as: ew = W ∗ ℎw ∈ Rd This transformation
projects the high-dimensional one-hot vectors into a lower-dimensional space while
preserving semantic information. In the learning word embedding, the embedding
layer parameters are learned during the NNLM training process. An objective function is used to optimize the prediction of surrounding words based on the current
word embedding.
•	 In the final feature vector construction, word embeddings are combined into sentence embedding using the sqrt combiner.

## Emoji modality

## 1.	 Emoji Processing—We have used emoji, a Python-based library for separating the
visual representation of an emoji, to extract the emoji features from the input text.
There are 1816 distinct kinds of emojis accessible in all.
## 2.	 Emoji Feature Extraction—Then, we employed Emoji2Vec, a pre-trained word
embedding model specifically designed for emojis for emoji representation. This

Page 12 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

model is based on a method designed to generate representations compatible with
word2vec, a popular text-based word embedding framework. In this model, Emojis
are mapped into a d-dimensional vector space according to their semantic similarity, as determined by the Unicode standard definitions of each one. So, embedding
emojis into a vector space enables the model to comprehend emojis not as isolated
symbols but as elements with semantic meanings, reducing the ambiguity to some
extent. If an input text contains nv emojis, we obtain the final emoji vector representation V as a matrix of dimensions nv ∗ 300, where each row represents the vector of
an individual emoji. The final emoji vector representation V is obtained as a matrix
V ∈ (nv ∗dv), capturing the semantic information of all emojis present.

Fusion of modalities
Early fusion or feature fusion approach is applied to combine the features from text and
emojis. This approach merges information from different modalities by directly concatenating their extracted features. This generates an augmented feature vector in a higherdimensional space, encompassing the combined information. The extracted feature
vectors or embeddings of text modality and emoji modality are concatenated. It employs
a simple feature-level fusion approach by concatenating the extracted textual and emoji
features. Mathematically, let Ŵ = (f1, f2, . . . , fd) be the textual feature vector with d
dimensions, represented in Rd where d = 128, E = (e1, e2, . . . , en) be the emoji feature
vector with n dimensions, represented in Rn where n = 300. The combined feature vector
C is then obtained by directly concatenating these two vectors:

f1, f2, ..., fd, e1, e2, ..., en

.

# C =

Model architecture
The combined feature vector C is made to pass through the modality encoder. The proposed multimodal emotion detection using a hybrid convolutional-recurrent model is
shown in Fig. 5. In Fig. 5, red circles show the emoji representations, while green circles
show textual representations. Similarly, the circle shape represents the recurrent component, and the square shape represents the convolutional component in the modality
encoder of the system. The modality encoder is a neural network which predicts the final
class label using the SoftMax function, as it is a multiclass classification detection problem. The Convolutional neural network layers are used for local feature extraction to
identify the patterns that might be indicative of specific emotions and, similarly, recurrent layers to capture long-range dependencies and con- textual information from the
sequence of text with emojis. Given an input text t with emojis, the proposed multimodal
model calculates a probability for each 28-emotion label using a series of layers. These
layers utilize the pre-trained word embeddings, where NNLM pre-trained embeddings

Page 13 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Fig. 5  Detailed architectural diagram of the proposed network

are used to convert each word in the phrase into a high-dimensional vector representation, and Emoji2Vec pre-trained embeddings are used in the case of emojis. Combined
vectors capture semantic and contextual information about the word and emojis, allowing the model to learn essential features from the entire sequence. The authors experimented with the combined architecture of CNN- LSTM and CNN-Bi-LSTM. Similarly,
the authors have used LSTM and bidirectional LSTMs (Bi-LSTMs). We adopted LSTM
to understand the long-term dependencies with context and a Bi-LSTM network to capture the sequence’s forward and backward context. LSTM, CNN-LSTM, BiLSTM, and
CNN-BiLSTM models are proficient at catching sequential and contextual dependencies in data. So, when emojis and text are combined in an early approach, these models
learn combined embeddings, allowing them to comprehend emoji meanings based on
the surrounding textual context. Also, bidirectional models can process input from both
directions, forward and back- ward, aiding the model in contextualizing emojis more
effectively in broader sentences. CNN can capture local patterns and relationships and
inherently shift-invariants. So, combining the characteristics of convolutional features
with sequential modeling will enhance the extraction of local patterns, which may disambiguate emoji meanings to improve understanding of multimodal emotion detection.

1. 	 Convolutional component: In the convolutional-recurrent multimodal model, initially, concatenated features (ci) will be passed through convolutional layers. The onedimensional convolutional layer is used where k filters are applied to the input text t

![page13_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page13_img1.jpeg)

Page 14 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

with emojis. Individually filter is represented as Γ of size s represented as a matrix of
dimension s ∗ d. Each filter slides across the sequence of s consecutive embeddings

(i∶i+s−1). This convolution operation produces a new feature ci for each window and is
expressed by Eq. 2

(2)
ci = f

F ∗e(i:i+s−1) + b

where:
f: Non-linear ReLU function.
b: Bias-vector.
Each filter, in the convolution operation, slides across all possible windows
of features in the sequence. This process generates a feature map, denoted as
cm = [cm1, cm2, . . . , cm(n−s+1)] , where n is the sentence length and s is the filter size. In
the first stage, if k filters are used, this operation creates k distinct feature maps. In this
specific case, the convolutional layer employs 128 filters, resulting in the generation of
128 distinct feature maps. After the convolutional layer, a max pooling layer is applied to
get 128 feature values corresponding to 128 feature maps. Then, these feature values are
provided to the recurrent layer sequentially.

## 2.	  Recurrent component: The convolution layer outputs a series of feature vectors.

cm1, cm2, ..., cmk, which are sequentially sent to the recurrent layer. For each combined
vector (denoted by cmi), LSTM computes only using one hidden state. At the same time,
the Bi-LSTM computes it’s forward and backward hidden states, denoted by ℎi−1 and
ℎi+1, respectively. These are obtained using the following equations Equation 3 and 4:
Forward hidden state:

(3)
hif = LSTMfd(cmi, hi−1)

Backward hidden state:

(4)
hib = LSTMbd(cmi, hi+1)

The final hidden state matrix for the feature vector representation of user input, ℎu,
is then constructed by concatenating the forward and backward hidden states for each
word in the sequence and is expressed by Eq. 5:
Final hidden state matrix:

(5)
Hu = [h1, h2, . . . .hn] ∈R(n ∗2∗dl)

Here, nc is the sequence length, dl is the number of hidden units in each LSTM
layer, and ℎu is the concatenation of ℎif and ℎib. This final matrix, Hu, captures the

Page 15 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

contextualized representation of the entire user input, serving as the input for subsequent processing. At each time step t, the recurrent network takes the input cmt and the
output of the previous step ℎt−1. It then creates the current output ℎt by applying a linear
transformation following a non-linear activation function and expressed by Eqs. 6 and 7.

(6)
ht = tanh(Wcmh ∗cmt + Whh ∗ht−1 + b)

## Why ∗ht + by

(7)
yt = tanh

where:

ℎt: hidden state in time t.
cmt:input feature vector from the convolutional layer in time t.
yt: output in time t.
Wcmℎ:Weight matrix for input to the hidden connection.
Wℎℎ:Weight matrix for hidden-to-hidden connection.
Wℎy:Weight matrix for hidden-to-output connection.
b: Bias vector for the hidden layer.
by: Bias vector for the output layer.
The final classification stage takes the output from the recurrent layer as input. In
simpler terms, the SoftMax function takes the recurrent layer’s output (Eq. 8)(possibly
containing high-dimensional representations) and transforms it into a probability distribution over the N possible output labels (emotions) for a given text input. And output
layer is defined as follows –

(8)
## P(e) = Softmax(yt)

where:
P (e): the probability distribution over emotion classes.

Experiments

Evaluation measures
In order to evaluate proposed multimodal system, we adopted accuracy, weighted precision, weighted recall, and weighted F1 score. Evaluating the performance of emotion detection models presents a challenge as an intrinsic imbalance in emotion class
distributions within most real-world benchmark datasets. Accuracy alone is a biased
measure in imbalanced classification problems. No single metric can definitively capture the model’s effectiveness in such scenarios. Therefore, to gain a more comprehensive understanding of the model’s performance, we have also used a classification report
and confusion matrix to analyze the emotion class-wise performance as it is a multiclass
classification problem. Similarly, accuracy and loss curves against time were used to
show the performance of the proposed system. Distinct colors are utilized to depict various models in curves, while the continuous line signifies the accuracy and loss during
training. On the other hand, dashed lines indicate the accuracy and loss of the validation.

Page 16 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

## Algorithm 1  Algorithm

## Experimental settings

The GoEmotion dataset was divided randomly into two parts of 80% 20% split for train
and test sets, respectively. And again, from the training set, 20% was used as a validation set. The statistics of the training, validation, and test set are shown in Table 3. For
single modality and multimodal classification, LSTM, Bi-LSTM and Hybrid architectures of CNN-LSTM and CNN-Bi-LSTM were adopted. To optimize performance, the

Page 17 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Table 3  Summary of hyperparameters values used in the proposed hybrid model

Hyperparameters
principal values

## Convolutional parameters

Embeddings
NNLM embeddings,
## Emoji2Vec embeddings

Filter size
128

Kernel size
5

Pooling
## Max = 2

Stride
## Default = 1

Padding
Same

Activation function
ReLU

## Recurrent parameters

Hidden units
64

Number of layers
1

Epochs
20

Activation functions
SoftMax

Loss function
## Categorical Cross Entropy

Optimizer
## Adam optimizer

Learning rate
0.001

Batch size
512

parameters were carefully selected through a what-if analysis. In a hybrid architecture,
a one-dimensional convolution layer with 128 filters was used. This convolutional layer
of kernel size five was used with the ReLU function. A max- pooling layer of size two
was used in the subsequent layer. To encode fused modalities, LSTM with 64 hidden
memory cells were used, followed by a flattened layer. So, Bi-LSTM versions resulted in
a 128-dimensional representation vector due to the bidirectional nature of the layers.
A fully connected layer of dimensions 28 was used in the subsequent layer. For model
optimization, the Adam optimizer is employed. The Adam optimizer is used in this specific scenario, with a learning rate of 0.001 and a batch size of 512. Similarly, Sparse Categorical cross-entropy loss was used with a learning rate of 0.001 was used in the final
experimental setting. The tests are performed on a computer that is equipped with a
12th generation Intel Core i7 CPU operating on the Linux subsystem, running at a frequency of 2.10 GHz, with 16 GB of RAM. The Jupyter framework is used for coding and
running Python. The implementation is carried out via the TensorFlow Keras library.

Results and discussions
A series of experiments were conducted to evaluate the proposed multimodal approach.
The proposed approach requires thorough analysis owing to the class imbalance present
in the dataset, whereby the”neutral” class exhibits a much higher number of samples
in comparison to the other classes. Imbalanced datasets might lead to biased predictions in the model, favouring the dominant class and disregarding the minority classes
as unimportant. Therefore, it is essential to deal with this matter. In order to assess the
model’s effectiveness in these circumstances, we use measures like as precision, recall,
and F1-score, which are particularly suitable for datasets with uneven distribution. An
alternative method involves resampling the dataset by either under- sampling or oversampling. Oversampling mitigates the problem of class imbalance by augmenting the

Page 18 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

quantity of data instances belonging to underrepresented classes. It does this by iteratively and randomly sampling data points from the minority class. We have implemented
the Random oversampling approach.
For the baseline models, experiments were conducted in different categories. Experiments were conducted for only text modality, only emoji modality and combined modality (Text + Emoji) in varying combinations of the network architectures. As stated
earlier, the goal of this study is to analyse the role of emojis in emotion detection; thus,
the authors do not put focus on improving or analyzing the performance of emotion
detection and treat it as an auxiliary task aiding the primary task. We show the comparison between our proposed approach and baselines, i.e., Baseline-1 [22], Baseline-2
[28] and baseline3 [29], which also made use of the same dataset. These baseline models
are trained only on the textual features. This study distinguishes itself within existing
research on this dataset by incorporating both textual and emoji features in its proposed
approach. No prior work has explored the combined influence of these modalities on
the task. Results are shown in Table 4. It summarizes the results of only text modality,
only emoji modality and combined Text + Emoji modality with LSTM, Bi-LSTM, CNN-
LSTM, and CNN-Bi-LSTM models in terms of various performance metrics. Accuracy
curves and loss curves are also one of the evaluation measures. Distinct colors are utilized to depict various models in curves, while the continuous line signifies the accuracy
and loss during training. On the other hand, dashed lines indicate the accuracy and loss
of the validation in Fig. 6.
As visible in Table  5, the only textual modality provides the best results compared
to only the emoji modality on the GoEmotions dataset. T and E represent textual features and emoji features, respectively. For only text modality setup, if we compare our
results, there is a performance improvement of 23.0 points ↑ with Baseline-1, 3.0 points
↑ with baseline-2 and 6.4 points ↑ with baseline-3 in the F1-score. Table 5 shows the
comparison of our best results of unimodal and multimodal with SOTA methods. For
Text + Emoji modalities, we observe that multimodal setups outperform both unimodal
modalities. We can observe that performance has remarkably improved with combining
Text + Emoji modality and the highest overall performance given by the CNN-Bi-LSTM
model with training accuracy of 77.80%, validation accuracy of 73.2% and F1-score of
71.68% with a multimodal approach. Figure 7 presents a comparison of models in terms
of different performance measures.
Table 4 also displays the precision, recall, and F1-score metrics for the multimodal
model. An investigation of precision and recall is essential for a thorough evaluation of a
model’s performance, especially when dealing with unbalanced datasets. Precision is the
value obtained by dividing the number of correctly predicted positive outcomes by the
total number of expected positive results. Recall measures the proportion of accurately
identified positive outcomes out of all the actual positive instances in the data. All models have shown the good precision and recall values for combined modalities as compared to unimodal. It means models are good at predicting positive instances of labels
when text and emojis are combined.
A confusion matrix is an effective tool for evaluating the efficiency of a model in multiclass classification tasks. This is a square matrix, which has an equal number of rows and
columns. Each row and column in the matrix represents a distinct class. Each column

Page 19 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

CNN-Bi-LSTM
70.28
65.57
68.57
70.62
69.14
53.70
37.99
45.52
36.43
36.96
77.80
73.29
71.01
72.96
71.68

Bi-LSTM
69.64
61.80
68.37
70.67
68.80
50.37
36.95
42.18
36.95
35.63
77.03
72.15
69.61
71.97
70.08

CNN-LSTM
67.32
62.00
60.83
63.84
61.73
51.93
37.72
46.68
37.72
38.94
72.45
67.70
65.32
67.60
65.96

LSTM
60.82
57.34
53.81
57.57
54.47
43.81
35.91
36.63
35.91
32.31
75.94
72.01
69.91
72.24
70.64

Accuracy
Precision recall F1-score
Accuracy
Precision recall F1-score
Accuracy
## Precision recall F1-score

Model
Only text
Only emoji
## Combined (text + emoji)

Training validation
Training validation
## Training validation

Table 4  Results of proposed multimodal models in terms of accuracy, precision, recall and weighted F1-score

Page 20 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Fig. 6  Accuracy curves and loss curves for proposed multimodal models on the GoEmotions dataset

Table 5  Comparison of SOTA methods with our results on the Empathetic dialogue dataset for
emotion detection

No
Reference
Deep learning method
Performance
metrics

1
Baseline—Text [22]
BERT
46.0

2
Baselines—Text [28]
RoBERTa
66.1

3
Baseline—Text [29]
GNN
62.7

4
Our best results—Text
CNN-Bi-LSTM
69.1

5
Our best results—Text + Emoji
CNN-Bi-LSTM
71.7

Fig. 7  Comparison of models for different performance measures

in the matrix represents the frequency of predictions made by the model for a certain
class label, compared to the actual class label in the data. Optimally, the diagonal elements, which indicate correctly categorized instances, possess greater values, indicating successful predictions. Whereas off-diagonal elements shows misclassifications. In
Fig. 8, all models shown diagonally high values which means correct predictions of the
models in each emotion class. As the authors can figure out from confusion matrix of
CNN-Bi-LSTM model has shown, “grief”, “relief”, “pride”, “nervousness” and “fear” are

![page20_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page20_img1.jpeg)

![page20_img2.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page20_img2.jpeg)

Page 21 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Fig. 8  Presents a confusion matrix for (a) LSTM, (b) Bi-LSTM, (c) CNN-LSTM, and (d) CNN-Bi-LSTM models

the emotions that are better predicted. Similarly, class “neutral” has the less number of
correct classifications. There are some emotion class pairs where we could see misclassifications, like in between “neutral and approval”, “neutral” and disapproval”. The less
accuracy for these emotions might indicates a similarity in the words, phrases and emojis used to convey them.

Ablation study
An ablation study is an effective technique used in artificial intelligence to analyse the
specific impacts of various components of a model on its overall performance. It entails
methodically eliminating or altering components of the model and analyzing the impact
of these modifications on its output. To understand the effect of emoji and the effect of
the convolutional component on the proposed model, we perform an ablation study. The
results are shown in Table 5. For the setups, we observe that the best-proposed model
outperformed w/o emoji and the proposed w/o convolutional component.

![page21_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page21_img1.jpeg)

Page 22 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

## Table 6  Ablation study

Best setup
Accuracy in %
## F1 score in %

Model w/o Text
37.99
36.96

Model w/o Emoji
65.57
69.14

Model w/o convolutional
72.15
70.08

Proposed Model
73.29
71.68

Table 7  Sample text with its corresponding predicted labels

Impact of emojis
Although our main focus is to study the effect of emojis on emotion detection, from
Tables  4, 6 and 7, We observe that emojis are helpful in improving the performance
of emotion detection. Empirically, we have shown that emoji helps emotion detection
(Table 6). We took some examples from the dataset (Table 7) and off the dataset also,
which are sentences that include emojis to show the effect of emojis. Each example
has an implicit/explicit emotion with emojis. We have included examples in both ways
where emojis are supporting text and where emojis are contrasting the text, such as sarcastically or ironically. Table 7 shows the sample text with an emoji with the predicted
label. Table 7 shows the predicted label for examples and whether the proposed model
predicted the label correctly or not by human evaluation. Interestingly, the predictions
made by the model were correct in many sarcastic cases, but in some cases, they were
incorrect. We observe that emoji’s emotion plays an important role in emotion detection. In the example,”I am happy with service!
”, the emotion displayed by the emojis is
negative, whereas the text is positive. This aids the model in understanding the contradistinction between the emotion displayed by the text and the emoji. Thus, it correctly

![page22_img1.jpeg](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page22_img1.jpeg)

Page 23 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

interprets that the example is sarcastic. In the example,”I’m scared to even ask my mom;
I might get yelled at
”, the model fails to comprehend the precise distinction between
the text as the emojis do not provide any further distinguishable information to the
model about it in training.

Real‑world scenarios of proposed system
The proposed model has the potential to be used in real-world scenarios as it com-
bines convolutional and recurrent layers with Emoji2Vec and NNLM (Neural Network
Language Model) embeddings. Combining the strengths of the proposed model with
domain-specific embeddings applies to a wide range of real-world scenarios, including
natural language text and emojis.

•	 Integration into current frameworks—[30]

The proposed system can be integrated into existing systems or current frameworks
where text and emoji-based emotion detection can be used as a supplemen-tary task.
In chatbots or virtual assistants, the proposed model can be used to detect the emotions of input text and to generate responses empathetically. Emotions such as happiness, confusion, and frustration can be easily recognized by combining text and emoji
cues, and more effective communication can be achieved. Similarly, in recommendation
systems, Community Management Tools, and Customer-Centric Marketing Systems,
the proposed model can provide more nuanced emotion detection by understanding the
emotional tone of emojis with text.

•	 Domain-Specific Customization—[31, 32]

Various fields, including healthcare, e-commerce, and social media, have distinct usage
patterns for emojis. In healthcare, emojis are used to denote symptoms or mental state
through emotions. In e-commerce, emojis are used to indicate product characteristics
or feelings related to product or service use. The proposed model can be trained using
domain-specific datasets, allowing predictions to be customized to the distinct needs of
each domain.

•	 Cross-Linguistic and Cultural Adaptability—[33]

Despite their widespread use, emojis may have variable meanings depending on the
languages and cultures. As per cultures, people use different emojis to convey their
thoughts and emotions. E.g., An emoji
, for instance, may mean ”thank you” in one
culture and ”prayer” in another. The proposed model, which utilizes Emoji2Vec embeddings to transform emojis into semantic vector spaces depending on their usage in
natural language, enhances its ability to recognize intended meanings across many cultural contexts. Similarly, Cross-linguistic and cultural adaptability can addressed by
training and fine-tuning the proposed model using multilingual datasets. To ensure it
understands cultural and linguistic fineness in emoji use, the proposed model can be
pre-trained on datasets annotated with different languages and then fine-tuned using
domain-specific data, which can further enhance the interpretation of emojis.

Page 24 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

Conclusion
This study presents a comprehensive review of work related to emoji-based emotion
and sentiment analysis in online social communication. This work investigates the role
of emojis in emotion detection and multimodality in text-based emotion detection. The
authors explored the potential of multimodal text-emoji fusion for text-based emotion
detection on the GoEmotions dataset using baseline deep neural networks. In the proposed hybrid framework of convolutional and recurrent networks, the authors effectively combined textual and emoji features, achieving superior performance compared
to models relying solely on text. It highlights the significant emotional information
embedded within emojis and their cooperative effect with textual cues. Additionally,
the ablation study provided valuable insights into the contribution of each modality,
revealing the importance of both textual and emoji features for optimal performance.
So, the simplicity of the concatenation of two modalities at the feature level makes it a
straightforward to implement and computationally efficient system. Additionally, combining features from various modalities has been empirically shown to improve prediction performance, leading to its widespread adoption within the research com- munity.
Future research directions include investigating the impact of different emoji embedding
techniques such as transformer-based embeddings can offer robust solutions for disambiguating emojis and improving the precision of the system to further enhance emotion
detection accuracy in online social communication. Similarly, the late fusion approach of
modalities could be evaluated to investigate the impact of emojis.

Acknowledgment
This paper has been supported by the RUDN University Scientific Projects Grant System, Project 202235-2-000.

Author contributions
Conceptualization, S.K.,  S.P., and K.K.; Methodology, S.K., S.P., and K.K.; Software, S.K.; Validation, S.K., S.P., and K.K., formal
analysis, S.K., and S.P.; investigation, S.K., S.P., and K.K; resources, S.K. and S.P.; data curation, S.K., and S.P.; writing—original
draft preparation, S.K.; writing—review and editing, S.P.; visualization, S.K.; supervision, K.K.; project administration, S.P.;
funding acquisition, K.K. All authors have read and agreed to the published version of the manuscript.

Funding
Open access funding provided by Symbiosis International (Deemed University). This work was supported by the
Research Support Fund (RSF) of Symbiosis International (Deemed) University, Pune, India.

Data availability
No datasets were generated or analysed during the current study.

Declarations

Ethics approval and consent to participate
Not applicable.

Consent for publication
Not applicable.

Competing interests
The authors declare no competing interests.

## Received: 25 June 2024   Accepted: 9 January 2025

References
	1.
Picard RW. Affective computing. Cambridge: MIT press; 2000.
	2.
Illendula A, Sheth A. Multimodal emotion classification. In: Companion Proceedings of the 2019 World Wide Web
Conference, 2019; pp. 439–449
	3.
Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., Lehmann, S.: Using millions of emoji occurrences to learn any-domain
representations for detecting sentiment, emotion and sarcasm. arXiv preprint. 2017. arXiv:​1708.​00524.

![page24_img1.png](Multimodal%20text-emoji%20fusion%20using%20deep%20neural%20networks%20for%20text-based%20emotion%20detection%20in%20online%20communication_images/page24_img1.png)

Page 25 of 25
Kusal et al. Journal of Big Data           (2025) 12:32

4.
Wijeratne, S., Balasuriya, L., Sheth, A., Doran, D.: Emojinet: Building a machine readable sense inventory for emoji.
In: Social Informatics: 8th International Con- ference, SocInfo 2016, Bellevue, WA, USA, November 11–14, 2016,
Proceedings, Springer. 2016; Part I 8, pp. 527–541.
	5.
Ligthart, A., Catal, C., Tekinerdogan, B.: Systematic reviews in sentiment analysis: a tertiary study. Artificial Intelligence
Review, 2021; 1–57
	6.
Sarker IH. Deep learning: a comprehensive overview on techniques, taxonomy, applications and research directions.
SN Comput Sci. 2021;2(6):420.
	7.
Wang D, Guo X, Tian Y, Liu J, He L, Luo X. Tetfn: a text enhancedtransformer fusion network for multimodal sentiment analysis. Pattern Recogni- tion. 2023;136:109259.
	8.
Huang C, Zhang J, Wu X, Wang Y, Li M, Huang X. Tefna: Text-centered fusion network with crossmodal attention for
multimodal sentiment analysis. Knowl-Based Syst. 2023;269:110502.
	9.
Shixin P, Kai C, Tian T, Jingying C. An autoencoder-based feature level fusion for speech emotion recognition. Digital
Communications and Networks. 2022.
## 10.	 Chauhan DS, Singh GV, Arora A, Ekbal A, Bhattacharyya P. An emoji- aware multitask framework for multimodal
sarcasm detection. Knowl-Based Syst. 2022;257:109924.
## 11.	 Kwon S, Go B-H, Lee J-H. A text-based visual context modulation neural model for multimodal machine translation.
Pattern Recogn Lett. 2020;136:212–8.
## 12.	 Liu, T., Du, Y., Zhou, Q.: Text emotion recognition using gru neural net- work with attention mechanism and
emoticon emotions. In: Proceedings of the 2020 2nd International Conference on Robotics, Intelligent Control and
Artificial Intelligence, 2020; pp. 278–282.
## 13.	 Li X, Zhang J, Du Y, Zhu J, Fan Y, Chen X. A novel deep learning-based sentiment analysis method enhanced with
emojis in microblog social networks. Enterp Inf Syst. 2022;17:1–22.
## 14.	 Constantinou O. Multimodal discourse analysis: media, modes and technologies. J Socioling. 2005;9(4):602.
	15.	 Dutta S, Ganapathy S. Hcam–hierarchical cross attention model for multi- modal emotion recognition. arXiv preprint. 2023. arXiv:​2304.​06910.
## 16.	 Poria S, Cambria E, Gelbukh A. Deep convolutional neural network textual features and multiple kernel learning for
utterance-level multimodal sentiment analysis. In: Proceedings of the 2015 Conference on Empirical Methods in
Natural Language Processing, 2015; pp. 2539–2544.
	17.	 Cao J, Prakash CS, Hamza W. Attention fusion: a light yet efficient late fusion mechanism for task adaptation in nlu.
In: Findings of the association for computational Linguistics: NAACL 2022, 2022; pp. 857–866
## 18.	 Bello A, Ng S-C, Leung M-F. A bert framework to sentiment analysis of tweets. Sensors. 2023;23(1):506.
## 19.	 Rei L, Mladenić D. Detecting fine-grained emotions in literature. Appl Sci. 2023;13(13):7502.
## 20.	 Mahto D, Yadav SC. Emotion prediction for textual data using glove based hebi-cudnnlstm model. Multimed Tools
Appl. 2023;83:1–26.
	21.	 Lee SJ, Lim J, Paas L, Ahn HS. Transformer transfer learning emotion detection model: synchronizing socially agreed
and self-reported emotions in big data. Neural Comput Appl. 2023;35(15):10945–56.
## 22.	 Gupta S, Singh A, Ranjan J. Multimodal, multiview and multitasking depression detection framework endorsed with
auxiliary sentiment polarity and emotion detection. Int J Syst Assur Eng Manag. 2023;14(Suppl 1):337–52.
## 23.	 Demszky D, Movshovitz-Attias D, Ko J, Cowen A, Nemade G, Ravi S. Goemotions: a dataset of fine-grained emotions.
arXiv preprint. 2020. arXiv:​2005.​00547.
## 24.	 Al-Azani S, El-Alfy E-SM. Early and late fusion of emojis and text to enhance opinion mining. IEEE Access.
2021;9:121031–45.
	25.	 Ameer I, Bölücü N, Sidorov G, Can B. Emotion classification in texts over graph neural networks: semantic representation is better than syntactic. IEEE Access. 2023. https://​doi.​org/​10.​1109/​ACCESS.​2023.​32815​44.
## 26.	 Chatterjee A, Narahari KN, Joshi M, Agrawal P. SemEval-2019 task 3: EmoContext contextual emotion detection in
text. In: May J, Shutova E, Herbelot A, Zhu X, Apidianaki M, Mohammad SM (eds). Proceedings of the 13th International Workshop on Semantic Evaluation, pp. 39–48. Asso- ciation for Computational Linguistics, Minneapolis,
Minnesota, USA. 2019. https://​doi.​org/​10.​18653/​v1/​S19-​2005 .
## 27.	 Saha, T., Upadhyaya, A., Saha, S., Bhattacharyya, P.: Towards sentiment and emotion aided multi-modal speech act
classification in twitter. In: Proceed- ings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: human language technologies, 2021; pp. 5727–5737
## 28.	 Bengio Y, Ducharme R, Vincent P. A neural probabilistic language model. Advances in neural information processing
systems, 2000; 13
## 29.	 Ngo A, Candri A, Ferdinan T, Kocoń J, Korczynski W. Studemo: A non- aggregated review dataset for personalized
emotion recognition. In: Proceedings of the 1st Workshop on Perspectivist Approaches to NLP@ LREC2022, 2022.
pp. 46–55.
	30.	 Kusal S, Patil S, Kotecha K, Aluvalu R, Varadarajan V. Ai based emo- tion detection for textual big data: techniques
and contribution. Big Data Cognit Comput. 2021;5(3):43.
## 31.	 Kusal S, Patil S, Choudrie J, Kotecha K, Vora D, Pappas I. A systematic review of applications of natural language
processing and future challenges with special emphasis in text-based emotion detection. Artif Intell Rev.
2023;56(12):15129–215.
	32.	 Acheampong FA, Wenyu C, Nunoo-Mensah H. Text-based emotion detec- tion: advances, challenges, and opportunities. Eng Rep. 2020;2(7):12189.
	33.	 Miah MSU, Kabir MM, Sarwar TB, Safran M, Alfarhood S, Mridha M. A multimodal approach to cross-lingual sentiment analysis with ensemble of transformer and llm. Sci Rep. 2024;14(1):9603.

Publisher’s Note
Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
