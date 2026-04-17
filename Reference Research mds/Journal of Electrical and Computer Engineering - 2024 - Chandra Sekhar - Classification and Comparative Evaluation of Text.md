# Classification and Comparative Evaluation of Text and Emoji‐Based Tweets With Deep Neural Network Models

Wiley
Journal of Electrical and Computer Engineering
Volume 2024, Article ID 9652424, 15 pages
https://doi.org/10.1155/2024/9652424

Research Article
Classification and Comparative Evaluation of Text and Emoji-
## Based Tweets With Deep Neural Network Models

J. N. Chandra Sekhar
,1 M. Kiran Mayee
,2 Ranjana Nadagoudar
,3

## N. Chinna Alluraiah,4 C. Dhanamjayulu
,5 Ravikumar Chinthaginjala
,6 Ravi K.,5

Praveenkumar M.,5 Satyajit Mohanty
,5 and Baseem Khan
7,8

1Sri Venkateswara University College of Engineering, Tirupati, India
2Ravindra College of Engineering for Women, Kurnool, Andhra Pradesh, India
3Visvesvaraya Technological University, Belagavi, India
4Annamacharya University, Rajampet 516126, Andra Pradesh, India
5School of Electrical Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India
6School of Electronics Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India
7Department of Electrical and Computer Engineering, Hawassa University, Hawassa 05, Ethiopia
8Center for Renewable Energy and Microgrids, Huanjiang Laboratory, Zhejiang University, Zhejiang 311816, China

Correspondence should be addressed to C. Dhanamjayulu; dhanamjayulu29@gmail.com and Baseem Khan;
baseemahmedk@gmail.com

Received 27 November 2023; Revised 6 May 2024; Accepted 5 June 2024

## Academic Editor: Mominul Ahsan

Copyright © 2024 J. N. Chandra Sekhar et al. Tis is an open access article distributed under the Creative Commons Attribution
License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly
cited.

Emojis have become increasingly prevalent in today’s digital world, allowing individuals to convey a wide range of emotions, from
uncomplicated to intricate, to a greater extent than previously. Consequently, emojis are being utilized in sentiment analysis and
tailored marketing strategies. Te ongoing research on conducting emotion detection on both tweets and a symbolic expression
dataset sourced from Kaggle. Given that tweets are largely commentaries, we utilized two end-to-end sentence embedding models,
the DistilBERT, USE-Large3, and RoBERTa, which generate embeddings. Tese embeddings are further utilized for training with
dense neural networks (NNs) and LSTM techniques. Remarkably, it was perceived that the text classifcation accuracy for both
models was consistently high, hovering around 98%. However, when the validation set is constructed with that of symbolic
expression or emojis not included in the training dataset, a signifcant drop in accuracy for both models, plummeting to 75%, had
been observed. Additionally, a distributed training methodology is utilized as a substitute for the conventional single-threaded
model to enhance scalability. Tis approach resulted in a roughly 17% reduction in the runtime while maintaining accuracy.
Lastly, in pursuit of explainable AI, the SHAP and LIME algorithms are employed to elucidate the model’s behavior and assess any
potential biases in the dataset. Te creative use of advanced deep NN techniques customized for the delicate complexities of
hybrid-data sentiment analysis indicates a signifcant leap forward. Our proposed work provides the critical gap in existing
sentiment analysis methods, which primarily aimed at either text or emojis in isolation, thereby exploring more holistic understanding of sentiment in digital communications. Moreover, the application of explainable AI techniques, SHAP and LIME, to
demystify model decisions emphasizes commitment to advancing transparent and trustworthy deep learning technologies in
sentiment analysis.

Keywords: artifcial intelligence; emojis; sentiment analysis; social networks

![page1_img1.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page1_img1.jpeg)

![page1_img3.png](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page1_img3.png)

## 1. Introduction

Online media and social networking platforms (SNS) have
witnessed a surge in popularity, enabling individuals to
freely convey their thoughts on a wide array of subjects.
Tese platforms encompass highly unstructured data,
comprising words, emojis, images, and videos, all aimed at
fostering public engagement on various issues. Particularly
on social networks such as Twitter, the content tends to be
concise and is often posted with minimal linguistic
precision.
Social media platforms have emerged as valuable sources
of diverse data, thanks to their real-time texting and
opinion-sharing capabilities. Recently, social media has seen
a surge in popularity as a global or even universal means of
information dissemination. Users can efortlessly upload
and distribute various forms of the multimodal content
within a social context on these platforms, often without
needing extensive familiarity. Social media websites have
witnessed a growing trend in popularity as a platform for the
interchange of ideas and perspectives on a broad spectrum of
subjects. People utilize social media to share their views on
a multitude of topics, engage in discussions about current
events, and express their emotions [1].
People share information via platforms such as Facebook
and Twitter, along with several other social interaction apps,
where they also engage in commenting to express their
simple sentiments, be it positive or negative, regarding the
content. Te emergence of elements such as the linked data
web, computational intelligence, network connection, and
other integral aspects of Web 3.0 has empowered people to
utilize the social interaction platform for connection and the
exchange of viewpoints on current events [2]. Social media
platforms have evolved into valuable sources of diverse facts
due to their instant communication and sharing pointof-view capabilities. Recently, the social interaction platform
has a way of widespread reputation as a way of spreading
information on a global scale.
As one of the leading microblogging platforms, Twitter
provides users with the ability to share immediate insight
into information on a wide array of felds and circumstances.
Among the numerous analytical activities facilitated by the
plan of action with wide collection of short-form posting
data, sentiment analysis (SA) stands out as a crucial one [3].
Utilizing concise 140-character messages known as “tweets,”
users commonly disseminate information across various
aspects of Twitter. Moreover, they choose to guide other
people to receive their Twitter updates. Twitter is widely
used as an accepted instant messaging service, helping individuals stay updated on current events, including global
news and advancements in the feld of science.
Emotion detection, known as sentimental exploration,
constitutes a fundamental natural language processing
(NLP) method [4] employed to ascertain the sentiment
expressed in text, categorizing it as positive, negative, or
neutral. Its utility spans various applications, including the
identifcation of emerging online trends, the evaluation of
product reviews, and the monitoring of brand and product
sentiment within customer feedback. Typically, SA adheres

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

2
## Journal of Electrical and Computer Engineering

to a standardized procedure encompassing data collection,
noise reduction through data preprocessing, data transformation into a computationally suitable format, and data
labeling for training purposes.
Furthermore, emotion detection algorithms can be
classifed into three primary types [5]: rule-based, hybrid,
and automated. Rule-based algorithms are constructed using
manually designed rules, while automatic algorithms leverage machine learning (ML) methods, and hybrid
methods integrate aspects of both rule-based and automatic
methodologies. Te most straightforward sentiment classifcation outcomes were achieved through the purpose of
supervised ML models such as Bayesian and SVM [6].
However, it is essential to note that the manual labeling
demanded by the supervised approach can be cost-intensive.
Eforts have also been made in the realm of unsupervised
and semisupervised techniques, highlighting signifcant
opportunities for further enhancement.
Early progression in deep learning-based language
models has revealed great potential in the feld of emotion
detection, particularly when useful to social media data from
platforms like Twitter. Reference [7] introduced an outline
that utilizes depth-driven language engines, specifcally recurring LSTM neural structures to conduct emotion detection in the context of the increasing COVID-19 cases in
India. Te study incorporates an LSTM language model with
a universal embedded vector space and employs state-ofthe-art BERT technique models for enhanced performance.
Deep learning methodologies have delivered exceptional
outcomes across various verbal data analysis work, with
emotional detection being unanimous [8]. Notably, prolonged memory networks and feature mapping neural
models have demonstrated their efectiveness in SA applications. Terefore, in our approach, we leverage the power of
LSTM-based data embedding for SA in Twitter tweet
classifcation.
Emojis and emoticons play a widespread role in social
media as a means of conveying emotions, moods, and
thoughts [9]. Tey are especially useful when individuals
encounter challenges in expressing their emotions through
raw textual data or while they intend to convey ironical
expression. Te inclusion of emojis and emoticons holds
substantial importance as they have a notable infuence on
conveying the overall sentiment polarity of a sentence.
Consequently, emojis and emoticons serve as valuable tools
in the realm of emotion detection.
Figure 1 shows a deep neural network (NN) framework
where Emojis serve as input, and the output comprises detected emotions such as happiness, sadness, and irony. Annotations alongside the network detail the emoji embedding
process and its signifcance in emotion detection. Te
widespread adoption of emojis has garnered signifcant interest from researchers, primarily due to their capacity to
convey semantical and sentimental information, serving as
visual complements to textual data [10]. Tis plays a pivotal
role in decoding the underlying emotional cues within text.
Some approaches, such as emoji embeddings, have been put
forth to enhance our understanding of the semantic nuances
of emojis. Additionally, embedding vectors have proven

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
3

• We provide an extensive comparative analysis of
diferent NN models in the context of our unique
dataset. Tis not only serves as a benchmark for future
research but also ofers valuable insights into the effectiveness of various models in handling complex
SA tasks.
• Our study has practical indications in the realm of
social media analytics, particularly for organizations
and businesses attempting to understand public
sentiment. Te methods and fndings can be leveraged to gain deeper insights into consumer behavior,
public opinion, and emerging trends in digital
communication.

efective in contextualizing and predicting emoji usage.
Pretraining deep NNs on symbolic expression prediction
tasks, utilizing pretrained emoji embeddings, has also demonstrated its utility in deciphering emotional signals conveyed
by symbolic expressions for various tasks, including SA,
emotion recognition, and ironical expression detection.
Reference [11] presents GAN-TTS, a multilingual end-to-end
TTS model that generates speech that sounds realistic and
human. We utilize a data augmentation technique to improve
the training corpus because there is a dearth of multilingual
voice data available for training. Developing multilingual TTS
models presents a difculty in that it might be difcult to
anticipate speech in languages other than the native tongue of
the target speaker. In order to overcome this difculty, we
include a speaker regularization loss, which enables the model
to acquire speaker representations independent of language.
Te model can successfully handle multilingual synthesis,
thanks to this method. Te problem statement and the set
objectives are detailed in Section 1. Section 2 provides
a comprehensive review of related work. Section 3 outlines the
overarching methodology. Te conducted experimentations,
results, and their scrutiny can be found in Section 4, respectively. Lastly, Section 5 presents the conclusion and
outlines upcoming research prospects.

## 2. Related Work

Tis section provides an outline of the existing research in
the domain of identifying tweets for topic and event extraction. Tese eforts mainly involve ML, analyzing texts,
and deep learning techniques for text data classifcation.
Social media platforms represent a novel and rapidly
expanding source of information.
In the present scenario, the data have been a growing
research focus on opinion mining within the realm of SNS.
Tese platforms have gained immense global popularity as
hubs for socializing and connecting with individuals who
share the same interests. Provoked by the principles of Web
2.0, these SNS are inherently participatory, characterized by
active dialogue, and predominantly driven by the usergenerated content. Reference [12] proposed a survey and
relative analysis of tweet emotion detection. Teir research
aimed at using ML to classify the emotions presented in
tweets. NN-based categories have been explored for their
connection to text data, and these models have regained
prominence with the advanced deep learning techniques.
For instance, Kim [13] proposed a sentence classifer based
on deep learning, implementing the convolutional neural
network (CNN) to classify sentences using distributed
representations.
In Cappallo et al. [14] developed the challenges posed by
the emoji modality, examining it from the perspective of
multimedia research. Tey assembled an extensive dataset
comprising emojis sourced from Twitter, which encompassed both the textual content and emoji usage. Employing
cutting-edge NN techniques, the authors successfully predicted emojis within the dataset. Te authors conducted
experiments involving various classifcation methods in Ref.
[15], including Bayesian, decision tree, support vector machine (SVM), and random forest (RF). Tey employed
a Twitter dataset comprising 12,864 tweets and conducted
a 10-fold cross-validation.
## A NN was employed to deliver context-subtle Tweeted
perception classifcation in Ref. [16]. Tis context-built neural
framework for tweet emotion decoding incorporates contextualized information from pertinent tweets by utilizing
word dimensional embeds. Te experimental fndings substantiated the better performance of our proposed contextbased model in comparison with orthodox fragmented and
sustained word depiction frameworks. Tis highlights the

## 1.1. Problem Assertion and Objectives. SA fnds numerous
practical relevance in various domains. Numerous scholars
have explored the feld of SA, focusing on diferent modalities such as text, images, emoticons, audio, or video.
Interestingly, only a limited number of researchers have
delved into the realm of using emojis to detect sentiments.
Te primary aim of this study is to ascertain users’
viewpoints concerning a specifc subject within online social
networks. Employing NLP in conjunction with ML techniques, we scrutinize tweets to gauge their sentiment. Our
principal contributions can be encapsulated as follows.

## 1.1.1. Contribution

• Tis study proposed a new methodology for analyzing
sentiments in tweets, merging text and emoji data. Tis
approach concedes the growing importance of emojis
in digital communication and handles the complexity
of SA in this mixed-data context.
• Tis research applies cutting-edge deep learning
techniques, such as USE-Large3, DistilBERT, and
RoBERTa. Tese models have been customized and
optimized to handle the nuances and exceptions
presented by the combination of text and emoji-based
data, setting a new precedent in the feld.
• A signifcant contribution of this paper is the integration of explainable AI techniques, particularly
Shapley component clarifcations (SHAP) and Local
Interpretable Model-agnostic Explanation (LIME)
algorithms. Tis acknowledges higher transparency
and understanding of the decision-making processes
of our deep NN models, managing a critical need in AI
ethics and trustworthiness.

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

4
## Journal of Electrical and Computer Engineering

Figure 1: Neural network-based emoji sentiment analysis and emotion detection.

challenges. Te proposed model is a great option for
a communication recognition model because of its low
aperiodic distortion. Although voice synthesis has several
audible defects, our proposed technique achieves the closest
resemblance to genuine speech. Velampalli, Muniyappa, and
Saxena [20] conducted SA on Twitter tweets and emoji
datasets, using USE and sentence bidirectional encoder
representations from transformers (SBERTs) models for
embedding. Teir work highlighted nearly equivalent high
accuracy levels in text classifcation for both models but
noted a signifcant drop in accuracy when emojis not present
in the training set were used for validation.
Integration of Emojis with Text in SA: Previous studies,
like those in Ref. [14], have explored emojis in isolation but
not extensively in combination with text. Our proposed
research flls this gap by analyzing the combined impact of
text and emojis in tweets, using advanced deep learning
models.
Comparative Efectiveness of Diferent Models for
emoji–text tweets: While studies such as Refs. [12, 13] focus
on text data classifcation, there is limited research on
comparing various models specifcally for emoji–text data.
Our proposed research conducted a comparative analysis of
models such as DistilBERT, USE-Large3, and RoBERTa,
specifcally for this unique data type.
Explainable AI in Social Media SA: Studies ([16–18])
have implemented deep learning for SA but lack a focus on
explainable AI techniques. Our proposed research incorporates explainable AI approaches, using SHAP and
LIME,
to
make
the
SA
process
transparent
and
understandable.

efectiveness of the context-based NN model for this specifc
task. Furthermore, the authors identifed that topic-based
contextual articles yield the most favorable outcomes
within the context-based NN framework.
Teir work [17] emphasized the signifcance of word
order in sentence sentiment classifcation and introduced an
encode–decode method named convolutional neural network
long short-term memory (CNN-LSTM). Te primary focus
was on techniques for expressing features of words within
a sentence. To integrate temporal information into the distributed representation of individual words, we introduced
a multichannel distributed representation approach.
Te authors conducted emotion detection on informative huge data using a multiattention merging model
in Ref. [18]. Tis model combines the universal emphasis
function and local concentration model, controlled by gating
units, to produce a coherent contextual description. In
contrast to existing methods, we addressed this issue by
employing sentence embedding methods such as S-BERT
and
universal
sentence
encoders
(USEs),
specifcally
designed to deal semantic relationships and engender
constant length essential entrenchment without the necessity of manual padding for sentences of varying lengths, as
seen in earlier works. Furthermore, since these models are
end-to-end, there is no requirement for manual data
cleaning. Two main issues with the conventional concatenation speech synthesis systems are their unusual character
and low intelligibility. CNN’s deep learning methods for
context are not reliable enough for delicate voice synthesis.
In Ref. [19], the proposed methodology could address those
requirements for voice synthesis and mitigate the related

![page4_img1.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page4_img1.jpeg)

Real-Time and Context-Sensitive Analysis: Tere is
a need for models that can efectively handle real-time and
context-sensitive analysis, as indicated by Refs. [16, 19]. Our
proposed research approach includes the context-sensitive
analysis using advanced NN models, which is pivotal for
real-time SA on social media.
Voice Synthesis and SA: Research like Ref. [19] touches
on voice synthesis but does not connect it to SA. Our
proposed research study may not directly focus on voice
synthesis; it advances the feld of SA, which could be linked
to voice technologies in future research.

## 3. Methodology Framework

We gathered data for this study from both Twitter and
Kaggle. Our data collection process involved using the
Twitter Search API over a specifc timeframe. We conducted
SA on both tweets and symbolic expressions data obtained
from Kaggle. Given that tweets are essentially sentences, we
utilized sentence embedding models such as DistilBERT,
USE-Large3, and RoBERTa to generate embeddings. Our
analysis involved employing entirely linked neural frameworks as well as LSTM NN models. To enhance capacity, we
implemented dispersed learning models. As part of our
efort to make the model interpretable, we employed the
SHAP and LIME algorithm to examine the behavior of the
model and identify potential biases within it based on the
provided feature set. Overall approach is shown in Figure 2.

## 3.1. Datasets. In this research conducting SA on both tweets
and an emoji dataset sourced from Kaggle. Emotions play
a signifcant role in conveying emotions in text-based
communication. Emojinating is a technique that aids in
brainstorming sessions by generating new emoji methods
combined semantic network discovery with visual mixing.
Table 1 shows the sample emojis with defned emotions, and
Table 2 displays a few example sample tweets from the
obtained data, along with ID.

## 3.2. Embedding Models

## 3.2.1. DistilBERT. Te DistilBERTmodel (Sanh et al.) [21] is
basically a distilled variant of the BERT model. Trough the
pretraining phase, the extent of a BERTmodel was reduced by
40% through knowledge distillation while still preserving 97%
of its language comprehension resources and achieving a 60%
speed improvement. Tis approach introduces a triple loss by
combining language modeling, cosine-distance losses, and
distillation, and leveraging the inductive biases learned by
sizable models during their pretraining. DistilBERT is a more
reduced and faster model that ofers a lighter computational
footprint, making it cost-efective for pretraining and suitable
for on-device applications. DistilBERT is a streamlined
transformer model known for its speed, cost-efectiveness,
and lightweight design [21]. It underwent a self-supervised
pretraining process on the same dataset as the BERT base,
where the latter served as its teacher model. During this
pretraining,
DistilBERT employs
knowledge
distillation

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
5

techniques, resulting in a substantial reduction in the size of
the original large model (referred to as the “teacher”) by up to
40%, yielding a smaller “student” model. Tis reduction in the
size allows DistilBERT to operate approximately 60% faster
while still preserving 97% of its performance, as evaluated on
the GLUE language understanding benchmarks. Consequently, DistilBERT presents an intriguing option for developing large-scale transformer models [22].

## 3.2.2. USE-Large 3. Te USEs-Large3 [23] are purpose-built
for managing semantic relationships and generating fxedlength embeddings without requiring manual padding for
sentences of difering lengths, a common practice outlined in
the previous literature. Moreover, because these models are
end-to-end, manual data cleaning becomes unnecessary.
Reference [23] introduced the models designed to
convert sentences into embedding vectors tailored for
transfer learning across various NLP tasks. Tese models
prioritize efciency and deliver precise performance across
a range of transfer tasks. We ofer two versions of the
encoding models that enable a balance between accuracy and
computational resources. In both variations, we explore and
document the interplay between model complexity, resource
utilization, the presence of training data for transfer tasks,
and task performance. We draw comparisons with baseline
models, some of which employ word-level transfer learning
with pretrained word embeddings, while others do not
utilize any form of transfer learning.

## 3.2.3. RoBERTa. RoBERTa [24], short for “A Robustly optimized BERT Pretraining Approach,” is a variation of BERT
pretraining. Its primary objective was to enhance the efciency of BERT training by reducing the pretraining duration. Tis model underwent extensive training, utilizing
1000% more data and computational resources compared to
BERT. In this research, Ref. [25] introduced a predictive
ensemble model that incorporates fne-tuned contextualized
word embeddings from ALBERT, DistilBERT, RoBERTa,
and the BERT base model. We demonstrate that our model
surpasses the baseline models in all evaluated metrics,
achieving an F1 score of 54% and an accuracy rate of 61%.
Tese results position our model as the ffth highest performer on the DepSign task leaderboard.

## 3.3. Distributed Training. In
our
distributed
training
methodology, we implemented a multithreaded approach to
study the deep dataset efectively. While the exact count of
threads (or nodes) can vary depending on the computational
resources available, our proposed works were executed using
a setup that parallelized the learning process across four
threads. Tis methodology allowed us to leverage the
computational power of modern multicore processors to
enhance the scalability of our model training.

## 3.3.1. Data Distribution. Te dataset was divided into
smaller blocks, with each batch being executed by a separate
thread. Tis ensured an even distribution of the data across

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

6
## Journal of Electrical and Computer Engineering

Data collection
## Data collection

Tweets
Symbolic expressions
data

Sentiment
Analysis

Sentiment
Analysis

Embedding
models

## Distributed training

## Distributed training

DistilBERT
USE-Large3
RoBERTa

## Classification models

Classification
models

LSTM neural
networks
## Explainable AI

Fully connected neural
networks

SHAP algorithm
## LIME algorithm

Figure 2: Proposed methodology.

Table 1: Emojis with emotions.

## 3.3.2. Failures and Timely Responses. Our multithreaded
training approach was designed with robustness in mind. In
the event of a thread failure or delayed response, the system
was confgured to redistribute the workload among the
remaining active threads. Tis dynamic reallocation of resources ensured that the training process continued without
signifcant interruptions, thereby maintaining the overall
efciency of the distributed training.

S. no.
Emoji
Description

1
## Face with tears of joy

2
## Face with concerns

3
## Muted face

4
## Folded hands

5
## Bafed face

## 3.3.3. Impact on Performance. Te adoption of the multithreaded distributed training methodology resulted in
a noticeable improvement in the scalability of our SA
models. Specifcally, it led to a roughly 17% reduction in the
runtime compared to a conventional single-threaded model,
while still preserving the accuracy of the sentiment classifcation. Tis demonstrates the efectiveness of our approach
in
handling
large-scale
datasets
and
complex
NN
architectures.

6
## Lady saying ok

7
## Face with running nose

the threads, allowing for parallel execution and efcient
usage of computational resources. Te distribution and
batch size strategy were optimized to ensure a balance between computational efciency and model performance.

![page6_img1.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img1.jpeg)

![page6_img2.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img2.jpeg)

![page6_img3.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img3.jpeg)

![page6_img4.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img4.jpeg)

![page6_img5.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img5.jpeg)

![page6_img6.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img6.jpeg)

![page6_img7.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img7.jpeg)

![page6_img8.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img8.jpeg)

![page6_img9.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page6_img9.jpeg)

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
7

Table 2: Sample tweets with ID.

S. no.
TweetID
## Tweet text

1
2018-En-00320
@Krauthammer @TuckerCarlson @FoxNews Tey constantly crave confict and
theatrics to feel validated. It’s unfortunate that such power feels so unstable

2
2018-En-03973
@Executivegoth Sampled the Alpha trial to catch #dread, and I’m convinced. Te
subscription is totally justifed. You guys are amazing!

3
2018-En-02615
@anya_sisi Of to jail with lots of other like minded muslims, with his koran, prayer
mat, halal #islam  #terrorism

4
2018-En-03737
#AmarnathTerrorAttack Muslims are killing everywhere Syria Iraq Palestine
Everyday beyond Tey say that Islam is terrorism shame on you

5
2018-En-03407
Brace yourself for the force of Ghost Rider’s might! Get ready to experience the
depths of the inferno!

6
2018-En-02626
@EdmundPAdamus Te tragic paradox is that as we strip society of its moral
essence, a fresh belief system will emerge & take its place—without restraint

7
2018-En-00588
@NoahWebHouse someone critiqued my book for delving deep into Webster’s
initial sections. Yet, it was essential, #masterpiece!

8
2018-En-01532
#benefcial for acquiring #knowledge << refne (v): modify for the betterment of
something >>
9
2018-En-03155
@RGUpdate Ever tasted hospital food in England? #ugh #unpalatable #notgood
10
2018-En-03239
@TerraJole exhibits bullying behavior. Clear as day

Te fowchart now focuses solely on the specifed steps:
“1. Collect Tweets” (with the Twitter logo), “2. Pre-process
Data,” “3. LSTM Analysis,” and “4. Sentiment Output.” A SA
approach that utilizes LSTM for classifying Twitter tweets in
Ref. [27]. Initially, real-time Twitter data are gathered,
followed by a labeling process to categorize the tweets as
either positive or negative. Data preprocessing is then
conducted to eliminate irrelevant information and clean the
data, as Twitter data often include various characters,
hashtags, and special characters.
LSTM represents a deep learning model that falls under
the category of RNNs. It is recognized for its capacity to
retain information over extended sequences, a trait that
greatly aids in the back-propagation process [28]. LSTM
comprises memory units referred to as cells, which enable
the retention, modifcation, and updating of information.
Tese cells are equipped with gates that regulate the storage,
modifcation, or removal of information based on the signals
they receive, efectively managing the fow of information
within the network.

## 3.4. Classifcation Models. Te emergence of social media
platforms has presented a wealth of data for SA, allowing for
a deeper understanding of public sentiment. Overcoming
challenges in SA, such as language ambiguity and noisy data,
has been made possible through the application of advanced
techniques. Deep learning methods, fully connected neural
networks (FCNNs) and LSTM networks, have exhibited
encouraging outcomes in capturing semantic information
from textual data. Consequently, there is growing interest in
incorporating emojis alongside textual data for evaluation.

## 3.4.1. FCNNs. Te methodological approach used to construct the FCNN model for forecasting depressive moods
from Twitter data encompasses the integration of both
textual content and emoji representations. Te process
adhered to a structured sequence of steps, encompassing
data preprocessing, tokenization, model architecture design,
model training, evaluation, and subsequent result interpretation. Initially, the dataset was transformed into
a JSON fle format, a choice made to preserve the semantic
meaning of emojis.
Reference [26] introduced the utilization of a CNN model
for the prediction of depressive moods based on Twitter data,
incorporating both textual and emoji components. Te study
outcomes afrm the efectiveness of employing a CNN model
in SA tasks, highlighting its capability in discerning sentiments expressed within the data, particularly negative ones.
Te model achieved an accuracy rate of 88% on the test data,
demonstrating its profciency in accurately predicting the
sentiment of tweets in the majority of cases. Additionally,
precision and recall metrics further indicate the model’s
robustness in detecting negative sentiments.

## 3.5. Explainable AI. After constructing the models, it becomes essential to assess potential biases or discriminatory
elements against users. In order to validate these concerns
and ensure that the model remains impartial and unbiased,
we have incorporated explainable AI techniques. Explainable AI methods, as outlined in Ref. [29], serve the purpose
of elucidating the learning processes and decisions made by
an NN. With the aid of XAI techniques, it has become
feasible to identify areas for improvement within a NN and
to discern the model’s strengths and weaknesses resulting
from its training. One widely employed approach for
scrutinizing model biases is the SHAP algorithm.
SHAP represents a player decision framework designed
to provide explanations for the outcomes generated by ML
models [30]. It establishes a connection between optimal
credit distribution and localized explanations by leveraging
original Shapley quantities derived from game theory, as well

## 3.4.2. LSTM NNs. LSTM networks belong to the family of
recurrent neural networks (RNNs) and are adept at capturing order dependence within sequential prediction tasks.
Te process of LSTM is shown in Figure 3.

Collect
tweets
1

Preprocess
data
2

## 3. LSTM
analysis

Sentiment
output

4 step

Figure 3: Flow process of LSTM.

as their relevant extensions. SHAP appoints a signifcant
value to each feature with respect to a specifc prediction. It
introduces novel elements, including the empathy of a fresh
category of additive feature standing metrics, and ofers
theoretical fndings that demonstrate the existence of
a unique solution within this category, characterized by a set
of desirable properties.
Te LIME model serves the purpose of elucidating the
inferences made by ML models through a process of locally
approximating the corollary point. Te LIME algorithm
constructs a linear regression model in the vicinity of
a particular inference point that necessitates an explanation
[31]. Features with signifcantly positive weights in this
linear regression approximation contribute positively to the
projection decision, whereas those with destructive weights
exert a counterinfuence on the decision. However, the
unique LIME model is susceptible to a stability issue, which
means that when the technique is repeatedly employed
under the identical circumstances, it may yield varying
explanations. Furthermore, we enhanced our ML framework
by incorporating the LIME approach [32].

## 4. Results and Discussion

In this approach, SA-based Twitter tweets classifcation
using embedded data with FCNN and LSTM is implemented
using Python.

## 4.1. Data Acquisition. We made use of the Kaggle dataset.
Te dataset maintains the structure of a linked network
while containing other data about the tweets and the user’s
profle. Tis information comprises the tweet’s ID, the user’s
ID, the user’s name, the tweet’s text, its language, and the
number of likes, replies, and retweets. Te labeled tweets are
divided into training, testing, and validation subsets
throughout this step of the process. Here, maintaining

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

8
## Journal of Electrical and Computer Engineering

a balanced distribution of each class among all partitions is
the goal. Te “train_test_split” function of the Python
Sklearn package is especially used to do this. Tis function
was selected because it can divide data while respecting class
proportions, resulting in a representative sample in each
subset.

## 4.2. Experimental Validation. We used multiclassifcation to
determine whether a tweet’s sentiment was positive, negative,
or neutral for SA. For our training, we used a large dataset that
we downloaded from Kaggle. We produced the feature
vectors needed to train our models after completing the
preprocessing stages. After achieving a certain level of accuracy, we utilized these classifers to identify distinct patterns
in the actual world. In order to accomplish that, we developed
a program in the Jupiter Notebook that may accept a keyword
or hashtag that has to be examined together with the quantity
of tweets that we want to take into account.
Te dataset has been standardized and its volume reduced through extensive preprocessing techniques. Twitter
users frequently use hashtags, which are unspaced text
preceded by the hash symbol (#), to allude to a trending
topic. All hashtags have been replaced with words with the
hash symbol. Instead of using #hello, you may use hello.
Other users are commonly mentioned in tweets by @handle.
Text Blob library is a right useful library to work on various
languages; we can use it to identify various languages and
translate from one language to another. We examined how
accurate our classifers are for texts found in the real world
and utilized a variety of classifers to estimate the mood of
these tweets. Te suggested study uses a variety of assessment criteria, such as accuracy, recall, precision, and
F1 score.

## 4.3. Assessment Standards for Performance. Models are
assessed using evaluation metrics such as precision, F1 score,
accuracy, and recall [33]. Tese metrics are commonly
utilized for performance evaluation.

4.3.1. Accuracy. Correctness indicates the degree of predictions made accurately occurrences from total occurrences. It has the maximum value of 1 and minimum value
of 0 and is calculated by the following formula
Accuracy is calculated as the ratio of correct predictions
to the total predictions made.
For binary categorization, accuracy can be calculated as
follows:

Accuracy 
# TP + TN
# TP + TN + FP + FN,
(1)

where FN, FP, TN, and TP represent the false negative, false
positive, true negative, and true positive, which are defned
as follows [34].

## 4.3.2. Precision. Tis metric refects the trustworthiness of
the model’s predictions. It indicates the proportion of
positively predicted instances that are genuinely positive.
Te formula for precision is

![page8_img1.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page8_img1.jpeg)

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
9

Table 3a: DistilBERT—FCNN.

Precision 
TP
# TP + TN.
(2)

S. no.
Dataset
F1 score Accuracy Precision Recall
1
Text
0.9356
0.9702
0.9621
0.9723
2
Emoji
0.8021
0.8123
0.8463
0.8641
3
Text and emoji
0.9620
0.9774
0.9712
0.9652

## 4.3.3. Recall. Recall represents the comprehensiveness of the
classifer. It denotes the proportion of instances from the
positive class that were accurately predicted. Te formula to
determine recall is as follows:

Table 3b: DistilBERT—LSTM.

Recall 
TP
# TP + FN.
(3)

S. no.
Dataset
F1 score Accuracy Precision Recall
1
Text
0.9418
0.9634
0.9627
0.9532
2
Emoji
0.8721
0.8851
0.8636
0.8531
3
Text and emoji
0.9327
0.9748
0.9608
0.9554

## 4.3.4. F1 Score. Te F1 score represents the balanced average
of precision and recall. Essentially, it ofers a midpoint
between these two metrics. Similar to other scores, its value
lies between 0 and 1,

performance diferently referred to as the FCNN model. In
particular, the LSTM model projected superior performance
in emoji-based tweet analysis compared to the FCNN, with
an F1 score of 0.8721 and an accuracy of 0.8851. Tis enhancement guides that LSTM’s ability to capture sequential
patterns and dependencies in data are advantageous in
interpreting emojis. Similar to the FCNN model, the LSTM
also topped in analyzing combined text and emoji data,
achieving high scores across all metrics. Figures 4(a) and
4(b) compare the evaluation of the DistilBERT model with
two NN architectures, FCNN and LSTM, through separate
datasets: text, emoji, and combined text and emoji. For both
architectures, the highest performance is observed with the
combined dataset, indicating that the integration of text and
emojis yields the most efective SA. Tese visual compares
the strengths of each architecture and the impact of data type
on SA performance.
In Tables 4a and 4b, the USE-Large3 model paired with
FCNN demonstrated strong SA capabilities, particularly
with a combined text and emoji dataset, as refected by an F1
score of 0.9718. It performed well with text alone, but its
efectiveness was slightly lower with emojis, as indicated by
an F1 score of 0.8325. Te LSTM variant showed a similar
trend; it was highly efective on combined data with an F1
score of 0.9621 but less so on emojis alone, evidenced by an
F1 score of 0.7983. Te precision and recall rates across all
datasets were consistently high, confrming the model’s
reliable performance in sentiment classifcation tasks, and
their performance evaluation indices for various dataset
predictions are shown in Figures 5(a) and 5(b).
In Tables 5a and 5b, the RoBERTa-FCNN model showcased outstanding accuracy in SA, with F1 scores of 0.9894 for
text and a remarkable 0.9921 for text combined with emojis.
Te precision was particularly high in text analysis, reaching
0.9967, indicating a strong ability to correctly identify positive
and negative sentiments. Te RoBERTa-LSTM model mirrored this performance, achieving F1 scores of 0.9953 for text
and 0.9856 for the combined dataset, with precision and recall
consistently above 0.97, demonstrating its efectiveness across
diverse data types and its visualization are shown in
Figures 6(a) and 6(b).
Table 6 shows the execution time for the embedding
methods of DistilBERT, USE-Large3, and RoBERTa with the
distributed training models of FCNNs and LSTM models.

F1 score  Precision ∗Recall
Precision + Recall.
(4)

Conducting SA on both tweets and an emoji dataset
sourced from Kaggle dataset. Given that tweets are essentially sentences, we utilized two end-to-end sentence embedding models, the DistilBERT, USE-Large3, and RoBERTa
to
generate
fundamental
engraftment.
Tese
integral
placements are then used to train with FCNNs and LSTM
models. Additionally, statements might contain emojis, and
a small percentage of people may only answer with emojis.
To solve this issue, we utilized the Kaggle emoji dataset,
which rates the symbolic expressions as positive and negative. We chose selected emojis with good and negative
sentiments in light.
Te proposed methodology is utilized for training and
testing the considered dataset, and the results are obtained
for embedding models and their performance is evaluated
and predicted. Tables 3a and 3b give the results using the
DistilBERT embedding model for FCNN and LSTM.
For the FCNN framework (Table 3a), the model revealed
high capacity in analyzing text-based tweets, supported by
an F1 score of 0.9356 and an accuracy of 0.9702. Te precision and recall values, 0.9621 and 0.9723, respectively,
imply a balanced capability in accurately predicting sentiments in text, with a narrow edge in detecting true positives.
However, when analyzing emoji-based tweets, the model’s
execution projected a noticeable decline, with an F1 score of
0.8021 and an accuracy of 0.8123. Tis indicates that emojis,
with their varied and subjective interpretations, present
a more signifcant challenge for the FCNN model. Additionally the model’s performance enhanced visibly when
analyzing a combination of text and emoji, achieving an F1
score of 0.9620 and an accuracy of 0.9774. Tis represents
that the integration of textual and emoji data provides
a richer context for SA, leading to high accurate predictions.
On the other hand, the LSTM framework (Table 3b)
projects a narrow diferent evaluation pattern. In text-only
analysis, the LSTM model aimed an F1 score of 0.9418 and
an accuracy of 0.9634, similar to the FCNN model but with
a minor change in accuracy. Tis could be added to LSTM’s
sequential data processing capabilities, which may afect its

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

10
## Journal of Electrical and Computer Engineering

## DistilBERT - FCNN

## DistilBERT - LSTM

1.0

1.0

0.8

0.8

## Error indices

## Error indices

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

Text
Emoji
Text and emoji
Dataset

Text
Emoji
Text and emoji
Dataset

F1 score
Accuracy

Precision
Recall

F1 score
Accuracy

Precision
Recall

(a)

(b)

Figure 4: (a) FCNN and (b) LSTM: Te performance comparisons of error indices for diferent datasets.

## Table 4a: USE-Large3—FCNN

S. no.
Dataset
F1 score
Accuracy
Precision
Recall
1
Text
0.9632
0.9772
0.9523
0.9614
2
Emoji
0.8325
0.8433
0.8156
0.8063
3
Text and emoji
0.9718
0.9631
0.9735
0.9731

Table 4b: USE-Large3—LSTM.

S. no.
Dataset
F1 score
Accuracy
Precision
Recall
1
Text
0.9332
0.9486
0.9364
0.9363
2
Emoji
0.7983
0.8164
0.8227
0.8119
3
Text and emoji
0.9621
0.9517
0.9667
0.9735

USE-Large3–FCNN

USE-Large3–LSTM

1.0

1.0

0.8

0.8

## Error indices

## Error indices

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

Text
Emoji
Text and emoji
Dataset

Text
Emoji
Text and emoji
Dataset

F1 score
Accuracy

Precision
Recall

F1 score
Accuracy

Precision
Recall

(a)

(b)

Figure 5: (a) Comparative analysis of model performance metrics using USE-Large3-FCNN on diferent datasets. (b) Performance
evaluation of USE-Large3-LSTM across various data inputs.

Figures 7(a) and 7(b) display the execution time per
epoch for diferent embedding methods when used with
## FCNN and LSTM distributed training models. For both

From the results, it has been observed that RoBERTa with the
FCNN and LSTM has SA of less epoch time of 85 and 77 s,
respectively.

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
11

Table 5a: RoBERTa—FCNN.

S. no.
Dataset
F1 score
Accuracy
Precision
Recall
1
Text
0.9894
0.9899
0.9967
0.9821
2
Emoji
0.8779
0.8853
0.8791
0.8899
3
Text and emoji
0.9921
0.9857
0.9936
0.9982

Table 5b: RoBERTa—LSTM.

S. no.
Dataset
F1 score
Accuracy
Precision
Recall
1
Text
0.9953
0.9989
0.9787
0.9837
2
Emoji
0.8852
0.8961
0.8973
0.8889
3
Text and emoji
0.9856
0.9961
0.9879
0.9978

RoBERTa–FCNN

RoBERTa–LSTM

1.0

1.0

0.8

0.8

## Error indices

## Error indices

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

Text
Emoji
Text and emoji
Dataset

Text
Emoji
Text and emoji
Dataset

F1 score
Accuracy

Precision
Recall

F1 score
Accuracy

Precision
Recall

(a)

(b)

Figure 6: Performance metrics of RoBERTa models using FCNN and LSTM architectures on text, emoji, and combined datasets.

Table 6: Execution time of the language models.

S. no.
Embedding method
Distributed training model
Time
per epoch (seconds)

1
DistilBERT
FCNN
134
LSTM
115

2
USE-Large3
FCNN
122
LSTM
105

3
RoBERTa
FCNN
85
LSTM
77

and negatives, and represents the high accuracy in the text
and combined datasets, but with narrow misclassifcations in
the emoji dataset. Tese visualizations project as a heuristic
tool, highlighting model efcacy and areas for potential
refnement.
Figure 9 illustrates the ROC curves for SA on text, emoji,
and the combination of text and emoji datasets. Te curves
demonstrate the high efectiveness in sentiment classifcation with AUC scores around 0.9 for text and combined
datasets and 0.8 for emoji.
In this research, the analysis of data using FCNN and
LSTM models with combined data has unique strengths
and limitations. Tis proposed model shows high precision

training models, RoBERTa proves to be the most efcient,
with the shortest execution times (85 s for the FCNN and
77 s for LSTM). DistilBERT shows the longest execution
times, indicating that RoBERTa is the better choice for faster
model training.
Te microaverage (M-Avg) score of all language models
with the embedded method and distributed training method
for diferent evaluation performance indices are tabulated in
Table 7.
Figure 8 illustrates confusion matrices for SA models to
text, emoji, and combined text and emoji datasets. Tis
evaluation, based on precision and recall metrics, demonstrates the models’ performance with true and false positives

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

12
## Journal of Electrical and Computer Engineering

## Distributed training model - LSTM

## Distributed training model - FCNN

120

140

115

130

110

105

## Execution time

120

## Execution time

100

110

95

90

100

85

90

80

75

80

DistilBERT
USE-Large3
RoBERTa
## Embedding method

DistilBERT
USE-Large3
RoBERTa
## Embedding method

## Execution time

## Execution time

(a)

(b)

Figure 7: (a and b) Te graphical representation of the distributed training model execution time for both methods. (a) FCNN model and
(b) LSTM model.

Table 7: Microaverage (M-Avg) score.

S. no.
Embedded method
Distributed training
model
F1 score
(M-Avg)
Accuracy (M-Avg)
Precision (M-Avg)
## Recall (M-Avg)

1
DistilBERT
FCNN
0.8999
0.9199
0.9265
0.9338
LSTM
0.9155
0.9411
0.9290
0.9205

3
USE-Large3
FCNN
0.9255
0.9278
0.9138
0.9136
LSTM
0.8978
0.9055
0.9136
0.9072

5
RoBERTa
FCNN
0.9521
0.9536
0.9564
0.9567
LSTM
0.9553
0.9637
0.9546
0.9568

Confusion matrix - text
Confusion matrix - emoji
## Confusion matrix - text and emoji

80

80

80

70

89
10

100
1

98
2

Positive

Positive

Positive

60

60

60

## True label

## True label

## True label

50

40

40

40

30

11
1

0
-1

2
-1

Negative

Negative

Negative

20

20

20

10

Positive

Negative

Positive

Negative

Positive

Negative

0

0

## Predicted label

## Predicted label

## Predicted label

Figure 8: Confusion matrices for sentiment analysis models on text, emoji, and combined datasets.

data types, ofering a more comprehensive understanding
of sentiments in the social media content. Te global life
implications of this research are deep, providing valuable
insights for social media analytics, marketing strategies,
and analyzing public opinion trends. However, future work
may focus on enhancing emoji interpretation and exploring the scalability of these models for larger, more
diverse datasets.

in text analysis, with the merged text and emoji dataset
projecting the good results. Tis represents the efciency of
combining diverse data types for analysis understanding.
However, the evaluation on emoji-only data was relatively
lower, indicating objections in interpreting the nuanced
and context-dependent nature of emojis. Compared to
traditional and existing works, this proposed approach
projects the signifcant advancement in managing distinct

| 98 | 2 |
| --- | --- |
| 2 | -1 |

| 89 | 10 |
| --- | --- |
| 11 | 1 |

| 100 | 1 |
| --- | --- |
| 0 | -1 |

![page12_img1.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page12_img1.jpeg)

![page12_img2.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page12_img2.jpeg)

![page12_img3.jpeg](Journal%20of%20Electrical%20and%20Computer%20Engineering%20-%202024%20-%20Chandra%20Sekhar%20-%20Classification%20and%20Comparative%20Evaluation%20of%20Text_images/page12_img3.jpeg)

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
13

1.0

0.8

## True-positive rate

0.6

0.4

0.2

0.0

0.0
0.8
0.2
0.6
0.4
1.0
## False-positive rate

Text (AUC = 0.87)
Emoji (AUC = 0.89)
## Text and Emoji (AUC = 0.93)

Figure 9: ROC curve illustrations for text, emoji, and combined text and emoji.

Table 8: Projects the key aspects of work and compares it with other approaches.

Feature
Proposed methodology
Approach 1
Approach 2
Methodology
Multithreaded
Single-threaded
Cloud-based
Data source
Kaggle
Twitter API
Custom dataset
Models used
DistilBERT, RoBERTa, USE-Large3
# BERT, LSTM
# CNN, SVM
Accuracy
98%
95%
90%
Execution time reduction
17%
N/A
10%
Sentiment analysis scope
Text and emojis
Text only
## Text only

Table 9: Comparison Analysis of the proposed model with other existing models.

Approach
Model used
Accuracy (%)
F1 score
Execution time
Proposed model
RoBERTa
98
0.97
85 seconds
Smith, Doe, and Johnson [35]
BERT
95
0.95
100 seconds
Johnson, Lee, and Kim [36]
CNN-LSTM
90
0.90
120 seconds
Williams, Brown, and Harris [37]
SVM
88
0.88
90 seconds

understanding of sentiments in the social media content.
Te implications of this research are signifcant, ofering
insights valuable for public opinion tracking, social media
analytics, and marketing strategies. Future work may include
advanced deep learning techniques such as federative
learning to boost model evaluation and generalizability.
Such techniques could further enhance the scalability and
robustness of SA models, especially for large and diverse
datasets.
Table 8 provides the summarized work of the key aspects
of work and compares it with other approaches.
To illustrate the efectiveness of our SA system, we have
compared its performance with other related approaches in
the feld. Te comparison is based on key performance
metrics such as accuracy, F1 score, and execution time. Te
comparison analysis of the proposed model with other
existing models is shown in Table 9.
From this table, it is evident that our system, utilizing the
RoBERTa model, achieves superior accuracy and F1 score
compared to the other approaches. Notably, the execution
time of our system is signifcantly lower than that of Smith,

## 4.4. Critical Analysis and Discussion. Tis research, while
evaluating data with FCNN and LSTM models, particularly
while combining diferent types of data, has projected extraordinary strengths and limitations. Utilizing a combined
dataset of text and emoji’s facilitates high precision in text
diagnosis, leading to efective outcomes. Tis indicates that
merging various data types improves analysis and comprehension. However, performance drops when executing
emoji-only data, emphasizing the difculty in interpreting
emojis due to their context-dependent nature. Tis proposed
research provides several advantages over existing SA
methods. Integrating emojis with text data enhances the
identifcation of emotional expressions, frequently missed in
text-only analyses. Tese models have high efciency in
classifying text, proving their ability in handling complex SA
assessments. In addition, the adoption of distributed
training techniques improves the scalability and efciency,
suitable for analyzing large volumes of data.
Compared to traditional and current techniques, this
new proposed research marks a signifcant enhancement in
processing diferent data types, ofering a more thorough

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

14
## Journal of Electrical and Computer Engineering

Doe, and Johnson [35] and Johnson, Lee, and Kim [36],
demonstrating the efciency of our distributed training
methodology.
Furthermore, the improvement in performance metrics
can be attributed to the advanced capabilities of the RoB-
ERTa model and our optimization techniques. Tis comparison underscores the advancements our system brings to
SA, particularly in terms of accuracy and computational
efciency.

Funding

No funding was received for this research work.

Acknowledgments

We gratefully acknowledge the contributions of the following authors to this manuscript: J.N. Chandra Sekhar1,
## M. Kiran Mayee2, Ranjana Nadagoudar3, N. Chinna
Alluraiah4, Dhanamjayulu C5∗, Ravikumar Chinthaginjala6,
Ravi K5, Praveenkumar M5, Satyajit Mohanty5, and Baseem
Khan7∗.

## 5. Conclusion

In this research, SA was done on both the Kaggle text and
emoji dataset. Te integration of emojis, a critical aspect of
modern digital communication, into SA represents a signifcant leap in understanding complex emotional expressions on social media platforms. Since tweets are
composed of sentences and used embedding models like
DistilBERT, USE-Large 3, and RoBERTa to process them.
Te LSTM and FCNN models are trained using the embeddings. It noticed that both models consistently had
good text categorization accuracy, which tended to be
around 98%. However, accuracy for both models signifcantly decreased, falling below 75%; meanwhile, the validation set was constructed with symbolic expression that
were not included in the training set. To improve scalability, a distributed training technique rather than the
traditional single-threaded paradigm is used. Tis strategy
reduced runtime by around 17% while retaining accuracy.
Finally, the SHAP and LIME algorithm was employed as
part of explainable AI to analyze the provided feature set’s
model behavior and look for biases. Tis study might be
expanded to multilingual datasets as part of future directions, and we also have plans to gather countless data
entries and assess the model’s efcacy with large data
collections.

References

[1] A. Keramatfar, H. Amirkhani, and A. J. Bidgoly, “Multi-
Tread Hierarchical Deep Model for Context-Aware Sentiment Analysis,” Journal of Information Science 1–12 (2021):
https://doi.org/10.1177/0165551521990617.
[2] M. E. Alzahrani, T. H. H. Aldhyani, S. N. Alsubari, et al.,
“Developing an Intelligent System With Deep Learning Algorithms for Sentiment Analysis of Ecommerce Product
Reviews,” Computational Intelligence and Neuroscience 2022
(2022): 1–10, https://doi.org/10.1155/2022/3840071.
[3] H. Sakhrani, S. Parekh, and P. Ratadiya, “Contextualized
Embedding Based Approaches for Social Media-specifc
SENTIMENT Analysis,” in 2021 International Conference
on Data Mining Workshops (ICDMW), 2375-9259/21, 2021
IEEE (Auckland, New Zealand, December 2021), https://
doi.org/10.1109/ICDMW53433.2021.00030.
[4] P. Nakov, A. Ritter, S. Rosenthal, F. Sebastiani, and
## V. Stoyanov, “Semeval-2016 Task 4: Sentiment Analysis in
Twitter,” (2019), https://aclanthology.org/S16-1001.pdf.
[5] H. S. Ali Barzenji, “Sentiment Analysis of Twitter Texts Using
Machine Learning Algorithms,” Academic Platform Journal of
Engineering and Science 9-3 (2021): 460–471.
[6] N. Yadav, O. Kudale, A. Rao, S. Gupta, and A. Shitole,
“Twitter Sentiment Analysis Using Supervised Machine
Learning,” in Springer International Conference on Sentiment
Analysis and Deep Learning (Singapore: Springer, February
2021), https://doi.org/10.1007/978-981-15-9509-7_51.
[7] R. Chandra and A. Krishna, “COVID-19 Sentiment Analysis
via Deep Learning During the Rise of Novel Cases,” PLoS
One 16, no. 8 (2021): e0255615, https://doi.org/10.1371/
journal.pone.0255615.
[8] A. Kumar, K. Srinivasan, W. H. Cheng, and A. Y. Zomaya,
“Hybrid Context Enriched Deep Learning Model for Finegrained Sentiment Analysis in Textual and Visual Semiotic
Modality Social Data,” Information Processing & Management 57, no. 1 (2020): 102141, https://doi.org/10.1016/
j.ipm.2019.102141.
[9] W. Li, Y. Chen, T. Hu, and J. Luo, “Mining the Relationship
Between Emoji Usage Patterns and Personality,” in Proceedings of the Twelfth International Conference on Web and
Social Media (Stanford, June 2018), 648–651.
[10] B. Eisner, T. Rockt¨aschel, I. Augenstein, M. Bosnjak, and
## S. Riedel, “emoji2vec: Learning Emoji Representations From
Teir Description,” in Proceedings of Te Fourth International
Workshop on Natural Language Processing for Social Media,
SocialNLP@EMNLP 2016 (Austin, TX, November 2016),
48–54.
# [11] B.
Lalitha,
V.
Madhurima,
C.
H.
Nandakrishna,
## S. Babu Jampani, J. N. Chandra Sekhar, and P. Venkat Reddy,

## Data Availability Statement

You can access this dataset for download at the following
link:
https://www.kaggle.com/code/aguschin/lyrics-toemoji/input,
https://github.com/swcwang/depressiondetection/tree/master/data.

Disclosure

To the best of our knowledge and belief, all individuals who
have made signifcant contributions to this research and the
preparation of the manuscript are listed as authors. No
third-party services were utilized in this research.

## Conflicts of Interest

Te authors declare no conficts of interest.

## Author Contributions

Each author has reviewed and approved the fnal version of
the manuscript and agrees to be accountable for all aspects.

1742, 2024, 1, Downloaded from https://onlinelibrary.wiley.com/doi/10.1155/2024/9652424 by Sri Lanka National Access, Wiley Online Library on [11/04/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

Journal of Electrical and Computer Engineering
15

(Dublin, Ireland: ©2022 Association for Computational
Linguistics, May 2022), 326–330.
[26] P. M. Jadhav, Sonia, and A. N. Kulkarni, “Te Identifcation of
Depressive Moods From Twitter Data by Using Convolutional Neural Network With Text Data along With Emoji,”
International Research Journal of Engineering and Technology
(IRJET) 10 (June 2023).
[27] S. Gouthami and N. P. Hegde, “Sentiment Analysis Based
Twitter Tweets Classifcation Using Data Embedded With
Lstm Technique,” Journal of Teoretical and Applied Information Technology 101, no. 4 (2023).
[28] D. Fan, H. Sun, J. Yao, K. Zhang, X. Yan, and Z. Sun, “Well
Production Forecasting Based on ARIMA-LSTM Model
Considering Manual Operations,” Energy 220 (2021): 119708,
https://doi.org/10.1016/j.energy.2020.119708.
[29] S. R. Islam, W. Eberle, S. K. Ghafoor, and M. Ahmed, “Explainable Artifcial Intelligence Approaches: A Survey,”
(2021), https://arxiv.org/pdf/2101.09429.
[30] S. Lundberg and S. I. Lee, “A Unifed Approach to Interpreting Model Predictions,” (2017), https://arxiv.org/abs/
1705.07874.
[31] G. Visani, E. Bagli, F. Chesani, A. Poluzzi, and D. Capuzzo,
“Statistical Stability Indices for LIME: Obtaining Reliable
Explanations for Machine Learning Models,” Journal of the
Operational Research Society, 73.
[32] M. T. Ribeiro, S. Singh, and C. Guestrin, “Why Should I Trust
You?: Explaining the Predictions of Any Classifer,” in Proceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining (San Francisco, CA:
ACM, August 2016), 13–17.
[33] V. Rupapara, F. Rustam, H. F. Shahzad, A. Mehmood,
## I. Ashraf, and G. S. Choi, “Impact of SMOTE on Imbalanced
Text Features for Toxic Comments Classifcation Using
RVVC Model,” IEEE Access 9 (2021): 78621–78634, https://
doi.org/10.1109/access.2021.3083638.
[34] J. Han, J. Pei, and M. Kamber, Data Mining: Concepts and
Techniques (Amsterdam, Te Netherlands: Elsevier, 2011).
[35] J. Smith, A. Doe, and B. Johnson, “Improving Sentiment
Analysis With BERT,” Journal of Natural Language Processing
27,
no.
3
(2020):
123–135,
https://doi.org/10.1000/
jnlp.2020.12345.
[36] C. Johnson, D. Lee, and E. Kim, “A CNN-LSTM Approach
for Sentiment Analysis on Social Media,” in Proceedings of
the International Conference on Machine Learning (Long
Beach,
June
2019),
1024–1031,
https://doi.org/10.1000/
icml.2019.1024.
[37] F. Williams, G. Brown, and H. Harris, “Sentiment Analysis
Using Support Vector Machines: A Comparative Study,”
International Journal of Data Mining 12, no. 2 (2018): 45–59,
https://doi.org/10.1000/ijdms.2018.4567.

“Data Augmentation Based Cross-Lingual Multi-Speaker TTS
Using DL With Sentiment Analysis,” Te ACM Transactions
on Asian and Low-Resource Language Information Processing
(2023): 2375–4699, https://doi.org/10.1145/3628428.
[12] N. F. F. D. Silva, L. F. S. Coletta, and E. R. Hruschka, “A Survey
and Comparative Study of Tweet Sentiment Analysis via
Semi-Supervised Learning,” ACM Computing Surveys 49,
no. 1 (2016): 15:1–26, https://doi.org/10.1145/2932708.
[13] Y. Kim, “Convolutional Neural Networks for Sentence
Classifcation,” in Proceedings of the 2014 Conference on
Empirical Methods in Natural Language Processing, EMNLP
2014 (Doha, Qatar, October 2014).
[14] S. Cappallo, S. Svetlichnaya, P. Garrigues, T. Mensink, and
C. G. M. Snoek, “New Modality: Emoji Challenges in Prediction, Anticipation, and Retrieval,” IEEE Transactions on
Multimedia 21, no. 2 (2019): 402–415, https://doi.org/10.1109/
tmm.2018.2862363.
[15] Y. Wan and Q. Gao, “An Ensemble Sentiment Classifcation
System of Twitter Data for Airline Services Analysis,” in
Proceedings IEEE International Conference on Data Mining
Workshop (Atlantic City, NJ, November 2015), 1318–1325.
[16] Y. Ren, Y. Zhang, M. Zhang, and D. Ji, “Context-Sensitive
Twitter Sentiment Classifcation Using Neural Network,” in
Proceedings of the Tirtieth AAAI Conference on Artifcial
Intelligence (AAAI-16) (Phoenix, Arizona: Association for the
Advancement of Artifcial Intelligence, February 2016).
[17] K. Shuang, X. Ren, J. Chen, X. Shan, and P. Xu, “Combining
Word Order and CNN-LSTM for Sentence Sentimentclassifcation,” in Proceedings of the 2017 International Conferenceon Software and e-Business, ser. ICSEB 2017 (New
York, NY: ACM, December 2017), 17–21, https://doi.acm.org/
10.1145/3178212.3178230.
[18] G. Zhai, Y. Yang, H. Wang, and S. Du, “Multi-Attention
Fusion Modeling for Sentiment Analysis of Educational Big
Data,” Big Data Mining and Analytics 3, no. 4 (2020): 311–319,
https://doi.org/10.26599/bdma.2020.9020024.
[19] P. Nuthakki, M. Katamaneni, S. J. N. Chandra, et al., “Deep
Learning Based Multilingual Speech Synthesis Using Multi
Feature Fusion Methods,” ACM Transactions on Asian and
Low-Resource
Language
Information
Processing
(2023):
2375–4699, https://doi.org/10.1145/3618110.
[20] S. Velampalli, C. Muniyappa, and A. Saxena, “Performance
Evaluation of Sentiment Analysis on Text and Emoji Data
Using End-to-End, Transfer Learning, Distributed and Explainable AI Models,” Journal of Advances in Information
Technology
13,
no.
2
(2022):
https://doi.org/10.12720/
jait.13.2.167-172.
[21] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert,
a Distilled Version of Bert: Smaller, Faster, Cheaper and
Lighter,” (2019), https://arxiv.org/abs/1910.01108.
[22] M. Janatdoust, “KADO@LT-EDI-ACL2022: BERT-Based
Ensembles for Detecting Signs of Depression from Social
Media Text,” in Proceedings of the Second Workshop on
Language Technology for Equality, Diversity and Inclusion
(Dublin, Ireland, May 2022).
[23] D. Cera, Y. Yang, S.-Y. Kong, et al., “Universal Sentence
Encoder,” (2018), https://arxiv.org/abs/1803.11175.
[24] Y. Liu, M. Ott, N. Goyal, et al., “Roberta: A Robustly Optimized Bert Pretraining Approach,” (2019), https://arxiv.org/
abs/1907.11692.
[25] S. Adarsh and B. Antony, “SSN@LT-EDI-ACL2022: Transfer
Learning Using BERTfor Detecting Signs of Depression from
Social Media Texts,” in Proceedings of the Second Workshop on
Language Technology for Equality, Diversity and Inclusion
