# Emoji, Text, and Sentiment Polarity Detection Using Natural Language Processing

information

Article
Emoji, Text, and Sentiment Polarity Detection Using Natural
## Language Processing

Shelley Gupta 1,*
, Archana Singh 2 and Vivek Kumar 3,*

1
Department of Computer Science and Engineering, Amity School of Engineering and Technology,
Noida 201303, India
2
Department of Artiﬁcial Intelligence, Amity School of Engineering and Technology, Noida 201303, India;
asingh27@amity.edu
3
Department of Mathematics & Computer Science, University of Cagliari, 09122 Cagliari, Italy
*
Correspondence: shelley.g17@gmail.com (S.G.); vivek.kumar@unica.it (V.K.)

Abstract: Virtual users generate a gigantic volume of unbalanced sentiments over various online
crowd-sourcing platforms which consist of text, emojis, or a combination of both. Its accurate analysis
brings proﬁts to various industries and their services. The state-of-art detects sentiment polarity using
common sense with text only. The research work proposes an emoji-based framework for cognitive–
conceptual–affective computing of sentiment polarity based on the linguistic patterns of text and
emojis. The proposed emoji and text-based parser articulates sentiments with proposed linguistic
features along with a combination of different emojis to generate the part of speech into n-gram
patterns. In this paper, the sentiments of 650 world-famous personages consisting of 1,68,548 tweets
have been downloaded from across the world. The results illustrate that the proposed natural
language processing framework shows that the existence of emojis in sentiments many times seems
to change the overall polarity of the sentiment. By extension, the CLDR name of the emoji is utilized
to evaluate the accurate polarity of emoji patterns, and a dictionary of sentiments is adopted for
evaluating the polarity of text. Eventually, the performances of three ML classiﬁers (SVM, DT, and
Naïve Bayes) are evaluated for proposed distinctive linguistic features. The robust experiments
indicate that the proposed approach outperforms the SVM classiﬁer as compared to other ML
classiﬁers. The proposed polarity detection generator has achieved an exceptional perspective of
sentiments presented in the sentence by employing the ﬂow of concept established, based on linguistic
features, polarity inversion, coordination, and discourse patterns, surpassing the performance of
extant state-of-the-art approaches.

Citation: Gupta, S.; Singh, A.; Kumar,

## V. Emoji, Text, and Sentiment Polarity

## Detection Using Natural Language

Processing. Information 2023, 14, 222.

Keywords: emojis; pattern; polarity inversion; sentiment polarity; ML classiﬁers

https://doi.org/10.3390/

info14040222

Academic Editors: Eftim Zdravevski,

## Petre Lameski and Ivan Miguel Pires

## 1. Introduction

## Received: 10 March 2023

The origination of the world wide web has caused an expanded use of social networking sites, electronic commerce sites, weblogs, forums, etc. The artists, players, layman,
professional organizations, etc. utter their sentiments, expressions, attitudes, and knowhow by means of a totally new communicating online language [1] consisting of both
textual and emoji [2,3]. Sentiment analysis is such a programmed process of analyzing
the huge count of views posted on social media about a given subject [4–12]. It facilitates
the companies to enhance their product quality aspects, and regulate market strategy and
client service from the user-generated content [1,13,14]. Sentiment analysis is an approach
that can be done at the sentence level, document level, and aspect level [15–19].
Today, NLP serves as a major medium of communication between people and machines. The research work [20] depicts a reassuring empirical ground for calculating the
excellence of NLP-based languages, clubbing the implicit perception of judgment as an
add-on criterion. Whereas the research work of [21] depicts an approach based on a corpus
to determine the complexity level of MOEPT, it contributes to regulating the complexity and

## Revised: 27 March 2023

## Accepted: 31 March 2023

## Published: 5 April 2023

Copyright:
© 2023 by the authors.

Licensee MDPI, Basel, Switzerland.

## This article is an open access article

distributed
under
the
terms
and

conditions of the Creative Commons

Attribution (CC BY) license (https://

creativecommons.org/licenses/by/

4.0/).

Information 2023, 14, 222. https://doi.org/10.3390/info14040222
https://www.mdpi.com/journal/information

![page1_img1.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page1_img1.png)

![page1_img2.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page1_img2.png)

![page1_img3.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page1_img3.png)

![page1_img4.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page1_img4.png)

Information 2023, 14, 222
2 of 18

material of MOEPT. The research work [22] marks the open challenges and forthcoming
control for contrastive NLP pertaining to image representation.
Textual tweets consist of alphabets, numbers, and special characters whereas emojis
are the pictorial representation of a user’s emotions and can be used with text or without
text. The emoji portrayal can be a picture, an encoded character, or a sequence of encoded
characters. Emojis have provided the world with a new language to express their emotions
in vibrant, multicolor, attractive, and amusing ways, with the need for few or no words in
the message [3,23–25].
In 1997, the concept of the emoji was initially used in Japanese mobile phones and
later adopted by companies such as Google, Apple, Twitter, Facebook, etc. Since, 2006,
web-based sentiment, such as the expression on Twitter, e-commerce reviews, social media
content, etc., has become an extensively popular research area known as sentiment analysis [1,26]. With the growth of emojis and web-based platforms, users have the choice of
proclaiming their conscience by employing text in combination with emojis [2,3,24,25].
The brand-new categories of Google emoji are nongender-speciﬁc emojis such as
🦯, 🦼, and 🦽, professions such as 🧑🏫judge, farmer, etc., gender emojis such as
red hair, curly haired, etc. https://blog.emojipedia.org/apple-emoji-turns-10/2020/08;
https://blog.emojipedia.org/apple-emoji-turns-10/#fn1; https://www.apple.com/in/
newsroom/2019/07/apple-offers-a-look-at-new-emoji-coming-to-iphone-this-fall/; https:
//blog.emojipedia.org/google-march-2020-pixel-feature-drop-emoji-changelog/). Emoji
14.0 (https://unicode.org/Public/emoji/14.0/emoji-test.txt accessed on 2 February 2023)
was released in September 2021, with emojis such as 🤝handshake, bubbles, pregnant
man, etc.
The concept-based sentiment analysis approaches [27] aim at semantic analysis using
semantic networks or web ontologies of text. It provides a combination of conceptual
and affective information [28] attached to natural language sentiments. It is intended to
empower comparative ﬁne-grained feature-based sentiment analysis in lieu of isolated
sentiments or opinions.
The proposed approach helps in determining the correct polarity of the sentences
with text only, text and emoji only, emoji only, along with multiple emojis as well. It
also determines the correct polarity with a pattern of coordinated discourse and polarity
inversion structures of online natural language sentiments. The correct polarity detection of
an online sentiment helps in evaluating product analysis [29], market competitor research,
mental wellbeing [30], etc. The correct sentiment polarity evaluation of social media posts
of individuals can be utilized in determining the historical and present anxiety, stress, and
depression levels as well, which in turn can help in reducing suicide cases in society as
well.
The paper presents sentiment polarity computing stationed on a text and emoji-based
tree generation, parser generator, and pattern formation. Thus, the central objectives of this
research article are enumerated as:

1.
Introducing a novel cognitive paradigm of sentiment polarity computing framework
based on parser generation by deconstructing the natural language concepts of online
sentiments into text and emoji;
2.
To propose a cognitive sentence level polarity detection using enormous complex
pattern rules for employing the linguistic features of the modern online natural
language, i.e., emoji in conjunction with text, text with multiple emojis, emoji only,
and text only;
3.
To familiarize with extensive rules of pattern-based coordinated, discourse, and
polarity inversion structures of online natural-language sentiment polarity-detection
generator;
4.
The evaluation of the introduced approach is based on three distinctive classiﬁers:
Naïve Bayes, support vector machine, and decision tree with three proposed online
linguistic features with emojis, in conjunction with text, text with multiple emojis,
emoji only, and text only;

Information 2023, 14, 222
3 of 18

5.
To determine which, among the three classiﬁers, works well with our proposed
approach;
6.
To conduct extensive experiments with complex sentences to implicate the robustness
and effectiveness of the suggested text and emoji-based sentiment polarity detection
approach.

The research article is constructed as follows. Section 2 provides the related work in
the ﬁeld of sentiment analysis polarity detection. Section 3 has contemplated the proposed
approach addressing tree formation in Section 3.1, the parsing algorithm in Section 3.2,
and the pattern formation rules for text and emoji in Section 3.3. Section 3.6 represents
the emoji, text, and ﬁnal polarity score evaluation. Section 4 provides the implementation
details, results, and discussion. Finally, Section 5 includes the conclusion and future work.

## 2. Related Work

The present state-of-art related to commonsense and knowledge-based conceptual
and affective sentiment analysis are discussed below in detail along with Table 1:
The framework [31] reﬁned the corpus utilizing Sentic LDA. It developed clusters
labeled with an aspect category; these clusters are then manually labeled based on the number of aspect lexicons available in them. The approach of OntoSenticNet [32] provides an
explanation of the hierarchy of concepts by establishing the relationship between concepts
and sentiment analysis. The research work in [33] employed common sense knowledge to
rig an aspect-based sentiment analysis and a targeted sentiment analysis. They utilized
LSTM and hierarchical attention to propose Sentic LSTM. This research work [34] is done
with text, and not employing the role of emojis/emoticons.
The research work of [35] expanded the rules of linguistics to extract concept-based
features. It has employed FCA to determine features and their association between concept
and ontologies relations.
The research work of [36] co-LSTM examines big online data ensuring scalability
and is also free from domain constraints. It is a hybrid model of CNN and LSTM. CNN
performs well for local feature selection, whereas LSTM is for big text sequential analysis.
The researchers here also did not consider the role of emojis in data analysis.
The commonsense-based textual-sentiment analysis [37] is equipped with a multiplepolarity attention framework. It evaluates the strength of various relational insights using
the knowledge base of ConceptNet. It efﬁcaciously enhances sentence presentation by
adopting a bidirectional LSTM approach coupled with multiple-polarity orthogonal attention. The state of the art has also not considered the role of emojis in this work.
The research work [38] applied latent dirichlet allocation (LDA) and probabilistic
latent semantic analysis (PLSA) algorithms enhancing the textual aspect-based sentiment
analysis utilizing the concepts, lexicon patterns, and negations for concept learning. It is a
graph-based approach and calculated the score among distinctive nodes using the SimRank
algorithm. This approach also ignores the role of emojis in sentiment analysis.
The research work in [39] named ﬁne-grained aspect-based sentiment (FiGAS) analysis
is also a textual data-based sentiment analysis for the ﬁnancial and economic domains
assigning the polarity scores between −1 and +1. This lexicon-based polarity-detection
approach caters to enormous semantic rules, but this approach also does not deal with
linguistic-feature emojis of the dataset.
The above-stated research gap and the popularity of emojis/emoticons among social
media users [3,23] promoted us to design a commonsense-based conceptual sentiment
polarity-detection-based framework.

Information 2023, 14, 222
4 of 18

Table 1. Comparative study of state of the art for various sentiment analysis approaches.

Approach
Linguistic
Features
Approach Used/
Classiﬁer Used
Dataset
Accuracy

235,793 hotel reviews
obtained from the hotel’s
review site tripadvisor.com,
Semeval-2014

Precision of 88.25% for
## Semeval-2014 dataset

Poria et al. [31]
Text
SenticLDA, dependency
trees, bag of words.

Dragoni et al. [32]
Text
SenticConcept, Domain,
Polarity Instance, and
Resource.

A semantic network of
100,000 concepts
-

88.80% for SentiHood
(development set) and
76.47% for Semeval-2015
dataset

Ma et al. [33]
Text
Sentic LSTM
## Semeval-2015, Sentihood

Khattak et al. [35]
Text
# SVM, MNB, LR, RF, KNN
Amazon phone reviews
87.5% with SVM classiﬁer

Airline review, US
presidential election
review, Movie review, and
car self-driving review

Behera et al. [36]
Text
# CNN, LSTM

98.4% for airline dataset

Liao et al. [37]
Text
Bidirectional LSTM model
with multipolarity
orthogonal attention
SMP2019-ECISA
88.7% for B+MPOA (BERT)

Pradhan et al. [38]
Text
Naïve Bayes
SemEval-14: Laptop and
restaurant
86.32% for restaurant and
82.64% for Laptop dataset.

English sentences in the
economic and ﬁnancial
domains from the
commercial Dow Jones
data, news, and analytics
(DNA) platform.

3.26 average algorithm
ranks by using the median
score of the nine
annotators.

Consoli et al. [39]
Text
-

90.78% with but and
adversatives and 92.18%
with polarity inversion

Proposed approach
Text + emoji
SVM
Naïve Bayes
## Decision tree

1, 68,548 tweets posted by
650 unique personages

Some examples of the state of the art related other than commonsense knowledgesentiment analysis involve:
The research work in [34] provides cohort analysis based on the solution of real-world
issues in the research of e-commerce customers. The research work in [40] accompanied
discriminative and semantic evaluations applying similarity variance for topic identiﬁcation
of Persian, integrating them with fuzzy similarity as well. Whereas the research work in [41]
indicates the in-depth utility of Markov models for processing natural language in machine
learning. The research work [9] aims at predicting the selection of an emoji automatically
for a text message and categorizing the tone of the message into seven categories using
algorithms of machine learning.
As indicated, the above literature review does not incorporate the role of sentiment
polarity evaluation incorporating the linguistic patterns of text, emojis, and multiple emojis,
along with coordinated structures, discourse structures, and polarity inversion. Thus, this
motivated us to propose an approach for the same.

## 3. Proposed Framework

As emojis are used extensively in online sentiments [2,25,42], sentiment polarity
detection is accurate in the context of affective cognitive computing by considering both
text and emojis. In the proposed model of concept-level sentiment analysis, we have
addressed the formation of patterns with the introduction of emojis for commonsense-

Information 2023, 14, 222
5 of 18

based sentiment polarity detection. The proposed cognitive–conceptual–affective sentiment
polarity detection framework of sentiment sentences (Si), incorporating text and emojis is
presented with in Figure 1. The new framework comprises the below segments:

(1)
Tree and parsing algorithm generation, a semantic tree, and a parser generator are
developed for sentiments consisting of text and emoji;
(2)
Pattern formation: it evolves the patterns for the combination of text and emoji-based
sentences to determine the accurate polarity of sentiment. It also considered polarity
inversion along with a coordinated and discourse structure-based complex patterns
for conceptual and context-based sentiment polarity computing;
(3)
Polarity evaluation: the polarity of text and emojis are evaluated to evolve the concluding polarity of sentiment sentence considering text and emoji-based patterns of
the proposed approach;
(4)
The three ML classiﬁcation techniques are used to train the model proposed;
(5)
Final polarity assessment is done based on the above steps two, three, and four. Its
generated values are positive, negative, or neutral polarity.

Figure 1. Proposed framework.

## 3.1. Tree Generation

The purpose of the proposed parser (Figure 2) is to break the online sentiments into a
logical form. It interprets online sentiment clauses into concepts to be used in commonsense
and affective computing [27,43,44]. The knowledge of the text and emoji linguistic features,
along with the conceptual and affective information of their patterns, helps in knowing the
emotion, score, and polarity associated with the sentiment more accurately.

![page5_img1.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page5_img1.png)

Information 2023, 14, 222
6 of 18

Sentence S1: Text and emoji both   We are going to the garden to bring some flowers and herbs.😀.

Root

Root

Root

S

FRAG

FRAG

NP

VP

S

😀

PRP

VBP

VP

VP

We

are

VBG

PP

TO

VP

to

VB

NP

going

TO

NP

bring

NP

CC

NP

to

DT

NN

NNS

and

DT

NNS

the

garden

flowers

some

herbs

(a)

Sentence S2: Text and multiple emojis  We are going to the garden to bring some flowers and herbs. 😔 🙂.

(b)
## Sentence S2: Emojis only   😍😍😊 😊

Root

Root

Root

Root

FRAG

FRAG

FRAG

FRAG

😍

😍

😊

😊

(c)

Figure 2. Tree formation for sentiments with (a) text and emoji both (b) text and multiple emojis
(c) emojis only.

In POS (part of speech), the object is a noun, pronoun, or noun phrase on which the
verb performs an action. The commonsense information can be represented by pairs of
objects and emojis. The object-emoji expressions that POS combinations considered are
depicted in below Table 2.

|  | Root S |
| --- | --- |

| Root |  |
| --- | --- |
|  |  |
| FRAG |  |
|  |  |
| 😍 |  |

| Root |  |
| --- | --- |
|  |  |
| FRAG |  |
|  |  |
| 😍 |  |

| Root |  |
| --- | --- |
|  |  |
| FRAG |  |
|  |  |
| 😊 |  |

![page6_img36.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page6_img36.png)

Information 2023, 14, 222
7 of 18

Table 2. POS combinations for n-gram based on text and emoji and multiple-emoji combinations.

POS Combinations
Description
Example

## Text + Emoji

# NOUN + EMOJI
Noun and emoji as standalone are added
to concept
car 😊, laptop 💻, ice cream 😍.

# NOUN + NOUN + EMOJI
Add two nouns as a single concept and
emojis as separate.
ice-cream ☀🤤, wheelchair 😭,

chocolate biscuits 😋.

# ADJ + NOUN + EMOJI
Adj + Noun as combinations is added to
the objects list. Emoji as isolated are
added to the concept.
expensive laptop 😍, beautiful car 🤩.

# ADJ + STOPWORD + EMOJI
The adjective and emoji are added to
concept.
lovely as ❤ 🩸, sparking as 🤩.

# NOUN + ADJ + EMOJI
In this pair, adjective, noun, and emoji as
standalone are added as a valid concept.
man, big🌹, ﬂower pink 🌻

# STOPWORD + NOUN + EMOJI
The stop word is discarded. The noun
and emoji are considered valid.
as man 💪, this ﬂower 🌻👌.

# STOPWORD + ADJ + EMOJI
Emoji and adjective are added as a
standalone concept.
as beautiful 🌹😍🙂, being happy😃

## Emoji only

Emojis
Each emoji is added as a standalone
concept.
🌹😍🙂

Emojis
Each emoji is added as a standalone
concept.
😍🙂✈😒😒

## 3.2. Parsing Algorithm Based on Linguistic Feature

The enhanced POS-based n-gram algorithm splits noun phrases along with the emoji
into n-grams. (One) First, Algorithm 1 identiﬁes the sentiment as accommodating emoji
and text both, emoji only, or text only, (Two) Second, Algorithm 2 In case the sentiment
consists of both Text and Emoji, Algorithm 2, will be used for parsing of sentiment. (Three)
whereas, if the sentiment contains emojis only then Algorithm 3 is used. Whereas, for tree
formation and parsing of sentiment sentence containing text, only then, the research work
in [45] is referred.

Algorithm 1: Identify the sentiment as accommodating emoji and text both, emoji only, or
text only

Input:
Sentiment sentence
Output:
Calling other algorithms based on the content of sentiment sentence.
For each sentiment sentence:
Emoji Unicode Library (sentiment sentence):
Determine the number of emojis in sentence
i.e., EmojiCount.
If (EmojiCount! = 0 && Text also exist)
# Sentiment contains text and emoji both
Algorithm 2 is called
If (EmojiCount! = 0 && Text do not exist)
# Sentiment contains emoji only
Algorithm 3 is called

|  |  |  | 14, 222 | Table 2. POS combinati |
| --- | --- | --- | --- | --- |
|  |  |  | S Combination | s |
|  |  |  | OUN + EMOJI | Noun and |
|  |  |  |  |  |
|  |  |  | + NOUN + EM | Add two OJI |
|  |  |  |  |  |
|  |  |  |  | Adj + Nou |
|  |  |  | + NOUN + EM | OJI the obje |
|  |  |  | TOPWORD + E | a The adje MOJI |
|  |  |  |  | In this pair |
|  |  |  | N + ADJ + EM | OJI standalon |
|  |  |  |  | The stop |

| Calling other algorith | ms based on the content of sentiment |  |
| --- | --- | --- |
| For each sentiment se | ntence: |  |
|  |  |  |
| Emoji Unicode Lib |  |  |
|  |  |  |
| Determine the n | umber of emojis in sentenc | e |
|  |  |  |
| i.e., EmojiCount. | && Text also exist) |  |
| If (EmojiCount! = 0 |  |  |
|  |  |  |
| # Sentiment cont |  |  |
|  | lled |  |
| Algorithm 2 is ca |  |  |
| If (EmojiCount! = 0 | && Text do not exist) |  |
|  |  |  |
| # Sentiment cont |  |  |
|  |  |  |
| Algorithm 3 is ca |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

![page7_img36.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img36.png)

![page7_img92.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img92.png)

![page7_img148.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img148.png)

![page7_img204.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img204.png)

![page7_img260.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img260.png)

![page7_img316.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img316.png)

![page7_img372.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img372.png)

![page7_img428.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img428.png)

![page7_img484.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img484.png)

![page7_img540.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img540.png)

![page7_img596.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img596.png)

![page7_img652.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page7_img652.png)

Information 2023, 14, 222
8 of 18

## Algorithm 2: Sentiment containing both Text and Emoji

Input:
Sentiment sentence containing text and emoji both.
Output:
Parsing of sentiment sentence.
Segregate the NounPhrase, emoji and bigram.

Determine NounPhrases and Emojis in Sentence
For ∀NounPhrase with adjacent Emoji:
Separate the NounPhrase into bigrams and emojis
Concept = ∅;
For ∀NounPhrase:
For ∀bigram with adjacent emoji in the phrase of Noun:
Tag the bigram with POS
If NOUN + EMOJI:
append noun and emoji to Concept
else if NOUN + NOUN + EMOJI:
append noun + noun and emoji to Concepts
else if ADJECTIVE + NOUN + EMOJI:
append noun, adjective + noun, emoji to Concepts
else if ADJECTIVE + STOPWORD + EMOJI:
append adjective and emoji to Concepts
else if NOUN + ADJECTIVE + EMOJI:
append noun, adjective and emoji to Concepts
elseif STOPWORD + NOUN + EMOJI:
append noun and emoji to Concepts
else if STOPWORD + ADJECTIVE + EMOJI:
append emoji and adjective to Concepts
else
append to Concepts: entire bigram and different
concepts of remaining Emojis as isolated.
end
end
end

Algorithm 3: The sentiment containing emojis only.

Input:
Sentiment sentence containing emoji.
Output:
Parsing of sentiment sentence.
Segregate the different emojis.
Determine Emojis in Sentence
For each Emoji:
Split different emojis
For each emoji:
Tag polarity category
Initialize concept to Null;
Append to Concepts: Different concepts of all Emojis.
end
end

## 3.2.1. Algorithm 1

Algorithm 1, stated below, takes the input as sentiment. Then, on the basis of its
content, further algorithms are called that use the n-gram approach for POS.

Information 2023, 14, 222
9 of 18

## 3.2.2. Proposed Algorithm 2: Text with Emoji, POS-Based n-Gram Algorithm

The proposed algorithm 2 shows the steps of processing the parsing of sentiment
containing text and emoji both. As depicted in Figure 2a, it segregates the bigrams, noun
phrase, and emojis from the POS. As an example, shown in Figure 2a,b, the algorithm
generates different concepts from bigrams ˄ emoji. This will help in identifying the
commonsense patterns of text with emoji, which is essential for accurate and complete
knowledge of sentiment or feelings expressed in online sentiments.

## 3.2.3. Proposed Algorithm 3: Emoji Only, POS-Based n-Gram Algorithm

Proposed Algorithm 3 shows the steps of processing the parsing of sentiment containing emojis only. As depicted in Figure 2c it segregates the various combinations of emojis.

## 3.3. Pattern Formation

The proposed pattern formation are linguistic rules utilized for the acquirement of the
sentiment’s polarity based on common sense and affective information consisting of text
and emoji both, text with multiple emojis, emoji only, or text only. The dependency relation
of the proposed pattern provides the ﬂow of sentiments running concept using text and
emojis. With the usage of emojis, the emotional and informational content communicated
by the online user in its sentiment becomes completely different.
Sentence S4:

(a)
It is an excellent approach.

(b)
It is an excellent approach 😏.

(c)
It is an excellent approach 😊.

For example, S4(a) is a positive sentiment, S4(b) is a positive sentence considering text
only i.e., “It is an excellent approach” and neutral if analyzed with complete sentiment

with emoji, i.e., “It is an excellent approach 😏”. Whereas in S4 (a) and S4(c), the sentiment
with text only and text and emoji is positive in both cases.
These examples clearly show the signiﬁcance of the emoji in sentiment polarity computing. Most of the studies of sentiment polarity computing [6,43,45,46] do not consider the
emotional and information content of emoji, without which the affective and commonsense
knowledge-based sentiment polarity detection [47] cannot be acquired completely. Thus,
in sentiment analysis, concept and context play an important role [32].
An example of a sentiment sentence (S5) reﬂects three different psychologies and
moods of wishing ‘Happy Birthday’. S5(a) shows general birthday wishes. S5(b) reﬂects the
feeling and mood of wishing happy birthday along with party mood. S5(c) is a multiword
and emoji expression intending to wish and attend birthday party.
Sentence S5:

(a)
Happy birthday!!!

(b)
## Happy birthday!!! Party 🎂 🕯🎉 🥳

(c)
## Happy birthday!!! Party 🎂 🕯 🎉 🏃😩⌚

Example S5(b) and S5(c) are the same as S5(a) if the sentiment analysis is done with
text only. In actuality, the semantics and psychology of the three sentiments of example
S5 are completely different because of the usage of different type, category, and counts of
emojis.

## 3.4. Polarity Inversion Pattern Rules

The polarity of an emoji also acts as an essential polarity-switching operator in online
sentiment expressions, as discussed in examples S4 and S5. The polarity inversion pattern
rules are mentioned below, and their examples are depicted in Table 3:

i.
Text and Emoji

Information 2023, 14, 222
10 of 18

•
Polarity of both text and emoji is positive, then the overall online sentiment
polarity is also positive;
•
In case text and emoji polarity is negative, then the overall online sentiment
polarity is also negative;
•
In case text and emoji are having opposite polarity, then the overall online
sentiment polarity is neutral.

ii.
Emoji only
•
The sentiment polarity is positive if all emoji’s polarity in sentiment is positive;
•
The sentiment polarity is negative if all emoji’s polarity in sentiment is negative;
•
The sentiment polarity is negative in case the count of emoji with negative
polarity is greater than the count of emoji with positive polarity, or vice versa;
•
The polarity of sentiment is neutral if the count of emojis having a positive
polarity is equal to the count of emojis having a negative polarity.

iii.
In case of multiple emojis in a sentence,
•
Firstly, the semantic pattern of text and the immediate emoji are formed and
their polarity is evaluated as per rule i;
•
Secondly, the polarity of the remaining emojis is determined, in case the positive emojis are more than the negative emojis, then the polarity of the remaining
emoji will be taken as positive or vice versa. In case the count of positive and
negative emojis used are equal, then the polarity of the remaining emojis will
be considered neutral;
•
The ﬁnal polarity of the sentence will be determined based on the commonsense concept and context generated from patterns of text and multiple emojis.
Examples are given in Table 4.

Table 3. Polarity inversion pattern sample.

Example
Text Polarity [45]
Emoji Polarity [23]
Proposed Approach
Polarity

I like it😃.
Pos.
Pos.
Pos.

I like it😒.
Pos.
Neg.
Neutral

I do not like it
Neg.
NA
Neg.

I do not like it😃.
Neg.
Pos.
Neutral

I did not appreciate it😒.
Neg.
Neg.
Neg.

I do not hate it😃.
Pos.
Pos.
Pos.

I do not dislike it😒.
Pos.
Neg.
Neutral

Table 4. The polarity of sentences based on the commonsense concept and context generated from
patterns of text and multiple emojis used in a sentence.

Sentences
Text and Immediate
Emoji Polarity Based on
## Proposed Approach

Remaining Emoji
Polarity [23]
Proposed Approach
Polarity

The guest house is not good to stay 🙂.
Neutral
-
## Neutral + −= Neutral

The guest house is not good to stay 🙂🙂🙂.
Neutral
Pos.
Neutral + Pos. = Pos.

The guest house is not good to stay😏😏.
Neg.
Neg.
Neg. + Neg. = Neg.

The guest house is not good to stay😏🙂.
Neg.
Pos.
Neg. + Pos. = Neutral

Information 2023, 14, 222
11 of 18

## 3.5. Coordinated and Discourse Pattern Rules

The proposed patterns illustrate the articulation of the different members of coordinated and discourse sentiment in a combination of emojis. The adversatives connect two
sentiments of opposite polarity such as but, still, however, otherwise, etc., similar to what
is shown in this particular section. Its examples have been depicted in Table 5.

Table 5. But and adversatives pattern.

Example
Left
Conjunct
[45]

Right
Conjunct
[45]

Text
Polarity
[45]

Emoji
## Polarity [23]

Proposed
Approach

The jewel is lovely but costly😂.
Pos.
Neg.
Pos.
Neg.
Neutral

The jewel is lovely but costly😭.
Pos.
Neg.
Neg.
Neg.
Neg.

The jewel is lovely but not costly😃.
Pos.
Pos.
Pos.
Pos.
Pos.

The jewel is lovely but not costly😢.
Pos.
Pos.
Neg.
Pos.
Neutral

The jewel is lovely but <cough cough cough>😃.
Pos.
undeﬁned
Pos.
Neg.
Neutral

The jewel is lovely but <cough cough cough>😭.
Pos.
undeﬁned
Neg.
Neg.
Neg.

The jewel is not lovely but <cough cough cough>😃.
Neg.
undeﬁned
Pos.
Pos.
Pos.

The jewel is not lovely but <cough cough cough>😭.
Neg.
undeﬁned
Neg.
Pos.
Neutral

<cough cough cough> but the bike is sporty😭.
undeﬁned
Pos.
Neg.
Pos.
Neutral

<cough cough cough> but the bike is sporty😃.
undeﬁned
Pos.
Pos.
Pos.
Pos.

<cough cough cough> but the bike is costly😭.
undeﬁned
Neg.
Neg.
Neg.
Neg.

<cough cough cough> but the bike is costly😃.
undeﬁned
Neg.
Pos.
Neg.
Neutral

The but and adversatives: in adversative sentiments, the second part of the sentence
dominates the sentiment of the ﬁrst part of the sentence using “But”. The various possibilities of the commonsense-based pattern for “but” are depicted in Table 5. The overall
polarity of the pattern is dependent on the second part of the adversative and the emoji.
Some of the selective rules are enumerated below:

i.
The polarity of both the adversative right member and the emoji is positive, then
the overall polarity will be positive, same is vice versa with negative polarity;
ii.
The polarity of the adversative right member and emoji are opposite, then the
overall polarity will be neutral;
iii.
The polarity of the adversative right member is undeﬁned, then the polarity of the
left member is inverted, then in this case:
•
The inverted polarity of the left member and emoji polarity are negative, then
polarity will be negative;
•
The inverted polarity of the left member and emoji polarity are opposite, then
the pattern polarity will be neutral.

iv.
The polarity of the adversative left member is undeﬁned, then in this case:
•
The polarity of both the right member and the emoji are positive, then the
pattern polarity will be positive;
•
The polarity of both the right member and the emoji are negative, then text
and emoji pattern polarity will be negative;
•
The polarity of the right member and the emoji are opposite, then the text and
emoji pattern polarity will be neutral.

v.
In case, of multiple emojis in a sentence:

Information 2023, 14, 222
12 of 18

•
Firstly, the sentic pattern of text and the immediate emoji are formed, and their
polarity is evaluated as per rules from i–iv;
•
Secondly, the polarity of the remaining emojis is determined. In case the positive emojis are more than negative emojis, then the polarity of the remaining
emoji will be taken as positive or vice versa;
•
In case the count of opposite polarity emojis used are equivalent, then the
polarity of the remaining emojis will be considered neutral.
•
The ﬁnal polarity of the sentence will be determined based on the commonsense concept and context generated from the suggested patterns of text and
multiple emojis. Examples are depicted in Table 6;

Table 6. The polarity of sentence based on the commonsense concept and context generated from
proposed patterns of multiple emojis used in sentence.

Sentences
Text and Immediate
Emoji Polarity Based on
## Proposed Approach

Remaining Emoji
Polarity [23]
Proposed Approach
Polarity

The guest house is good to stay but the room

size is small 🙂🙂🙂.
Neutral
Pos.
Neutral + Pos. = Pos.

## I wish you very Happy Anniversary but

without party😟😟??
Neg.
Neg.
Neg. + Neg. = Neg.

## She dances very beautifully but her dress was

also awesome😜😜.
Neutral
Neg.
Neutral + Neg. = Neg.

She dances very beautifully but😜😜.
Neg.
Neg.
Neg. + Neg. = Neg.

## 3.6. Emoji, Text, and Final Polarity Evaluation

The preprocessed tweets are divided into text and emoji parts. The common sensebased polarity of text is evaluated using the existing SenticNet [45] approach. The polarity
of the emojis used in tweets are evaluated by using a CLDR-based emoji score and polaritydetermining approach [23].
The ﬁnal sentiment polarity is evaluated by combining the role of proposed patterns of
text and emoji, text and multiple emojis, emoji only, and text only, generated using the rules
and techniques elaborated in Sections 3.1–3.3 with the calculated text and emoji polarity
depicted above in Section 3.6.
The detailed calculation of sentiment polarity using the proposed approach and using
state of the art are represented in Tables 3–6.

## 4. Experiment, Results, and Discussion

The dataset of tweets (sentences, Si) downloaded using Twitter (https://pypi.python.
org/pypi/tweepy/2020/09 accessed on 2 February 2023) API (application programming
interface) consists of approximately 168,548 tweets posted by the 650 different top most
followed multitudinal personages across the world [3,24,25].
The downloaded user-generated dataset is preprocessed to eliminate the special
characters, pictures, slang, etc. The emojis and their Unicode are referenced through
the emojipedia (https://emojipedia.org/emoji-14.0/2021/11), whereas libraries of sentic
net (https://pypi.org/project/senticnet/2021/07) and VADER (https://pypi.org/project/
vaderSentiment/2021/02 accessed on 2 February 2023) [48] are also used. From the downloaded dataset, 30% is used as a training dataset and 70% is taken as a testing dataset. The
training dataset is manually annotated for evaluating the performance.
We have considered a dataset of 4158 emojis [23] and 1281 emoticons (https://pc.net/
emoticons/browse/a/2021/01 accessed on 2 February 2023) in the research work.

Information 2023, 14, 222
13 of 18

Tables 3–7 also reﬂect the consistency of the proposed framework, irrespective of
domain.

Table 7. Proposed work results for various ensemble of the linguistic features available in sentiment
sentences.

Sentences
Text Polarity [45]
Proposed
Approach
Accurate
Polarity

The guest house is good to stay but the

room size is small 🙂🙂🙂.
Neg.
Pos.
Pos.

## I wish you very Happy Anniversary!!! Party

😟😟??
Pos.
Neg.
Neg.

She dances very beautifully 😈 😈 😈 .
Pos.
Neg.
Neg.

The idea was not good🤪
Neg.
Neutral
Neutral

Consider an example in Table 3; I like it😒. When analyzed for text only, it gives
positive sentiment polarity [49], whereas, on analyzing the emoji only it gives negative
polarity. However, according to the proposed approach to analyzing both the text and
emoji part of the sentence, the sentiment polarity acquired is neutral, which is the more
accurate polarity of the sentence.
As mentioned in Table 4, the accurate polarity of the sentence, The guest house is not

good to stay😏 🙂, is neutral. Whereas the polarity of text and emoji, The guest house is not

good to stay😏, in combination as per rule of the patterns of polarity inversion of text and

emoji gives polarity negative and🙂, emoji only, is of polarity positive.
Consider the example of <cough cough cough> but the bike is sporty😭 , mentioned in
Table 5. The left part of but is undeﬁned, whereas, then the polarity of text, the bike is sporty,
i.e., the right conjunct is of positive polarity and the polarity of😭 is negative. Thus, the
polarity of the sentence, as per the proposed approach, is neutral, which is also the correct
polarity of the sentence.
Table 7 declares the correct polarity evaluation by the proposed approach following
the polarity inversion, coordinated and discourse structures, and distinctive proposed
patterns with multiple emojis.
The performance of our proposed model is evaluated based on four evaluation parameters [25,50–52].
Accuracy: the ratio of correctly-predicted samples to the total observations.

Accuracy =
# TP + TN
# TP + FP + FN + TN

Precision: it is the ratio of truly-positive samples to the complete true predicted cases.

Precision =
TP
# TP + FP

Recall: it refers to the ratio of correctly-predicted positive samples to the actual
positives.

recall =
TP
# TP + FP

The F1 score is the evaluation of the harmonic mean between precision and recall.

## F1 score = 2 ∗precision ∗recall

precision + recall

|  | th |
| --- | --- |

Information 2023, 14, 222
14 of 18

Table 8 depicts the machine learning classiﬁers [53–60] relative to the performance of
the proposed approach by using accuracy, recall, and F score. A sequence of experiments
is performed with each of the proposed linguistic features to analyze the importance of
the respective feature in improving the performance of the classiﬁer. The SVM classiﬁer
has shown the highest accuracy of 82.8 among the three classiﬁers with all three linguistic
features.

Table 8. The experiment results of various ML classiﬁers for different combinations of linguistic
features.

ML Classiﬁer
Linguistic Features
Recall
F-Score
Accuracy

Text only, emoji only, and combination of text with emoji
(Proposed approach: All features)
75.5
79.8
82.8

SVM

Text only
74.9
77.4
80.4

Emoji only
64.3
68.6
69.3

Text only, emoji only and combination of text with emoji
(Proposed approach: All features)
69.9
72.4
75.4

## Naïve Bayes

Text only
67.7
70.4
71.4

Emoji only
60.3
67.5
69.2

Text only, emoji only and combination of text with emoji
(Proposed approach: All features)
73
76.6
79.3

## Decision Tree

Text only
71.3
74.9
79.3

Emoji only
69.5
70.6
70.8

Figure 3 depicts the performance comparison of the three classiﬁers using the proposed
approach. Figures 4–6 depict the values of the F-score, recall, and accuracy for the linguistic
patterns of text and emoji, text and multiple emoji, emoji only, and text only using ML
classiﬁers SVM, Naïve Bayes, and decision tree, respectively.
Table 9 declares the accuracy of the proposed research work above the present approach based on sentiment sentences equipped with ‘but’ adversatives and polarity inversion sentiment sentences with different combinations of emojis using the SVM classiﬁer.
Along with text and multiple emojis, the proposed approach shows a 90.78% accuracy for
but and adversatives, whereas an accuracy of 92.12% for polarity inversion using the SVM
classiﬁer.

Figure 3. Classiﬁers relative performance for the proposed approach.

| Decision Tree | Text only 71.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 74.9 79.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Em | oji |  | onl | y |  |  |  |  |  |  |  |  | 69. | 5 | 70.6 70.8 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Figure 3 depicts the performance comparison of the th approach. Figures 4–6 depict the values of the F-score, rec patterns of text and emoji, text and multiple emoji, em classifiers SVM, Naïve Bayes, and decision tree, respectiv Table 9 declares the accuracy of the proposed rese proach based on sentiment sentences equipped with ‘but sion sentiment sentences with different combinations of Along with text and multiple emojis, the proposed appro |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ree classifiers using the proposed all, and accuracy for the linguistic oji only, and text only using ML ely. arch work above the present ap ’ adversatives and polarity inver emojis using the SVM classifier ach shows a 90.78% accuracy for |
|  |  | dep |  | icts | th | e perf | or | ma | n | ce | comp | ari | son | of |  |  |
|  |  | ure |  | s 4 | –6 | depict | th | e v | a | lu | es of t | he | F-s | cor |  |  |

![page14_img1.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page14_img1.png)

Information 2023, 14, 222
15 of 18

Figure 4. Different linguistic features relative performance for SVM classiﬁer.

80

60

Values

40

20

0

All features
Text only
## Emoji only

## Linguistic Features

Recall
F-score
Accuracy

Figure 5. Relative performance of different linguistic features for Naïve Bayes classiﬁer.

Figure 6. Different linguistic feature's relative performance for decision tree classiﬁer.

Table 9. The state-of-the-art comparison with the proposed approach of text and multiple emoji
patterns of ‘but’ and polarity inversion sentiment sentences using the SVM classiﬁer.

Approach
But and Adversatives Accuracy
## Polarity Inversion Accuracy

Poria et al. [45]
87.9%
88.6%

Socher et al. [7]
56.6%
64.4%

Proposed approach
90.78%
92.12%

|  | Figure 5. | Relative performance of different linguistic features for Naïve Bayes classifier. |
| --- | --- | --- |

![page15_img1.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page15_img1.png)

![page15_img2.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page15_img2.png)

![page15_img3.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page15_img3.png)

![page15_img4.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page15_img4.png)

![page15_img5.png](Emoji%20Text%20%20and%20Sentiment%20Polarity%20Detection%20Using%20Natural%20Language%20Processing_images/page15_img5.png)

Information 2023, 14, 222
16 of 18

Thus, the proposed approach helps in determining the correct polarity of the sentences
with text only, text and emoji only, emoji only, and multiple emoji. It also determines the
correct polarity with the pattern of coordinated, discourse, and polarity inversion structures
of online natural-language sentiments. The correct polarity detection of an online sentiment
helps in evaluating product analysis, market competitor research, mental wellbeing, etc.

## 5. Conclusions and Future Scope

The proposed sentiment polarity computing is an approach that provides a conceptual and affective level of NLP while incorporating the role of emojis. It conglomerates
commonsense computing with a contextual perception of concept ﬂow within a sentence
aggregating the role of its linguistic features.
As is clear from the examples given in Tables 3–7 usage of one or multiple emojis with
or without text plays a very crucial role in sentiment prediction. The results clearly indicate
that the sentiment polarity evaluation, along with text and emoji, can invert the polarity
results predicted with the text only. Thus, the proposed approach plays a crucial role in
generating a correct and accurate sentiment knowledge of the expressions. Table 9 clearly
indicates the signiﬁcance of the proposed approach over the existing approaches with the
use of the proposed linguistic pattern-rule-based approach coupled with complex ‘but’ and
‘polarity inversion’ rules signiﬁcantly improved the performance of approaches.
The limitations of the proposed approach lie in that efforts can be built to do the
same kind of analysis utilizing more distinctive and complex sentences and online content
such as photos, graphic interchange format (GIF), video recordings, etc. The current
research work provides further recommendation to do a detailed generation of dependency
rules using perceptive, cognitive, and other computing models as well. In addition, the
future direction aims to incorporate the available domain knowledge to better interpret the
linguistic patterns of text.

Author Contributions: Conceptualization, S.G. and A.S.; methodology, S.G.; formal analysis, V.K.;
resources, V.K.; data curation, S.G.; writing—original draft, S.G.; writing—review & editing, A.S. and
V.K.; supervision, A.S. All authors have read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Institutional Review Board Statement: Not applicable.

Data Availability Statement: No new data were created and data is available on demand due to
privacy or ethical restrictions.

Conﬂicts of Interest: The authors declare no conﬂict of interest.

References

1.
Garg, K. Sentiment analysis of Indian PM’s “Mann Ki Baat”. Int. J. Inf. Technol. 2020, 12, 37–48. [CrossRef]
2.
Fernández-Gavilanes, M.; Juncal-Martínez, J.; García-Méndez, S.; Costa-Montenegro, E.; González-Castaño, F.J. Creating emoji
lexica from unsupervised sentiment analysis of their descriptions. Expert Syst. Appl. 2018, 103, 74–91. [CrossRef]
3.
Gupta, S.; Singh, A.; Ranjan, J. An Online Document Emoji-Based Classiﬁcation Using Twitter Dataset. In Proceedings of Data
Analytics and Management; Springer: Singapore, 2022; pp. 409–417.
4.
Li, X.; Wang, B.; Li, L.; Gao, Z.; Liu, Q.; Xu, H.; Fang, L. Deep2s: Improving Aspect Extraction in Opinion Mining with Deep
Semantic Representation. IEEE Access 2020, 8, 104026–104038. [CrossRef]
5.
Vilares, D.; Peng, H.; Satapathy, R.; Cambria, E. Babelsenticnet: A commonsense reasoning framework for multilingual sentiment
analysis. In 2018 IEEE Symposium Series on Computational Intelligence (SSCI), Bangalore, India, 18–21 November 2018; IEEE: Piscataway,
NJ, USA, 2018; pp. 1292–1298.
6.
Lo, S.L.; Cambria, E.; Chiong, R.; Cornforth, D. A multilingual semi-supervised approach in deriving Singlish sentic patterns for
polarity detection. Knowl. Based Syst. 2016, 105, 236–247. [CrossRef]
7.
Socher, R.; Huval, B.; Manning, C.D.; Ng, A.Y. Semantic compositionality through recursive matrix-vector spaces. In Proceedings
of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language
Learning, Jeju Island, Republic of Korea, 12–14 July 2012; Association for Computational Linguistics: Stroudsburg, PA, USA, 2012;
pp. 1201–1211.

Information 2023, 14, 222
17 of 18

8.
Bhandari, A.; Kumar, V.; Thien Huong, P.T.; Thanh, D.N. Sentiment analysis of COVID-19 tweets: Leveraging stacked word
embedding representation for identifying distinct classes within a sentiment. In Artiﬁcial Intelligence in Data and Big Data Processing,
Proceedings of ICABDE 2021, Ho Chi Minh, Vietnam, 18–19 December 2021; Springer International Publishing: Cham, Switzerland,
2022; pp. 341–352.
9.
Saha, S.; Kumar, K.A. Emoji Prediction Using Emerging Machine Learning Classiﬁers for Text-based Communication. J. Math. Sci.
Comput. 2022, 1, 37–43. [CrossRef]
10.
Gupta, S.; Singh, A. A Vocabulary-Based Framework for Sentiment Analysis. In Computational Intelligence, Proceedings of the
2nd International Conference on Information Technology InCITe, Noida, India, 3–4 March 2022; Springer Nature: Singapore, 2023; pp.
507–515.
11.
Chan, J.Y.L.; Bea, K.T.; Leow, S.M.H.; Phoong, S.W.; Cheng, W.K. State of the art: A review of sentiment analysis based on
sequential transfer learning. Artif. Intell. Rev. 2023, 56, 749–780. [CrossRef]
12.
Chang, V.; Liu, L.; Xu, Q.; Li, T.; Hsu, C.H. An improved model for sentiment analysis on luxury hotel review. Expert Syst. 2023,
40, e12580. [CrossRef]
13.
Eke, C.I.; Norman, A.A.; Shuib, L.; Nweke, H.F. Sarcasm identiﬁcation in textual data: Systematic review, research challenges and
open directions. Artif. Intell. Rev. 2020, 53, 4215–4258. [CrossRef]
14.
Dey, A.; Jenamani, M.; Thakkar, J.J. Senti-N-Gram: An n-gram lexicon for sentiment analysis. Expert Syst. Appl. 2018, 103, 92–105.
[CrossRef]
15.
Zobeidi, S.; Naderan, M.; Alavi, S.E. Opinion mining in Persian language using a hybrid feature extraction approach based on
convolutional neural network. Multimed. Tools Appl. 2019, 78, 32357–32378. [CrossRef]
16.
Tripathy, A.; Anand, A.; Rath, S.K. Document-level sentiment classiﬁcation using hybrid machine learning approach. Knowl. Inf.
Syst. 2017, 53, 805–831. [CrossRef]
17.
Kathuria, A.; Gupta, A.; Singla, R.K. AOH-Senti: Aspect-Oriented Hybrid Approach to Sentiment Analysis of Students’ Feedback.
SN Comput. Sci. 2023, 4, 152. [CrossRef]
18.
Mujawar, S.S.; Bhaladhare, P.R. An Aspect based Multi-label Sentiment Analysis using Improved BERT System. Int. J. Intell. Syst.
Appl. Eng. 2023, 11, 228–235.
19.
Xuyang, W.A.N.G.; Shuai, D.O.N.G.; Jie, S.H.I. Multimodal Sentiment Analysis with Composite Hierarchical Fusion. J. Front.
Comput. Sci. Technol. 2023, 17, 198.
20.
Wei, Z.; Chen, Y.; Zhao, Q.; Zhang, P.; Zhou, L.; Ren, J.; Piao, Y.; Qiu, B.; Xie, X.; Wang, S.; et al. Implicit Perception of Differences
between NLP-Produced and Human-Produced Language in the Mentalizing Network. Adv. Sci. 2023, 10, 2203990. [CrossRef]
21.
Chen, L.C.; Chang, K.H.; Yang, S.C.; Chen, S.C. A Corpus-Based Word Classiﬁcation Method for Detecting Difﬁculty Level of
English Proﬁciency Tests. Appl. Sci. 2023, 13, 1699. [CrossRef]
22.
Rethmeier, N.; Augenstein, I. A Primer on Contrastive Pretraining in Language Processing: Methods, Lessons Learned, and
Perspectives. ACM Comput. Surv. 2023, 55, 1–17. [CrossRef]
23.
Gupta, S.; Singh, A.; Ranjan, J. Emoji Score and Polarity Evaluation Using CLDR Short Name and Expression Sentiment. In
International Conference on Soft Computing and Pattern Recognition; Springer: Cham, Switzerland, 2020; pp. 1009–1016.
24.
Gupta, S.; Singh, A.; Ranjan, J. Online Document Content and Emoji-Based Classiﬁcation Understanding from Normal to
Pandemic COVID-19. Int. J. Perform. Eng. 2022, 18, 710–719. [CrossRef]
25.
Gupta, S.; Singh, A.; Ranjan, J. Multimodal, multiview and multitasking depression detection framework endorsed with auxiliary
sentiment polarity and emotion detection. Int. J. Syst. Assur. Eng. Manag. 2023, 1–16. [CrossRef]
26.
Dashtipour, K.; Poria, S.; Hussain, A.; Cambria, E.; Hawalah, A.Y.; Gelbukh, A.; Zhou, Q. Multilingual sentiment analysis: State
of the art and independent comparison of techniques. Cogn. Comput. 2016, 8, 757–771. [CrossRef]
27.
Cambria, E.; Hussain, A. Sentic Computing: A Common-Sense-Based Framework for Concept-Level Sentiment Analysis; Springer:
Berlin/Heidelberg, Germany, 2015; ISBN 978-3-319-23654-4.
28.
Cambria, E. Affective computing and sentiment analysis. IEEE Intell. Syst. 2016, 31, 102–107. [CrossRef]
29.
Martis, E.; Deo, R.; Rastogi, S.; Chhaparia, K.; Biwalkar, A. A Proposed System for Understanding the Consumer Opinion of a
Product Using Sentiment Analysis. In Advances in Intelligent Systems and Computing, Proceedings of the 2nd International Conference
on Sentiment Analysis and Deep Learning, Bangkok, Thailand, 16–17 June 2022; Springer Nature: Singapore, 2023; pp. 555–568.
30.
Benrouba, F.; Boudour, R. Emotional sentiment analysis of social media content for mental health safety. Soc. Netw. Anal. Min.
2023, 13, 17. [CrossRef]
31.
Poria, S.; Chaturvedi, I.; Cambria, E.; Bisio, F. Sentic LDA: Improving on LDA with semantic similarity for aspect-based sentiment
analysis. In 2016 International Joint Conference on Neural Networks (IJCNN), Vancouver, BC, Canada, 24–29 July 2016; IEEE: Piscataway,
NJ, USA, 2016; pp. 4465–4473.
32.
Dragoni, M.; Poria, S.; Cambria, E. OntoSenticNet: A commonsense ontology for sentiment analysis. IEEE Intell. Syst. 2018, 33,
77–85. [CrossRef]
33.
Ma, Y.; Peng, H.; Cambria, E. Targeted aspect-based sentiment analysis via embedding commonsense knowledge into an attentive
LSTM. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, New Orleans, LA, USA, 2–7 February 2018; Volume 32.
34.
Fedushko, S.; Ustyianovych, T. E-commerce customers behavior research using cohort analysis: A case study of COVID-19. J.
Open Innov. Technol. Mark. Complex. 2022, 8, 12. [CrossRef]

Information 2023, 14, 222
18 of 18

35.
Khattak, A.; Asghar, M.Z.; Ishaq, Z.; Bangyal, W.H.; Hameed, I.A. Enhanced concept-level sentiment analysis system with
expanded ontological relations for efﬁcient classiﬁcation of user reviews. Egypt. Inform. J. 2021, 22, 455–471. [CrossRef]
36.
Behera, R.K.; Jena, M.; Rath, S.K.; Misra, S. Co-LSTM: Convolutional LSTM model for sentiment analysis in social big data. Inf.
Process. Manag. 2021, 58, 102435. [CrossRef]
37.
Liao, J.; Wang, M.; Chen, X.; Wang, S.; Zhang, K. Dynamic commonsense knowledge fused method for Chinese implicit sentiment
analysis. Inf. Process. Manag. 2022, 59, 102934. [CrossRef]
38.
Pradhan, A.; Senapati, M.R.; Sahu, P.K. Improving sentiment analysis with learning concepts from concept, patterns lexicons and
negations. Ain Shams Eng. J. 2022, 13, 101559. [CrossRef]
39.
Consoli, S.; Barbaglia, L.; Manzan, S. Fine-grained, aspect-based sentiment analysis on economic and ﬁnancial lexicon. Knowl.
Based Syst. 2022, 247, 108781. [CrossRef]
40.
Parsafard, P.; Veisi, H.; Aﬂaki, N.; Mirzaei, S. Text Classiﬁcation based on Discriminative-Semantic Features and Variance of
Fuzzy Similarity. Int. J. Intell. Syst. Appl. 2022, 2, 26–39. [CrossRef]
41.
Almutiri, T.; Nadeem, F. Markov Models Applications in Natural Language Processing: A Survey. Int. J. Inf. Technol. Comput. Sci.
2022, 2, 1–16. [CrossRef]
42.
Neel, L.A.; McKechnie, J.G.; Robus, C.M.; Hand, C.J. Emoji alter the perception of emotion in affectively neutral text messages. J.
Nonverbal Behav. 2023, 47, 83–97. [CrossRef]
43.
Cambria, E.; Havasi, C.; Hussain, A. Senticnet 2: A semantic and affective resource for opinion mining and sentiment analysis. In
Proceedings of the Twenty-Fifth International Florida Artiﬁcial Intelligence Research Society Conference, Marco Island, FL, USA,
23–25 May 2012; pp. 202–207.
44.
Cambria, E.; Li, Y.; Xing, F.Z.; Poria, S.; Kwok, K. Senticnet 6: Ensemble application of symbolic and subsymbolic ai for sentiment
analysis. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management CIKM, Online,
19–23 October 2020.
45.
Poria, S.; Hussain, A.; Cambria, E. Sentic Patterns: Sentiment Data Flow Analysis by Means of Dynamic Linguistic Patterns. In
Multimodal Sentiment Analysis; Springer: Cham, Switzerland, 2018; pp. 117–151.
46.
Susanto, Y.; Livingstone, A.; Ng, B.C.; Cambria, E. The hourglass model revisited. IEEE Intell. Syst. 2020, 35, 96–102. [CrossRef]
47.
Divate, M.S. Sentiment analysis of Marathi news using LSTM. Int. J. Inf. Technol. 2021, 13, 2069–2074. [CrossRef]
48.
Hutto, C.; Gilbert, E. Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the
International AAAI Conference on Web and Social Media, Ann Arbor, MI, USA, 1–4 June 2014; Volume 8, pp. 216–225.
49.
Susanto, Y.; Cambria, E.; Ng, B.C.; Hussain, A. Ten years of sentic computing. Cogn. Comput. 2022, 14, 5–23. [CrossRef]
50.
Kumar, V.; Recupero, D.R.; Helaoui, R.; Riboni, D. K-LM: Knowledge Augmenting in Language Models within the Scholarly
Domain. IEEE Access 2022, 10, 91802–91815. [CrossRef]
51.
Tan, L.; Tan, O.K.; Sze, C.C.; Goh, W.W.B. Emotional Variance Analysis: A new sentiment analysis feature set for Artiﬁcial
Intelligence and Machine Learning applications. PLoS ONE 2023, 18, e0274299. [CrossRef]
52.
Mahalleh, E.R.; Gharehchopogh, F.S. An automatic text summarization based on valuable sentences selection. Int. J. Inf. Technol.
2022, 14, 2963–2969. [CrossRef]
53.
Agarwal, M.; Singh, A.; Arjaria, S.; Sinha, A.; Gupta, S. ToLeD: Tomato leaf disease detection using convolution neural network.
Procedia Comput. Sci. 2020, 167, 293–301. [CrossRef]
54.
Kumar, V.; Recupero, D.R.; Riboni, D.; Helaoui, R. Ensembling classical machine learning and deep learning approaches for
morbidity identiﬁcation from clinical notes. IEEE Access 2020, 9, 7107–7126. [CrossRef]
55.
Gopi, A.P.; Jyothi RN, S.; Narayana, V.L.; Sandeep, K.S. Classiﬁcation of tweets data based on polarity using improved RBF kernel
of SVM. Int. J. Inf. Technol. 2023, 15, 965–980. [CrossRef]
56.
Sinha, A.; Gupta, S.K.; Tiwari, A.; Chaturvedi, A. Deep Learning: An Overview and Innovative Approach in Machine Learning.
In Hidden Link Prediction in Stochastic Social Networks; IGI Global: Hershey, PA, USA, 2019; pp. 108–134.
57.
Singh, S.K.; Khamparia, A.; Sinha, A. Explainable Machine Learning Model for Diagnosis of Parkinson Disorder. In Biomedical
Data Analysis and Processing Using Explainable (XAI) and Responsive Artiﬁcial Intelligence (RAI); Springer: Singapore, 2022; pp. 33–41.
58.
Singh, S.K.; Sinha, A.; Yadav, S. Performance Analysis of Machine Learning Algorithms for Erythemato-Squamous Diseases
Classiﬁcation. In 2022 IEEE International Conference on Distributed Computing and Electrical Circuits and Electronics (ICDCECE),
Ballari, India, 23–24 April 2022; IEEE: Piscataway, NJ, USA, 2022; pp. 1–6.
59.
Gupta, S.; Bisht, S.; Gupta, S. Sentiment Analysis of an Online Sentiment with Text and Slang Using Lexicon Approach. In Smart
Computing Techniques and Applications; Springer: Singapore, 2021; pp. 95–105.
60.
Khanday, A.M.U.D.; Rabani, S.T.; Khan, Q.R.; Rouf, N.; Mohi Ud Din, M. Machine learning based approaches for detecting
COVID-19 using clinical text data. Int. J. Inf. Technol. 2020, 12, 731–739. [CrossRef] [PubMed]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual
author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to
people or property resulting from any ideas, methods, instructions or products referred to in the content.
