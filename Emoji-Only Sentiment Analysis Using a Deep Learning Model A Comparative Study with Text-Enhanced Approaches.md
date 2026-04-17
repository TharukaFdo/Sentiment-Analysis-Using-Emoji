# Emoji-Only Sentiment Analysis Using a Deep Learning Model: A Comparative Study with Text-Enhanced Approaches

# INTE 41323

E m o j i - O n l y  S e n t i m e n t  A n a l y s i s

U s i n g  a  D e e p  L e a r n i n g  M o d e l :

# A  C O M P A R A T I V E  S T U D Y  W I T H
# T E X T - E N H A N C E D  A P P R O A C H E S

## Group 11

IM/2021/011 - Tharushika
IM/2021/020 - Pasan
IM/2021/058 - Dhanitha
IM/2021/062 - Hasindu

![page1_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page1_img1.png)

![page1_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page1_img2.png)

i n t r o d u c t i o n
1

## Sentiment analysis extracts emotions from text

data

## Widely used in social media and online platforms

## Communication is now multimodal (text + emojis)

## Emojis carry strong emotional meaning

Can complement or replace text

| i n t r 1 |  |  |  |  |  | o d u c t i o n |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

![page2_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page2_img1.png)

![page2_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page2_img2.png)

![page2_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page2_img3.png)

![page2_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page2_img4.png)

Problem Statement
2

# TRADITIONAL MODELS HEAVILY RELY ON TEXT BASED ANALYSIS

# IGNORE EMOJIS IS LOSS OF EMOTIONAL CONTEXT

# EMOJIS CAN REINFORCE SENTIMENT, CHANGE MEANING &
# INVERT POLARITY (ESPECIALLY IN SARCASTIC EXPRESSIONS)

| Problem Statement 2 TRADITIONAL MODELS HEAVILY R IGNORE EMOJIS IS LOSS OF EMOT EMOJIS CAN REINFORCE SENTIM INVERT POLARITY (ESPECIALLY I | ELY ON TEXT BASED ANALYSIS IONAL CONTEXT ENT, CHANGE MEANING & N SARCASTIC EXPRESSIONS) |
| --- | --- |

|  | TRADITIONAL MODELS HEAVILY R | ELY ON TEXT BASED ANALYSIS |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  | IGNORE EMOJIS IS LOSS OF EMOT | IONAL CONTEXT |  |

|  | EMOJIS CAN REINFORCE SENTIM INVERT POLARITY (ESPECIALLY I | ENT, CHANGE MEANING & N SARCASTIC EXPRESSIONS) |  |
| --- | --- | --- | --- |

![page3_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page3_img1.png)

![page3_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page3_img2.png)

![page3_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page3_img3.png)

![page3_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page3_img4.png)

3
## Existing Approaches

## Unimodal (Text-Only) Frameworks

Traditional lexicon-based models (VADER, SenticNet) and standard
machine learning (SVM, Naive Bayes).
Focus on extracting sentiment from textual syntax and semantics
alone.

## Multimodal Fusion (Text + Emoji) Frameworks

Early and late fusion techniques that concatenate text and emoji
features.

Pre-existing work focuses almost exclusively on adding emojis to text to
boost or intensify performance rather than investigating emojis as a
primary source.

| Existing Approaches 3 Unimodal (Text-Only) Frameworks Traditional lexicon-based mode machine learning (SVM, Naive B Focus on extracting sentiment f alone. Multimodal Fusion (Text + Emoji) Fr Early and late fusion technique features. Pre-existing work focuses almost e boost or intensify performance rat primary source. | ls (VADER, SenticNet) and standard ayes). rom textual syntax and semantics ameworks s that concatenate text and emoji xclusively on adding emojis to text to her than investigating emojis as a |
| --- | --- |

![page4_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page4_img1.png)

![page4_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page4_img2.png)

![page4_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page4_img3.png)

![page4_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page4_img4.png)

## Comparison of existing approaches

4

Research Name
(Source)
What They Have Done
Technology Stack
Their Output
Limitations
Their Gap with Your
Research

Relied on manually
defined linguistic
rules and POS
parsing, which
struggle with the
stylized and evolving
nature of digital slang.

While they rely on
traditional ML and manual
rules, we use an advanced
BERT-LSTM model to
automatically capture deep
bidirectional contextual
relationships [User Context].

Emoji, Text, and Sentiment
Polarity Detection Using
Natural Language
Processing (Gupta et al.,
2023)

Evaluated sentiment
polarity across three
distinct pipelines: text and
emoji combined, emoji only,
and text only.

SVM, Decision Tree, and
Naive Bayes classifiers
utilizing POS-based n-gram
linguistic features and a
sentiment dictionary.

Achieved a maximum
accuracy of 82.8% using
the SVM classifier with
all linguistic features.

Emoji-only accuracy
(0.74) was competitive
with text-only (0.80)
but was 32 times faster
to train and used 40x
less data.

Focused exclusively
on the financial
domain (StockTwits)
and lacked a
combined (text +
emoji) third pipeline.

We are expanding their
"emoji as proxy" findings by
adding a combined pipeline
and testing the model's
reliability on complex social
media sarcasm

Investigated if emojis alone
could serve as a reliable
proxy for sentiment by
comparing emoji-only vs.
text-only pipelines

The Role of Emojis in
Sentiment Analysis of
Financial Microblogs
(Mahrous et al., 2023)

BiLSTM and Logistic
Regression models using a
Twitter-RoBERTa base
sentiment tokenizer.

Primarily focused on
Chinese microblogs
(Sina Weibo) and
viewed emoji-only
results as a baseline
rather than a
standalone research
focus.

We are taking their proof of
emoji dominance and
applying a modern BERT-
LSTM architecture to see if
emojis carry the "correct"
sentiment in sarcastic
## English comments

EA-Bi-LSTM (Emoji
Attention Bi-LSTM) with
custom Skip-Gram
embeddings for words and
emojis trained on 3.5M
posts.

Found that emoji-only
features (85.76%)
significantly
outperformed text-only
features (69.78%) in
sentiment prediction.

Developed a deep learning
model to capture the impact
of emojis on text sentiment,
including an emoji-only
baseline for comparison.

Emoji-Based Sentiment
Analysis Using Attention
Networks (Lou et al., 2020)

| Research Name (Source) | What They Have Done | Technology Stack | Their Output | Limitations | Their Gap with Your Research |
| --- | --- | --- | --- | --- | --- |
| Emoji, Text, and Sentiment Polarity Detection Using Natural Language Processing (Gupta et al., 2023) | Evaluated sentiment polarity across three distinct pipelines: text and emoji combined, emoji only, and text only. | SVM, Decision Tree, and Naive Bayes classifiers utilizing POS-based n-gram linguistic features and a sentiment dictionary. | Achieved a maximum accuracy of 82.8% using the SVM classifier with all linguistic features. | Relied on manually defined linguistic rules and POS parsing, which struggle with the stylized and evolving nature of digital slang. | While they rely on traditional ML and manual rules, we use an advanced BERT-LSTM model to automatically capture deep bidirectional contextual relationships [User Context]. |
| The Role of Emojis in Sentiment Analysis of Financial Microblogs (Mahrous et al., 2023) | Investigated if emojis alone could serve as a reliable proxy for sentiment by comparing emoji-only vs. text-only pipelines | BiLSTM and Logistic Regression models using a Twitter-RoBERTa base sentiment tokenizer. | Emoji-only accuracy (0.74) was competitive with text-only (0.80) but was 32 times faster to train and used 40x less data. | Focused exclusively on the financial domain (StockTwits) and lacked a combined (text + emoji) third pipeline. | We are expanding their "emoji as proxy" findings by adding a combined pipeline and testing the model's reliability on complex social media sarcasm |
| Emoji-Based Sentiment Analysis Using Attention Networks (Lou et al., 2020) | Developed a deep learning model to capture the impact of emojis on text sentiment, including an emoji-only baseline for comparison. | EA-Bi-LSTM (Emoji Attention Bi-LSTM) with custom Skip-Gram embeddings for words and emojis trained on 3.5M posts. | Found that emoji-only features (85.76%) significantly outperformed text-only features (69.78%) in sentiment prediction. | Primarily focused on Chinese microblogs (Sina Weibo) and viewed emoji-only results as a baseline rather than a standalone research focus. | We are taking their proof of emoji dominance and applying a modern BERT- LSTM architecture to see if emojis carry the "correct" sentiment in sarcastic English comments |

![page5_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page5_img1.png)

![page5_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page5_img2.png)

![page5_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page5_img3.png)

research gap

5

Some models reach high accuracy for combined data but reduced
accuracy for emoji-based pipelines.

Research is often restricted to one area and lacks generalized
comparative frameworks that test these models across varied,
real-world social communication.

While emojis can change the meaning, there is a lack of systematic
evaluation of emoji-only pipelines versus text-enhanced ones in
emoji rich sarcastic contexts.

![page6_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page6_img1.png)

![page6_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page6_img2.png)

![page6_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page6_img3.png)

![page6_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page6_img4.png)

## Research questions

7

To what extent can an emoji-only deep learning model achieve comparable
accuracy and F1-score to text-only and multimodal (text + emoji) models?

Can an emoji-only model more accurately capture sentiment in sarcastic or ironic
content compared to a text-only model?

When text sentiment is ambiguous or misleading, do emojis have a stronger
influence on the final sentiment prediction?

![page7_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page7_img1.png)

![page7_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page7_img2.png)

![page7_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page7_img3.png)

![page7_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page7_img4.png)

## Research objectives

8

To develop and compare emoji-only, text-only, and multimodal models
using accuracy and F1-score.

To evaluate model performance on sarcastic and contradictory sentiment
data.

To measure the impact of emojis when the text sentiment is neutral or
misleading.

![page8_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page8_img1.png)

![page8_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page8_img2.png)

![page8_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page8_img3.png)

![page8_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page8_img4.png)

p r o p o s e d  a p p r o a c h

6

U n i f i e d  f r a m e w o r k

c o m p a r i n g
TEXT-ONLY
EMOJI-ONLY
# TEXT + EMOJI

m o d e l  a r c h i t r c t u r e

# HYBRID BERT + LSTM

d a t a s e t  u s e d
EMOJI-RICH
SARCASM-FOCUSED

| p r o p o s e d a p p r o a c h 6 U n i f i e d f r a m e w o r k c o m p a r i n g m o d e l a r c h i t r c t u r e d a t a s e t u s e d | TEXT-ONLY EMOJI-ONLY TEXT + EMOJI HYBRID BERT + LSTM EMOJI-RICH SARCASM-FOCUSED |
| --- | --- |

| U n i f i e d f r a m e w o r k c o m p a r i n g |  |  |
| --- | --- | --- |

| m o d e l a r c h i t r c t u r e |  |  |
| --- | --- | --- |

| d a t a s e t u s e d |  |  |
| --- | --- | --- |

![page9_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page9_img1.png)

![page9_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page9_img2.png)

![page9_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page9_img3.png)

![page9_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page9_img4.png)

t r a i n i n g  d a t a s e t

We generated a dataset of 100,000 comments containing emoji-rich content to ensure
sufficient data for training. This was created because existing datasets lacked the level
of emoji diversity required for accurately analyzing emoji-based sentiment.

![page10_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page10_img1.png)

![page10_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page10_img2.png)

![page10_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page10_img3.png)

m e t h a d o l o g y

![page11_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page11_img1.png)

![page11_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page11_img2.png)

![page11_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page11_img3.png)

Results

![page12_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page12_img1.png)

![page12_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page12_img2.png)

![page12_img3.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page12_img3.png)

![page12_img4.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page12_img4.png)

f u t u r e  w o r k s

For future work, we can explore using advanced emojiprocessing libraries in Python to better capture the semantic
meaning of emojis.

Additionally, the model can be trained and evaluated on larger,
real-world datasets to improve its robustness and
generalization in practical applications.

Extend the framework to support multiple languages and
cultural interpretations of emojis.

![page13_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page13_img1.png)

![page13_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page13_img2.png)

c o n c l u s i o n

This study compared text-only, emoji-only, and combined
models for sentiment analysis. The results show that emojis
carry strong emotional information and can perform
competitively with text, especially in detecting sarcasm and
hidden sentiment.

As for the result, the emoji only pipeline identified the most
sentiment

![page14_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page14_img1.png)

![page14_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page14_img2.png)

r e f e r e n c e

[1] Q. Bai, Q. Dan, Z. Mu, and M. Yang, “A Systematic Review of Emoji: Current Research and Future Perspectives,” Front. Psychol., vol. 10, p. 2221, Oct. 2019, doi: 10.3389/fpsyg.2019.02221.
[2] M. A. Ullah, S. M. Marium, S. A. Begum, and N. S. Dipa, “An algorithm and method for sentiment analysis using the text and emoticon,” ICT Express, vol. 6, no. 4, pp. 357–360, Dec. 2020, doi:
10.1016/j.icte.2020.07.003.
[3] Q. A. Xu, C. Jayne, and V. Chang, “An emoji feature-incorporated multi-view deep learning for explainable sentiment classification of social media reviews,” Technological Forecasting and Social
Change, vol. 202, p. 123326, May 2024, doi: 10.1016/j.techfore.2024.123326.
[4] Y. Karulkar, D. T. Vora, S. Vaddepalli, and Y. Thakur, “Are Emojis the New Words? A Sentiment Analysis of Social Media Brand Conversations,” Journal of International Technology and Information
Management, vol. 33, no. 1, pp. 144–181, Mar. 2025, doi: 10.58729/1941-6679.1610.
[5] J. N. Chandra Sekhar et al., “Classification and Comparative Evaluation of Text and Emoji‐Based Tweets With Deep Neural Network Models,” Journal of Electrical and Computer Engineering, vol.
2024, no. 1, p. 9652424, Jan. 2024, doi: 10.1155/2024/9652424.
[6] H. Tang, W. Tang, D. Zhu, S. Wang, Y. Wang, and L. Wang, “EMFSA: Emoji-based multifeature fusion sentiment analysis,” PLoS ONE, vol. 19, no. 9, p. e0310715, Sep. 2024, doi:
10.1371/journal.pone.0310715.
[7] M. Alfreihat, O. S. Almousa, Y. Tashtoush, A. AlSobeh, K. Mansour, and H. Migdady, “Emo-SL Framework: Emoji Sentiment Lexicon Using Text-Based Features and Machine Learning for Sentiment
Analysis,” IEEE Access, vol. 12, pp. 81793–81812, 2024, doi: 10.1109/ACCESS.2024.3382836.
[8] Y. Lou, J. Zhou, J. Zhou, D. Ji, and Q. Zhang, “Emoji multimodal microblog sentiment analysis based on mutual attention mechanism,” Sci Rep, vol. 14, no. 1, p. 29314, Nov. 2024, doi:
10.1038/s41598-024-80167-x.
[9] Y. Lou, Y. Zhang, F. Li, T. Qian, and D. Ji, “Emoji-Based Sentiment Analysis Using Attention Networks,” ACM Trans. Asian Low-Resour. Lang. Inf. Process., vol. 19, no. 5, pp. 1–13, Sep. 2020, doi:
10.1145/3389035.
[10] N. Alamsyah, G. Bayu Wibisono, T. Parama Yoga, Budiman, and A. Hendra, “Emoji-Based Sentiment Classification Using Ensemble Learning with Cross-Validation: A Lightweight Approach for
Social Media Analysis: Klasifikasi Sentimen Berbasis Emoji Menggunakan Ensemble Learning dengan Validasi Silang: Pendekatan Ringan untuk Analisis Media Sosial,” Nuansa Informatika, vol. 19, no. 2,
pp. 81–87, Jul. 2025, doi: 10.25134/ilkom.v19i2.396.
[11] S. Gupta, A. Singh, and V. Kumar, “Emoji, Text, and Sentiment Polarity Detection Using Natural Language Processing,” Information, vol. 14, no. 4, p. 222, Apr. 2023, doi: 10.3390/info14040222.
[12] A. Joseph, S. Carvalho, N. Saldanha, and P. Shaikh, “Emotion Detection Based on Text and Emojis,” in 2024 IEEE International Conference on Information Technology, Electronics and Intelligent
Communication Systems (ICITEICS), Bangalore, India: IEEE, Jun. 2024, pp. 1–6. doi: 10.1109/ICITEICS61368.2024.10625102.
[13] P. T. Sai, G. H. Sri, and T. L. Surekha, “Extraction of Emojis and Texts to Intensify Opinion Mining using Machine Learning and Deep Learning Models,” in 2023 2nd International Conference on
Automation, Computing and Renewable Systems (ICACRS), Pudukkottai, India: IEEE, Dec. 2023, pp. 829–837. doi: 10.1109/ICACRS58579.2023.10404790.
[14] C. Liu et al., “Improving sentiment analysis accuracy with emoji embedding,” Journal of Safety Science and Resilience, vol. 2, no. 4, pp. 246–252, Dec. 2021, doi: 10.1016/j.jnlssr.2021.10.003.
[15] S. Kusal, S. Patil, and K. Kotecha, “Multimodal text-emoji fusion using deep neural networks for text-based emotion detection in online communication,” J Big Data, vol. 12, no. 1, p. 32, Feb. 2025,
doi: 10.1186/s40537-025-01062-4.
[16] S. Velampalli, C. Muniyappa, and A. Saxena, “Performance Evaluation of Sentiment Analysis on Text and Emoji Data Using End-to-End, Transfer Learning, Distributed and Explainable AI Models,”
JAIT, vol. 13, no. 2, 2022, doi: 10.12720/jait.13.2.167-172.
[17] A. Khan, D. Majumdar, and B. Mondal, “Sentiment analysis of emoji fused reviews using machine learning and Bert,” Sci Rep, vol. 15, no. 1, p. 7538, Mar. 2025, doi: 10.1038/s41598-025-92286-0.
[18] Z. Chen, Y. Cao, X. Lu, Q. Mei, and X. Liu, “SEntiMoji: an emoji-powered learning approach for sentiment analysis in software engineering,” in Proceedings of the 2019 27th ACM Joint Meeting on
European Software Engineering Conference and Symposium on the Foundations of Software Engineering, Tallinn Estonia: ACM, Aug. 2019, pp. 841–852. doi: 10.1145/3338906.3338977.
[19] A. Mahrous, J. Schneider, and R. Di Pietro, “The Role of Emojis in Sentiment Analysis of Financial Microblogs,” in 2023 Fourth International Conference on Intelligent Data Science Technologies
and Applications (IDSTA), Kuwai, Kuwait: IEEE, Oct. 2023, pp. 76–84. doi: 10.1109/IDSTA58916.2023.10317863.
[20] E. Gordon, N. Kuppa, R. Tummala, and S. Anasuri, “Understanding Textual Emotion Through Emoji Prediction,” Aug. 13, 2025, arXiv: arXiv:2508.10222. doi: 10.48550/arXiv.2508.10222.

![page15_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page15_img1.png)

![page15_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page15_img2.png)

T h a n k  y o u !

![page16_img1.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page16_img1.png)

![page16_img2.png](Emoji-Only%20Sentiment%20Analysis%20Using%20a%20Deep%20Learning%20Model%20A%20Comparative%20Study%20with%20Text-Enhanced%20Approaches_images/page16_img2.png)
