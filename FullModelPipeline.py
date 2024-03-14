#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[439]:


import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import chromadb
from googleapiclient import discovery
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import re
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import requests
import os
import csv
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import pathlib
import textwrap
from langchain.vectorstores import Chroma
from vertexai.preview import generative_models
import multiprocessing
from scipy.stats import percentileofscore
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from readability import Readability
from sklearn.feature_extraction.text import CountVectorizer
import time
import sys
import site
from sklearn.ensemble import RandomForestClassifier
import transformers
from transformers import BertTokenizer, BertModel
from sklearn.tree import DecisionTreeClassifier
import torch
from FlagEmbedding import FlagReranker
from vertexai import preview
from typing import Dict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import gradio as gr
warnings.filterwarnings("ignore")


# ## Necessary Functions

# In[435]:


def score_to_label(score):
    # Map the score back to the corresponding label
    if score < 16.666:
        return "pants-fire"
    elif score < 33.333:
        return "false"
    elif score < 50:
        return "barely-true"
    elif score < 66.666:
        return "half-true"
    elif score < 83.333:
        return "mostly-true"
    else:
        return "true"


# In[2]:


def predict_label(confidence):
    if confidence < 0.166:
        return "pants-fire"
    elif confidence < 0.33:
        return "false"
    elif confidence < 0.5:
        return "barely-true"
    elif confidence < 0.666:
        return "half-true"
    elif confidence < 0.833:
        return "mostly-true" 
    else:
        return "true"


# In[3]:


def get_word_embeddings(tokens, model):
    embeddings = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


# In[4]:


def get_ngram_embeddings(text, n, model):
    words = text.split()
    ngrams = [words[i:i + n] for i in range(len(words) - n + 1)]  
    embeddings = [model.wv[gram] for gram in ngrams if all(word in model.wv for word in gram)]
    return embeddings


# In[5]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


# In[6]:


def filter_short_strings(text):
    return '' if len(text) < 7 else text


# In[7]:


def tokenize_into_sentences(text):
    return sent_tokenize(text)


# In[8]:


def tokenize_into_chunks(text, min_words=75):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(current_chunk) + len(words) < min_words:
            current_chunk.extend(words)
        else:
            if any(sentence.endswith(p) for p in ['.', '!', '?', '¡', '¿']):
                chunks.append(' '.join(current_chunk))
                current_chunk = words
            else:
                current_chunk.extend(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# In[9]:


def output_clean(text):
    # Remove spaces before or after "'" mark
    text = text.replace(" '", "'").replace("' ", "'")
    # Remove white space before ","
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    # Remove white space before or after "”" or "“" character
    text = text.replace("“ ", "“").replace(" ”", "”")
    text = text.replace("Score:", " Score:")
    text = text.replace(" ’", "’").replace("’ ", "’")
    text = text.replace("$ ", "$")
    text = text.strip()  # Remove leading/trailing whitespace
    return text


# In[10]:


def make_plot(input):
    truth_scores = predict_tabular_classification_sample(project="dsc-180a-b09",
                                                         endpoint_id="4607809140427849728",
                                                         instance_dict={"article": input})
    
    reordered_indices = [truth_scores[0]['classes'].index(c) for c in ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']]
    classes_reordered = [truth_scores[0]['classes'][i] for i in reordered_indices]
    scores_reordered = [truth_scores[0]['scores'][i] for i in reordered_indices]
    
    # Define color transition from red to green
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(classes_reordered)))
    # Plot the bar chart with color transition
    plt.figure(figsize=(6, 4))
    bars = plt.bar(classes_reordered, scores_reordered, color=colors)
    
    # Add title and labels
    plt.title('Predictive Auto ML Truthfulness Scores')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Show plot
    plt.tight_layout()
    return plt


# In[11]:


def label_to_score(label):
    # Invert the scale and linearly map the values
    score = (5 - final_score_mapping[label]) * 100 / 5
    return score


# In[12]:


#ABC News for Left
def abc_updated_news():
    abc_url = "https://abcnews.go.com/"
    abc = requests.get(abc_url)
    abc_soup = BeautifulSoup(abc.content, 'html')
    abc_soup_url = abc_soup.find('a', {'class': 'AnchorLink News News--xl'}).get('href')
    abc_top_article = requests.get(abc_soup_url)
    abc_soup = BeautifulSoup(abc_top_article.content)
    abc_content = abc_soup.find('div', {'data-testid': 'prism-article-body'}).text
    cleaned_abc = abc_content.replace('\'', '')
    abc_headline = abc_soup.find('div', {'data-testid': 'prism-headline'}).text
    return abc_headline, cleaned_abc, abc_soup_url


# In[13]:


#Fox news for right 
def fox_updated_news():
    fox_url = "https://moxie.foxnews.com/google-publisher/latest.xml"
    fox = requests.get(fox_url)
    fox_soup = BeautifulSoup(fox.content, 'xml')
    fox_link = fox_soup.find('item').find('link')
    fox_link_str = str(fox_link)
    fox_link = fox_link_str[6:-7]
    fox_headline = fox_soup.find('item').find('title').text
    fox_content = fox_soup.find('item').find('content:encoded').text
    fox_content_soup = BeautifulSoup(fox_content, 'html.parser')
    for strong_tag in fox_content_soup.find_all('strong'):
        strong_tag.extract()
    cleaned_fox = fox_content_soup.get_text(strip=True)
    cleaned_fox = cleaned_fox.replace('\xa0', '')
    cleaned_fox = cleaned_fox.replace('\\', '')
    return fox_headline, cleaned_fox, fox_link


# In[14]:


#NPR as center
def npr_updated_news():
    npr_url = "https://www.npr.org/"
    npr = requests.get(npr_url)
    npr_soup = BeautifulSoup(npr.content, 'html')
    npr_soup_url = npr_soup.find('div', {'class': 'story-text'})
    npr_soup_url = npr_soup_url.find_all('a')[1]['href']
    npr_article_soup = requests.get(npr_soup_url)
    npr_article_soup = BeautifulSoup(npr_article_soup.content)
    npr_headline = npr_article_soup.find('div', {'class': 'storytitle'}).text
    all_text = npr_article_soup.find('div', {'id': 'storytext'}).find_all('p')
    full_text = ''
    for i in all_text[2:]:
        full_text+=i.text
    full_text = full_text.replace('\'', '').replace('\n', '')
    npr_headline = npr_headline.replace('\n', '')
    return npr_headline, full_text, npr_soup_url


# In[15]:


def npr_output_parse(npr_output):
    npr_output = re.sub(r'``', '"', npr_output)
    npr_output = re.sub(r'" ', '"', npr_output)
    npr_output = re.sub(r'"([^ ])','" \\1', npr_output)
    return npr_output


# In[16]:


def fox_output_parse(fox_output):
    fox_output = re.sub(r'``', '"', fox_output)
    fox_output = re.sub(r'" ', '"', fox_output)
    fox_output = re.sub(r'"([^ ])','" \\1', fox_output)
    return fox_output


# In[17]:


def abc_output_parse(abc_output):
    abc_output = re.sub(r'``', '"', abc_output)
    abc_output = re.sub(r'"\s+', '"', abc_output)
    return abc_output


# In[425]:


def number_to_label(number):
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping[number]


# In[470]:


def scale_and_combine(automl):
    scaling_dict = {"pants-fire": 0, "false": 0.2, "barely-true": 0.4, "half-true": 0.6, "mostly-true": 0.8, "true": 1}
    average_score = 0
    for label, score in automl:
        average_score += scaling_dict[label] * score
    
    bounded_score = max(0.25, min(average_score, 0.80))
    normalized_score = (bounded_score - 0.25) / (0.80 - 0.25)
    print(bounded_score)
    print(normalized_score)
    scaled_variable = normalized_score * 100
    return normalized_score


# ## Datasets

# In[406]:


#Politifact articles
pf_articles = pd.read_csv("Webscraping/politifact_articles.csv")
pf_articles = pf_articles.drop(columns='Unnamed: 0')
pf_articles = pf_articles.drop(columns='Tldr_text_statements')
pf_articles.rename(columns={'Statement': 'Title'}, inplace=True)
pf_articles = pf_articles.dropna()


# In[411]:


#Politifact truth datasets
pf_statements = pd.read_csv("Data/Politifact_Data/CSV/politifact_truthometer_df.csv")
pf_statements = pf_statements.drop(columns='Unnamed: 0')
pf_statements = pf_statements.drop(columns='Unnamed: 0.1')
pf_statements = pf_statements.drop(columns='Tldr_text_statements')
pf_statements = pf_statements.dropna()
pf_statements_full = pf_statements
pf_statements = pf_statements.sample(frac=0.4, random_state=42)


# In[412]:


factcheckorg_articles = pd.read_csv("Webscraping/factcheckorg_webscrape_200pages.csv")
factcheckorg_articles['List_data'].fillna('', inplace=True)
factcheckorg_articles['List_data'] = factcheckorg_articles['List_data'].apply(filter_short_strings)
factcheckorg_articles = factcheckorg_articles.dropna(subset=['Text'])
factcheckorg_articles['Text'] = factcheckorg_articles['Text'].str.replace('Para leer en español, vea esta traducción de Google Translate.', '')
factcheckorg_articles['Text'] = factcheckorg_articles['Text'].str.replace(r' Editor’s Note:.*$', '', regex=True)
factcheckorg_articles = factcheckorg_articles.reset_index()
factcheckorg_articles = factcheckorg_articles.drop(columns=['index', 'Unnamed: 0'])
factcheckorg_articles['Title_and_Date'] = factcheckorg_articles['Title'] + ' , ' + factcheckorg_articles['Date']
factcheckorg_articles = factcheckorg_articles.drop(columns=['Title', 'Date'])


# In[413]:


sciencefeedbackorg_articles = pd.read_csv("Webscraping/science_feedback.csv")
sciencefeedbackorg_articles = sciencefeedbackorg_articles.drop(columns='Unnamed: 0')


# In[414]:


scicheckorg_articles = pd.read_csv("Webscraping/scicheck_data.csv")
scicheckorg_articles['Title_and_Date'] = scicheckorg_articles['Title'] + ' , ' + scicheckorg_articles['Date']
scicheckorg_articles = scicheckorg_articles.drop(columns=['Title', 'Date', 'Unnamed: 0'])
scicheckorg_articles.dropna(inplace=True)


# In[415]:


pf_articles.head(2)


# In[416]:


pf_statements.head(2)


# In[417]:


factcheckorg_articles.head(2)


# In[418]:


sciencefeedbackorg_articles.head(2)


# In[419]:


scicheckorg_articles.head(2)


# ## Predictive Models

# ### Irisa's Models

# In[30]:


#Data Cleaning
def read_dataset(csv):
    df = pd.read_csv(csv)
    df = df.drop(columns=["percentages", "check_nums"]).drop_duplicates().dropna()
    
    mapping = {
        "TRUE": 0,
        "mostly-true": 1,
        "half-true": 2,
        "barely-true": 3,
        "FALSE": 4,
        "pants-fire": 5
    }
    
    df["label"] = df["label"].map(mapping)
    
    df = df[pd.to_numeric(df["label"], errors="coerce").notna()]
    df = df[["content","article","summaries","label"]]
    df["content"] = df["content"].str.replace(r'[“\”]', '', regex=True)
    df["summaries"] = df["summaries"].str.replace(r'[\[\]\'"]', '', regex=True)
    df.columns = ["title", "article", "summary", "label"]

    return df

df = read_dataset("/Users/nicholasshor/Desktop/School/FifthYear/Fall/DSC180A/Data/politifact_data_combined_prev_ratings.csv")
df = df = df[df['summary'] != '']
df.head(2)


# #### Feature 1: Sentiment Analysis  (pos=1, neg=-1, neu=0)

# In[31]:


# 1. Sentiment Analysis Using NLTK

analyzer = SentimentIntensityAnalyzer()
df["sentiment"] = df["article"].apply(lambda x: analyzer.polarity_scores(x)["compound"])


# #### Feature 2: Quality of Writing (Type-Token Ratio (TTR))

# In[32]:


# 1. Remove stopwords and punctuation & Make lowercase

punctuation = set(string.punctuation)
stopwords = set(stopwords.words("english"))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    return " ".join(filtered_words)

def remove_punctuation(text):
    cleaned_text = ''.join([char for char in text if char not in punctuation])
    return cleaned_text

df["article"] = df["article"].apply(lambda x: x.lower())
df["article"] = df["article"].apply(remove_punctuation)
df["article"] = df["article"].apply(remove_stopwords)

# 2. TTR = unique_words/total_words

df['ttr'] = df['article'].apply(lambda x: x.split()).apply(lambda words: len(set(words)) / len(words))


# #### Feature 3: Expressiveness (Adjectives)

# In[33]:


# 1. Open List of Adjectives (Link: https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913)
    ### Additional Sources: https://github.com/taikuukaits/SimpleWordlists/tree/master

with open("/Users/nicholasshor/Desktop/School/FifthYear/Fall/DSC180A/Data/adjectives.txt", "r") as file:
    adjectives = [line.strip() for line in file]
    
# 2. Count adjectives

def count_adjectives(text):
    words = text.split()
    adjective_count = sum(1 for word in words if word.lower() in adjectives) / len(words)
    return adjective_count

df["adjectives"] = df["article"].apply(count_adjectives)


# #### Predictions (One vs Rest)

# In[34]:


X = df.drop(columns=["title","article","summary","label"])
y = df["label"]

irisa_X_train, X_test, y_train, y_test_multi = train_test_split(X, y, test_size=0.2, random_state=42)


# In[237]:


X_test


# In[424]:


sentiment_percentile = percentileofscore(irisa_X_train['sentiment'], irisa_X_train['sentiment'])[0]
ttr_percentile = percentileofscore(irisa_X_train['ttr'], irisa_X_train['ttr'])[0]
adjectives_percentile = percentileofscore(irisa_X_train['adjectives'], irisa_X_train['adjectives'])[0]


# In[38]:


sentiment_percentile


# In[39]:


classifiers = [
    #KNeighborsClassifier(2),
    #SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(),
    GaussianNB()]
    #QuadraticDiscriminantAnalysis()]

for classifier in classifiers:
    irisa_clf = OneVsOneClassifier(classifier).fit(irisa_X_train, y_train)
    predictions = irisa_clf.predict(X_test)
    print(accuracy_score(y_test_multi, predictions))


# In[46]:


#testing new article input
#clickbait
tfidf_vectorizer = TfidfVectorizer()
tfidf_title = tfidf_vectorizer.fit_transform(["Example news title"])
tfidf_article = tfidf_vectorizer.transform([news])
cosine_sim = cosine_similarity(tfidf_title, tfidf_article)
irisa_clickbait = cosine_sim.diagonal()[0]

#sentiment prediction
irisa_sentiment = analyzer.polarity_scores(news)["compound"]

#quality of writing prediction
words = news.split()
irisa_qor_ratio = len(set(words)) / len(words)

#sensationalism
irisa_sensationalism = count_adjectives(news)

#adding to df for prediction
irisa_data = {
    "sentiment": [irisa_sentiment],
    "ttr": [irisa_qor_ratio],
    "adjectives": [irisa_sensationalism]
}

irisa_pred_df = pd.DataFrame(irisa_data)

#irisa final prediction
final_prediction = irisa_clf.predict(irisa_pred_df)[0]


# In[47]:


mapping = {
        "TRUE": 0,
        "mostly-true": 1,
        "half-true": 2,
        "barely-true": 3,
        "FALSE": 4,
        "pants-fire": 5
    }


# In[49]:


irisa_final_label_pred = number_to_label(final_prediction)


# ### Lohit's Model

# In[51]:


#data load and clean
data = pd.read_csv("/Users/nicholasshor/Desktop/School/FifthYear/Fall/DSC180A/Data/politifact_data_combined_prev_ratings.csv")
noise_labels = set(['full-flop', 'half-flip', 'no-flip'])
data = data.query("label not in ['full-flop', 'half-flip', 'no-flip']")


# In[52]:


data.head(2)


# In[53]:


X = data[['content', 'article']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data with unigrams and bigrams
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train['content'] + " " + X_train['article'])
X_test_tfidf = tfidf.transform(X_test['content'] + " " + X_test['article'])

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))


# In[55]:


# Prepare the test instance
X_test_instance = ["example_title" + " " + news]

# Vectorize the test instance using the same TF-IDF vectorizer trained on the training data
X_test_instance_tfidf = tfidf.transform(X_test_instance)

# Make predictions for the test instance
y_pred_instance = classifier.predict(X_test_instance_tfidf)

y_pred_proba = classifier.predict_proba(X_test_instance_tfidf)
positive_class_proba = y_pred_proba
overall_score = (positive_class_proba[0][0] * 0.2) + (positive_class_proba[0][1] * 1) + (positive_class_proba[0][2] * 0.4) + (positive_class_proba[0][3] * 0.6) + (positive_class_proba[0][4] * 0.8) + (positive_class_proba[0][5] * 0.0)
lohit_predicted_label = predict_label(overall_score)
lohit_predicted_label


# ### Nick's Models

# #### Factor 1: Flesch-Kincaid Grade Level Formula

# In[227]:


r = Readability(news_test)
fk = r.flesch_kincaid()
flesch_score = fk.score
if flesch_score > 12:
    diff = flesch_score - 12
    fk_rating = 100 - (diff * 6)
elif flesch_score < 8:
    diff = 8 - flesch_score
    fk_rating = 100 - (diff * 6)
else:
    fk_rating = 100
fk_rating


# #### Factor 2: Sentiment

# In[228]:


sia = SentimentIntensityAnalyzer()
moving_sentiment_value = 0
number_of_paragraphs = 0
paragraphs = news.split('\n\n')
for i in paragraphs:
    cleaned_text = ' '.join(i.split()).replace("\'", '')
    compound_sentiment_score = sia.polarity_scores(cleaned_text)['compound']
    moving_sentiment_value += compound_sentiment_score
    number_of_paragraphs += 1
overall_sentiment = moving_sentiment_value / number_of_paragraphs
overall_sent_score = 100
if overall_sentiment < -0.2:
    overall_sent_score = 100 + (overall_sentiment * 100)
overall_sent_score


# #### Factor 3: Clickbait

# In[59]:


clickbait = pd.read_csv("Data/Clickbait_Data/clickbait_data.csv")


# In[61]:


#determining clickbait
names = ["Linear SVM"] 

classifiers = [SVC(kernel="linear", C=0.025, probability=True)]


#Preprocess, train/test split, and 
clickbait['PreprocessedTitle'] = clickbait['headline'].apply(preprocess_text)
X_train_click, X_test_click, y_train_click, y_test_click = train_test_split(clickbait['PreprocessedTitle'], clickbait['clickbait'], test_size=0.2, random_state=42)
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train_click)
X_test_counts = count_vectorizer.transform(X_test_click)


max_score = 0.0
max_class = ''
# iterate over classifiers
for name, clf_ in zip(names, classifiers):
    start_time = time.time()
    clf_.fit(X_train_counts, y_train_click)
    score = 100.0 * clf_.score(X_test_counts, y_test_click)
    print('Classifier = %s, Score (test, accuracy) = %.2f,' %(name, score), 'Training time = %.2f seconds' % (time.time() - start_time))
    
    if score > max_score:
        clf_best = clf_
        max_score = score
        max_class = name

print(80*'-' )
print('Best --> Classifier = %s, Score (test, accuracy) = %.2f' %(max_class, max_score))


# In[63]:


article_title_processed = preprocess_text("example news")
article_title_vectorized = count_vectorizer.transform([article_title_processed])
clickbait_probability = clf_best.predict_proba(article_title_vectorized)
confidence_not_clickbait = clickbait_probability[:, 0]
confidence_not_clickbait = confidence_not_clickbait[0]
nick_predicted_label = predict_label(confidence_not_clickbait)
nick_predicted_label


# In[64]:


factors_combined = ((confidence_not_clickbait * 100) + overall_sent_score + fk_rating) / 3
nicks_predicted_label = predict_label(factors_combined)
nicks_predicted_label


# ### Henry's Model

# In[433]:


user_site_packages = site.USER_SITE
sys.path.append(user_site_packages)
print(sys.path)


# In[66]:


columns = ['ID', 'Label', 'Statement', 'Subject(s)', 'Speaker','Speaker\'s Job Title', 'State Info', 'Party Affiliation','Barely True', 'False', 'Half True', 'Mostly True', 'Pants on Fire','Context']
df = pd.read_csv('/Users/nicholasshor/Desktop/School/FifthYear/Fall/DSC180A/Data/Liar_plus/train.tsv', delimiter='\t', header = None, quoting=csv.QUOTE_NONE)
df = df.drop(columns=[0, 15])
df = df.rename(columns=dict(zip(df.columns, columns)))
df = df.dropna()


# In[67]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_ = BertModel.from_pretrained('bert-base-uncased')


# In[68]:


statements = df['Statement'].tolist()
labels = df['Label'].tolist()
#tokenizeing the statements
tokenized_statements = [tokenizer(statement, return_tensors="pt", truncation=True, padding=True) for statement in statements]


# In[69]:


# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
henry_model = model_.to(device)

# Move tokenized statements to GPU
tokenized_statements_gpu = [inputs.to(device) for inputs in tokenized_statements]

# Extract BERT embeddings
with torch.no_grad():
    henry_model.eval()
    statement_embeddings = [henry_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy() for inputs in tokenized_statements_gpu]


# In[70]:


#flatten
X_embeddings = np.vstack(statement_embeddings)


# In[71]:


#combine the truth counts with the embeddings
X_embeddings_df = pd.DataFrame(X_embeddings, columns=[f"embedding_{i}" for i in range(X_embeddings.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(X_embeddings_df, labels, test_size=0.2, random_state=42)


# In[72]:


rf_classifier = RandomForestClassifier(n_estimators=3000, random_state=42, min_samples_split = 2, min_samples_leaf = 1, max_depth = 30)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))


# ### Code for pred scores combining and model testing

# In[175]:


final_score_mapping = {
        "true": 0,
        "TRUE": 0,
        " true":0,
        "mostly-true": 1,
        " mostly-true": 1,
        "half-true": 2,
        " half-true": 2,
        "barely-true": 3,
        " barely-true": 3,
        "false": 4,
        " false": 4,
        "FALSE": 4,
        "pants-fire": 5,
        " pants-fire": 5
    }


# In[84]:


labels_as_scores = []
for i in all_ratings:
    labels_as_scores.append(label_to_score(i))
pred_models_ave = sum(labels_as_scores) / len(labels_as_scores)
final_pred_label = score_to_label(pred_models_ave)
final_pred_label


# ## Chunking

# In[322]:


#FactCheckOrg Article Chunking
factcheckorg_articles['chunks_text'] = factcheckorg_articles['Text'].apply(tokenize_into_chunks)
factcheckorg_articles['chunkslistdata'] = factcheckorg_articles['List_data'].apply(tokenize_into_chunks)

# Determine the maximum number of chunks across both columns
max_chunks_text = factcheckorg_articles['chunks_text'].apply(len).max()
max_chunks_list_data = factcheckorg_articles['chunkslistdata'].apply(len).max()
max_total_chunks = max(max_chunks_text, max_chunks_list_data)

# Create columns for each chunk in both 'Text' and 'List_data'
for i in range(1, max_total_chunks + 1):
    factcheckorg_articles[f'chunk_text_{i}'] = factcheckorg_articles['chunks_text'].apply(lambda x: x[i - 1] if len(x) >= i else None)
    factcheckorg_articles[f'chunklistdata{i}'] = factcheckorg_articles['chunkslistdata'].apply(lambda x: x[i - 1] if len(x) >= i else None)

# Drop unnecessary columns
factcheckorg_articles = factcheckorg_articles.drop(columns=['chunks_text', 'chunkslistdata', 'Text', 'List_data'])


# In[323]:


#Politifact Statement Text Chunking
pf_statements['chunks'] = pf_statements['Text'].apply(tokenize_into_chunks)

max_chunks = pf_statements['chunks'].apply(len).max()

for i in range(1, max_chunks + 1):
    pf_statements[f'chunk_{i}'] = pf_statements['chunks'].apply(lambda x: x[i - 1] if len(x) >= i else None)

pf_statements = pf_statements.drop(columns=['chunks', 'Text'])


# In[89]:


#Politifact Articles Chunking
pf_articles['chunks'] = pf_articles['Text'].apply(tokenize_into_chunks)

max_chunks = pf_articles['chunks'].apply(len).max()

for i in range(1, max_chunks + 1):
    pf_articles[f'chunk_{i}'] = pf_articles['chunks'].apply(lambda x: x[i - 1] if len(x) >= i else None)

pf_articles = pf_articles.drop(columns=['chunks', 'Text'])


# In[90]:


#SciCheckOrg Articles Chunking
scicheckorg_articles['chunks_text'] = scicheckorg_articles['Text'].apply(tokenize_into_chunks)

# Determine the maximum number of chunks across both columns
max_chunks_text = scicheckorg_articles['chunks_text'].apply(len).max()

# Create columns for each chunk in both 'Text' and 'List_data'
for i in range(1, max_chunks_text + 1):
    scicheckorg_articles[f'chunk_text_{i}'] = scicheckorg_articles['chunks_text'].apply(lambda x: x[i - 1] if len(x) >= i else None)

# Drop unnecessary columns
scicheckorg_articles = scicheckorg_articles.drop(columns=['chunks_text', 'Text'])


# ## Vector Database

# In[91]:


chroma_client = chromadb.Client()


# In[92]:


RAG_CONTEXT_VDB = chroma_client.create_collection(name="RAG_CONTEXT_VDB")


# In[93]:


RAG_STATEMENTS_VDB = chroma_client.create_collection(name="RAG_STATEMENTS_VDB")


# In[94]:


#Adding pf statement justifications to Context VDB
ids_list = []
metadata_list = []
chunks_list = []
start_id = RAG_CONTEXT_VDB.count() + 1

for index, row in pf_statements.iterrows():
    statement = row['Statement']
    claimer = row['Claimer']
    for col in pf_statements.columns:
        if col.startswith('chunk_'):
            chunk = row[col]
            if chunk is not None:
                chunks_list.append(chunk)
                metadata_list.append({"Statement": statement, "Context": "Yes", "Claimer": claimer})
                ids_list.append(f"id{start_id}")
                start_id += 1


# In[95]:


#Adding pf truth-o-meter justifications to vector database in batches of 5000 (max batch size is just over 5000)
start_size = 0
batch_size_increment = 5000
batch_size = 5000
for i in range(((len(chunks_list)//batch_size)+1)):
    RAG_CONTEXT_VDB.add(
        documents=chunks_list[start_size:batch_size],
        metadatas=metadata_list[start_size:batch_size],
        ids=ids_list[start_size:batch_size])
    start_size = start_size + batch_size_increment
    batch_size = batch_size + batch_size_increment
    print(start_size)


# In[96]:


#Adding politifact truth-o-meter statements to Statements VDB
statements_list = []
ids_list = []
metadata_list = []
start_id = RAG_STATEMENTS_VDB.count() + 1

for index, row in pf_statements_full.iterrows():
    truth_value = row['Truth_value']
    claimer = row['Claimer']
    statement = row['Statement']

    metadata_list.append({"Statements truthfulness":truth_value,"Claimer": claimer})
    statements_list.append(statement)
    
    ids_list.append(f"id{start_id}")
    start_id += 1


# In[97]:


#Adding pf truth-o-meter statements to vector database in batches of 5000 (max batch size is just over 5000)
start_size = 0
batch_size_increment = 5000
batch_size = 5000
for i in range(((len(chunks_list)//batch_size)+1)):
    RAG_STATEMENTS_VDB.add(
        documents=statements_list[start_size:batch_size],
        metadatas=metadata_list[start_size:batch_size],
        ids=ids_list[start_size:batch_size])
    start_size = start_size + batch_size_increment
    batch_size = batch_size + batch_size_increment
    print(start_size)


# In[98]:


#Adding factcheck.org data to Context VDB
chunks_list = []
titles_list = []
ids_list = []
start_id = RAG_CONTEXT_VDB.count() + 1

for index, row in factcheckorg_articles.iterrows():
    title = row['Title_and_Date']
    for col in factcheckorg_articles.columns:
        if col.startswith('chunk_'):
            chunk = row[col]
            if chunk is not None:
                chunks_list.append(chunk)
                titles_list.append({"Title_and_Date": title, "Context": "Yes"})
                ids_list.append(f"id{start_id}")
                start_id += 1
        elif col.startswith('chunklist'):
            chunk = row[col]
            if chunk is not None:
                chunks_list.append(chunk)
                titles_list.append({"Title_and_Date": title, "Context": "Yes"})
                ids_list.append(f"id{start_id}")
                start_id += 1


# In[99]:


#Adding factcheckorg text to vector database in batches of 5000 (max batch size is just over 5000)
start_size = 0
batch_size_increment = 5000
batch_size = 5000
for i in range(((len(chunks_list)//batch_size)+1)):
    RAG_CONTEXT_VDB.add(
        documents=chunks_list[start_size:batch_size],
        metadatas=titles_list[start_size:batch_size],
        ids=ids_list[start_size:batch_size])
    start_size = start_size + batch_size_increment
    batch_size = batch_size + batch_size_increment
    print(start_size)


# In[100]:


#adding SciCheckOrg articles to Context VDB
chunks_list = []
titles_list = []
ids_list = []
start_id = RAG_CONTEXT_VDB.count() + 1

for index, row in scicheckorg_articles.iterrows():
    title = row['Title_and_Date']
    for col in scicheckorg_articles.columns:
        if col.startswith('chunk_'):
            chunk = row[col]
            if chunk is not None:
                chunks_list.append(chunk)
                titles_list.append({"Title_and_Date": title, "Context": "Yes"})
                ids_list.append(f"id{start_id}")
                start_id += 1


# In[101]:


RAG_CONTEXT_VDB.count()


# In[102]:


#Adding scicheckorg text to vector database in batches of 5000 (max batch size is just over 5000)
start_size = 0
batch_size_increment = 5000
batch_size = 5000
for i in range(((len(chunks_list)//batch_size)+1)):
    RAG_CONTEXT_VDB.add(
        documents=chunks_list[start_size:batch_size],
        metadatas=titles_list[start_size:batch_size],
        ids=ids_list[start_size:batch_size])
    start_size = start_size + batch_size_increment
    batch_size = batch_size + batch_size_increment
    print(start_size)


# In[103]:


#adding ScienceFeedbackOrg statements to Statements VDB
statements_list = []
ids_list = []
metadata_list = []
start_id = RAG_STATEMENTS_VDB.count() + 1

for index, row in sciencefeedbackorg_articles.iterrows():
    truth_value = row['label']
    statement = row['claim']

    metadata_list.append({"Statements truthfulness":truth_value})
    statements_list.append(statement)
    
    ids_list.append(f"id{start_id}")
    start_id += 1


# In[104]:


#Adding pf sciencefeedback statements to vector database in batches of 5000 (max batch size is just over 5000)
start_size = 0
batch_size_increment = 5000
batch_size = 5000
for i in range(((len(chunks_list)//batch_size)+1)):
    RAG_STATEMENTS_VDB.add(
        documents=statements_list[start_size:batch_size],
        metadatas=metadata_list[start_size:batch_size],
        ids=ids_list[start_size:batch_size])
    start_size = start_size + batch_size_increment
    batch_size = batch_size + batch_size_increment
    print(start_size)


# ## FULL GEN AI MODEL

# In[245]:


#loading live examples
abc_headline, abc_content, abc_link = abc_updated_news()
npr_headline, npr_content, npr_link = npr_updated_news()
fox_headline, fox_content, fox_link = fox_updated_news()


# #### Perspective API Setup

# In[117]:


PERSPECTIVE_API_KEY = 'AIzaSyCElMgVeT2_ng6hSnJMNHXt4t78fOv8J9U'


# In[118]:


#thresholds of output
attributeThresholds = {
    'INSULT': 0.8,
    'TOXICITY': 0.8,
    'THREAT': 0.5,
    'SEXUALLY_EXPLICIT': 0.5,
    'PROFANITY': 0.8
}
requestedAttributes = {}
for key in attributeThresholds:
    requestedAttributes[key] = {}


# #### Flag Embedding Reranker import/load

# In[116]:


reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)


# #### AutoML Predict Function and Setup

# In[129]:


def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",):
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    prediction_list=[]
    predictions = response.predictions
    for prediction in predictions:
        prediction_list.append(dict(prediction))
    return prediction_list


# In[130]:


testing=predict_tabular_classification_sample(
    project="dsc-180a-b09",
    endpoint_id="4607809140427849728",
    instance_dict={"article": news})
testing[0]['scores']


# #### Liar Liar Plus Dataset Testing

# This section is only used for testing the performance of the generative model and is not necessary for ones own implementation.

# In[480]:


# def GenAI_article_truth_processing(news_article, history, examples, headline):
#     history_output = []

        
#     #instantiating RAG re-ranking mecahnism
#     reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

#     #converting full news article to string
#     news_article = f"""{news_article}"""
#     example_news_provider = f"""{news_article}"""
#     headline = f"""{headline}"""

#     #setting up pre-processed examples correctly
#     if news_article == "ABC" or news_article == "abc" or news_article == "Abc":
#         news_article = abc_content
#         headline = abc_headline
#     elif news_article == "NPR" or news_article == "npr" or news_article == "Npr":
#         news_article = npr_content
#         headline = npr_headline
#     elif news_article == "FOX" or news_article == "Fox" or news_article == "fox":
#         news_article = fox_content
#         headline = fox_headline
#     overall_synopsis = "Overall Score"
        
#     #getting history for context
#     history = history or []

#     #predictive models
#     #IRISAS PREDICTIONS

#     #sentiment prediction
#     irisa_sentiment = analyzer.polarity_scores(news_article)["compound"]
    
#     #quality of writing prediction
#     words = news_article.split()
#     irisa_qor_ratio = len(set(words)) / len(words)
    
#     #sensationalism
#     irisa_sensationalism = count_adjectives(news_article)
    
#     #adding to df for prediction
#     irisa_data = {
#         "sentiment": [irisa_sentiment],
#         "ttr": [irisa_qor_ratio],
#         "adjectives": [irisa_sensationalism]
#     }
    
#     irisa_pred_df = pd.DataFrame(irisa_data)
    
#     #irisa final prediction
#     irisa_final_prediction = irisa_clf.predict(irisa_pred_df)[0]
#     irisa_final_label_pred = number_to_label(irisa_final_prediction)

#     #irisas prediction percentile
#     sentiment_percentile = percentileofscore(irisa_X_train['sentiment'], irisa_pred_df['sentiment'])[0]
#     ttr_percentile = percentileofscore(irisa_X_train['ttr'], irisa_pred_df['ttr'])[0]
#     adjectives_percentile = percentileofscore(irisa_X_train['adjectives'], irisa_pred_df['adjectives'])[0]

#     #LOHITS PREDICTIONS
#     X_test_instance = [headline + " " + news_article]

#     # Vectorize the test instance using the same TF-IDF vectorizer trained on the training data
#     X_test_instance_tfidf = tfidf.transform(X_test_instance)
    
#     # Make predictions for the test instance
#     y_pred_instance = classifier.predict(X_test_instance_tfidf)
    
#     y_pred_proba = classifier.predict_proba(X_test_instance_tfidf)
#     positive_class_proba = y_pred_proba
#     overall_score = (positive_class_proba[0][0] * 0.2) + (positive_class_proba[0][1] * 1) + (positive_class_proba[0][2] * 0.4) + (positive_class_proba[0][3] * 0.6) + (positive_class_proba[0][4] * 0.8) + (positive_class_proba[0][5] * 0.0)
#     lohit_ngram_prediction = predict_label(overall_score)
    
#     truth_scores = predict_tabular_classification_sample(project="dsc-180a-b09",
#                                                          endpoint_id="4607809140427849728",
#                                                          instance_dict={"article": news})
#     reordered_indices = [truth_scores[0]['classes'].index(c) for c in ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']]
#     classes_reordered = [truth_scores[0]['classes'][i] for i in reordered_indices]
#     scores_reordered = [truth_scores[0]['scores'][i] for i in reordered_indices]
#     final_automl_score = (scores_reordered[0] * 0) + (scores_reordered[1] * 0.2) + (scores_reordered[2] * 0.4) + (scores_reordered[3] * 0.6) + (scores_reordered[4] * 0.8) + (scores_reordered[5] * 1)
#     auto_ml_score_label = predict_label(final_automl_score)
#     final_pred = (overall_score + final_automl_score) / 2
    
#     lohit_final_prediction = predict_label(final_pred)


#     #NICKS PREDICTIONS
#     # #readability
#     # r = Readability(news_article)
#     # fk = r.flesch_kincaid()
#     # flesch_score = fk.score
#     # if flesch_score > 12:
#     #     diff = flesch_score - 12
#     #     fk_rating = 100 - (diff * 10)
#     # elif flesch_score < 8:
#     #     diff = 8 - flesch_score
#     #     fk_rating = 100 - (diff * 10)
#     # else:
#     #     fk_rating = 100

#     #sentiment
#     sia = SentimentIntensityAnalyzer()
#     moving_sentiment_value = 0
#     number_of_paragraphs = 0
#     paragraphs = news_article.split('\n\n')
#     for i in paragraphs:
#         cleaned_text = ' '.join(i.split()).replace("\'", '')
#         compound_sentiment_score = sia.polarity_scores(cleaned_text)['compound']
#         moving_sentiment_value += compound_sentiment_score
#         number_of_paragraphs += 1
#     overall_sentiment = moving_sentiment_value / number_of_paragraphs
#     overall_sent_score = 100
#     if overall_sentiment < -0.2:
#         overall_sent_score = 100 + (overall_sentiment * 100)

#     #clickbait
#     if len(headline) > 0:
#         article_title_processed = preprocess_text(headline)
#         article_title_vectorized = count_vectorizer.transform([article_title_processed])
#         clickbait_probability = clf_best.predict_proba(article_title_vectorized)
#         confidence_not_clickbait = clickbait_probability[:, 0]
#         confidence_not_clickbait = confidence_not_clickbait[0]
#         nick_predicted_label = predict_label(confidence_not_clickbait)
#     else:
#         confidence_not_clickbait = 0

#     if confidence_not_clickbait == 0:
#         factors_combined = (overall_sent_score)
#         nicks_predicted_label = predict_label(factors_combined)
#     else:
#         factors_combined = ((confidence_not_clickbait * 100) + overall_sent_score) / 2
#         nicks_predicted_label = predict_label(factors_combined)

#     #HENRYS PREDICTIONS
#     # Tokenize the single text example
#     tokenized_news = tokenizer(news, return_tensors="pt", truncation=True, padding=True)
    
#     # Use GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_henry = model_.to(device)
    
#     # Move tokenized example to GPU
#     tokenized_news_gpu = tokenized_news.to(device)
    
#     # Extract BERT embeddings for the tokenized example
#     with torch.no_grad():
#         model_henry.eval()
#         statement_embedding = model_henry(**tokenized_news_gpu).last_hidden_state.mean(dim=1).cpu().numpy()
    
#     # Use the RandomForestClassifier to predict the label for the single text example
#     y_pred_news = rf_classifier.predict(statement_embedding)
    
#     henry_final_prediction = y_pred_news[0]

#     #combining all group mates predictive scores
#     all_ratings = [irisa_final_label_pred,lohit_final_prediction,nicks_predicted_label,henry_final_prediction]
#     labels_as_scores = []
#     for i in all_ratings:
#         labels_as_scores.append(label_to_score(i))
#     pred_models_ave = sum(labels_as_scores) / len(labels_as_scores)
#     final_pred_label = score_to_label(pred_models_ave)

#     # if len(headline) > 0:
#     #     pred_score_output = f"""The overall score created from our predictive models is {pred_models_ave}. This means the article has been evaluated to be {final_pred_label}. The individual scores of the predictive models are as follows.
#     #     Full-text n-gram analysis: {lohit_ngram_prediction}
#     #     Full-text BERT embedding prediction: {henry_final_prediction}
#     #     Google AUTO ML full-text analysis: {auto_ml_score_label}
#     #     Readability Score: {round(fk_rating,2)}
#     #     Not Clickbait Probability: {round(confidence_not_clickbait*100,2)}%
#     #     Quality of Writing Percentile: {round(ttr_percentile,2)}%
#     #     Sensationalism Score Percentile: {round(adjectives_percentile,2)}%
#     #     Sentiment Score Percentile: {round(sentiment_percentile,2)}%"""
#     # else:
#     #     pred_score_output = f"""The overall score created from our predictive models is {pred_models_ave}. This means the article has been 
#     #     evaluated to be {final_pred_label}. The individual scores of the predictive models are as follows.
#     #     Full-text n-gram analysis: {lohit_ngram_prediction}
#     #     Full-text BERT embedding prediction: {henry_final_prediction}
#     #     Google AUTO ML full-text analysis: {auto_ml_score_label}
#     #     Readability Score: {round(fk_rating,2)}
#     #     Quality of Writing Percentile: {round(ttr_percentile,2)}
#     #     Sensationalism Score Percentile: {round(adjectives_percentile,2)}
#     #     Sentiment Score Percentile: {round(sentiment_percentile,2)}"""
        
#     #Pre-processed examples output
#     if example_news_provider == "ABC":
#         history_output.append([news_article, abc_final_output])
#         return history_output, history_output, pred_score_output, overall_synopsis
#     elif example_news_provider == "NPR":
#         history_output.append([news_article, npr_final_output])
#         return history_output, history_output, pred_score_output, overall_synopsis
#     elif example_news_provider == "FOX":
#         history_output.append([news_article, fox_final_output])
#         return history_output, history_output, pred_score_output, overall_synopsis
        
    
#     #GEN AI
#     #instantiating gemini pro model
#     PROJECT_ID = "gen-lang-client-0321728687"
#     REGION = "us-central1"
#     vertexai.init(project=PROJECT_ID, location=REGION)
#     model = generative_models.GenerativeModel("gemini-pro")
#     config = {"max_output_tokens": 2048, "temperature": 0.0}
    
#     safety_config = {
#     generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH
#     }
#     chat = model.start_chat()

#     #PerspectiveAPI output check instantiation
#     client = discovery.build(
#       "commentanalyzer",
#       "v1alpha1",
#       developerKey=PERSPECTIVE_API_KEY,
#       discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#       static_discovery=False,
#         )
    
#     #chunking news article for improved processing
#     chunked_article_list = tokenize_into_chunks(news_article, 50)
    
#     #getting context and fact checks from vector database based on the provided input
#     all_response_text = []
#     context_list = []
#     for i in range(len(chunked_article_list)):
#         input = chunked_article_list[i]
#         context = RAG_CONTEXT_VDB.query(
#             query_texts=[input],
#             n_results=7,
#         )
#         context_list.append(context)
        
#     fact_checks_list=[]
#     for i in range(len(chunked_article_list)):
#         input = chunked_article_list[i]
#         fact_checks = RAG_STATEMENTS_VDB.query(
#             query_texts=[input],
#             n_results=7,
#         )
#         fact_checks_list.append(fact_checks)

#     #creating history list so that gen ai model has additional context when analyzing chunked statements 
#     for i in range(len(context_list)):
#         input=chunked_article_list[i]
#         fact_checks = fact_checks_list[i]
#         context = context_list[i]
#         prev_chunk = chunked_article_list[i - 1] if i > 0 else None
#         next_chunk = chunked_article_list[i + 1] if i + 1 < len(chunked_article_list) else None
        
#         history = [prev_chunk, input, next_chunk]
        
#         #re-ranking RAG results for fact check statements from RAG_STATEMENTS_VDB
#         statement_rerank_list = []
#         for j in range(len(fact_checks['ids'][0])):
#             reranking_statementSearch = [input, fact_checks['documents'][0][j]]
#             statement_rerank_list.append(reranking_statementSearch)
    
        
#         scores = reranker.compute_score(statement_rerank_list)
#         combined_statement_scores = list(zip(scores, statement_rerank_list, fact_checks['metadatas'][0]))
#         sorted_combined_data = sorted(combined_statement_scores, key=lambda x: x[0], reverse=True)
#         sorted_statement_scores, sorted_statement_rerank_list, sorted_factCheck_metadata = zip(*sorted_combined_data)
    
#         #re-ranking RAG results for context statements from RAG_CONTEXT_VDB
#         context_rerank_list = []
#         for k in range(len(context['ids'][0])):
#             reranking_contextSearch = [input, context['documents'][0][k]]
#             context_rerank_list.append(reranking_contextSearch)
            
#         scores = reranker.compute_score(context_rerank_list)
#         combined_context_scores = list(zip(scores, context_rerank_list, context['metadatas'][0]))
#         sorted_combined_data = sorted(combined_context_scores, key=lambda x: x[0], reverse=True)
#         sorted_context_scores, sorted_context_rerank_list, sorted_context_metadata = zip(*sorted_combined_data)

#         #getting top 3 most relevant pieces of context and fact checks from RAG
#         context_window = 3
#         prepared_context = []
#         prepared_fact_checks = []
#         for i in range(context_window):
#             prepared_context.append([sorted_context_metadata[i], sorted_context_rerank_list[i][1]])
#             prepared_fact_checks.append([sorted_factCheck_metadata[i], sorted_statement_rerank_list[i][1]])

#         #Changing chunks from list of strings to one combined string for Gen AI processing
#         chunk_history_string = ''
#         for chunk in history:
#             if chunk != None:
#                 chunk_history_string += chunk + " "


#         #generating initial response with prompt template
#         responses = model.generate_content(f"""Answer the question below marked inside <<<>>> in a full sentence based on the
#         knowledge I have provided you below, as well as information you already have access to answer the question.
#         Use the additional information I've provided below within the ((())) symbols to help you. 
#         (((
#         Refer to these fact-checked statements to determine your answer and be sure to pay close attention to the 
#         metadata that is provided: {prepared_fact_checks}.
#         Also use the following context to help answer the question: {prepared_context}.
#         You may also use the chat history provided to help you understand the context better if available: {chunk_history_string}.
#         Ensure that you use all this information and think about this question step-by-step using the provided information.
#         )))
#         <<<
#         Question: How true is the following statement? + {input}. You must provide your rating by giving back just a single label from
#         the following list that is listed in increasing levels of truthfulness. The labels are as follows. 
#         ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']. Your response must match the format provided exactly.
#         >>>
#        """,
#             generation_config=config,
#             stream=True,
#             safety_settings=safety_config,                          
#         )
        
#         #obtaining individual responses
#         response_text = ""
#         #response_text += "Statement: " + input
#         for response in responses:
#             try:
#                 response_text += response.text
#             except (IndexError, ValueError) as e:
#                 continue
#         response_text = response_text.replace("\n\n", ". ")
#         all_response_text.append(response_text)
        
                                  
#     #combining all responses    
#     entire_text_string = ""
#     for text in all_response_text:
#         entire_text_string += text
#     cleaned_text = entire_text_string
    
#     #this section is finding and removing the statements that can't be rated by the chatbot
#     unratable_sentences = []
#     rated_sentences = []

#     final_text = ""
#     for response in all_response_text:
#         if "article does not" in response.lower() or "context does not" in response.lower() or "statement is not" in response.lower() or "Statement: Score" in response:
#             unratable_sentences.append(response)
#         else:
#             rated_sentences.append(response)
#             final_text += response
    
#     not_enough_context = len(unratable_sentences)
#     enough_context = len(rated_sentences)
#     all_statements_count = len(all_response_text)


#     #total score calculation with regex
#     pattern = r'Score:\s(\d+)\.'
#     total_score = 0
#     matches = re.findall(pattern, cleaned_text)
#     for match in matches:
#         score = int(match)
#         total_score += score
#     if enough_context == 0:
#         average_score = 0
#     else:
#         average_score = total_score / enough_context
#     rounded_average = round(average_score, 1)

#     # #creating output in nice format for user
#     # output_intro = f"""{enough_context} out of {all_statements_count} statements in the text could be rated. The following score and explanation is based on these {enough_context} statements. The average truthfulness score from these {all_statements_count} statements is {rounded_average}/100. Some of the lowest rated statements are provided below."""
#     # tweaking_output = re.sub(r'(Score:\s*\d+\.)(?!\s*Explanation:)', r'\1 Explanation:', final_text)
#     # parts = re.split(r"(?=Statement:)", tweaking_output)
#     # split_parts=[]
#     # # Clean each part and add to split_parts
#     # for part in parts:
#     #     cleaned_part = output_clean(part)
#     #     split_parts.append(cleaned_part)
    
#     # # Initialize variables to store lowest scores and their respective entries
#     # lowest_scores = [(float('inf'), ''), (float('inf'), ''), (float('inf'), '')]
    
#     # # Iterate through each string entry
#     # for entry in split_parts:
#     #     # Find all occurrences of "Score: " followed by a number until a "."
#     #     scores = re.findall(r' Score:\s(\d+)\.', entry)
#     #     # Convert scores to integers and update lowest_scores if necessary
#     #     for score in scores:
#     #         score_int = int(score)
#     #         if score_int < lowest_scores[-1][0]:
#     #             lowest_scores[-1] = (score_int, entry)
#     #             lowest_scores.sort()
                
#     # # Extract the entries for the three lowest scores
#     # lowest_entries = [entry for score, entry in lowest_scores]

#     # #reformatting for better readability
#     # summary_output = ""
#     # for statement in lowest_entries:
#     #     # Replace "Statement:", "Score:", and "Explanation:" with a new line followed by the keyword
#     #     formatted_statement = re.sub(r'(Statement:|Score:|Explanation:)', r'\n\1', statement)
#     #     # Append the formatted statement to the output
#     #     summary_output += formatted_statement.strip() + "\n"
    
#     #     # Add a new line after each statement
#     #     summary_output += "\n"

#     # output = output_intro + "\n\n" + summary_output

#     # #Perspective API output safety check
#     # analyze_request = {
#     #   'comment': { 'text': output},
#     #   'requestedAttributes': requestedAttributes
#     # }
#     # response = client.comments().analyze(body=analyze_request).execute()
    
#     # attributes_surpassed = []
#     # for key in response['attributeScores']:
#     #     if response['attributeScores'][key]['summaryScore']['value'] > attributeThresholds[key]:
#     #         attributes_surpassed.append((key, response['attributeScores'][key]['summaryScore']['value']))
    
#     # #crafting output warning message if necessary or regular output message  
    
#     # if len(attributes_surpassed) == 1:
#     #     attributes_violated = ""
#     #     for i in attributes_surpassed:
#     #         attributes_violated += i[0] + " "
#     #     warning_message = f"""We're sorry, the output message surpasses our threshold for the {attributes_violated}category so we cannot safely provide a response. Please try again with a different input."""
#     #     history_output.append([news_article, warning_message])
        
#     # elif len(attributes_surpassed) > 1:
#     #     attributes_violated = ""
#     #     counter = 1
#     #     attributes_count = len(attributes_surpassed)
#     #     for i in attributes_surpassed:
#     #         attributes_violated += i[0] + " "
#     #         if counter < attributes_count:
#     #             attributes_violated += "and "
#     #         counter += 1
#     #     warning_message = f"""We're sorry, the output message surpasses our threshold for the {attributes_violated}categories so we cannot safely provide a response. Please try again with a different input."""
#     #     history_output.append([news_article, warning_message])

#     # else:
#     overall_synopsis = "overall analysis"
#     history_output.append([news_article, all_response_text])
#     return history_output, history_output, final_pred_label, overall_synopsis


# In[192]:


liar_liar_plus = pd.read_csv("Data/Liar_plus/train.tsv", delimiter='\t', header=None)
liar_liar_plus = liar_liar_plus[[3, 2]]
liar_liar_plus.dropna(inplace=True)
llp_statements = liar_liar_plus[3].sample(n=500, random_state=42)
llp_labels = liar_liar_plus[2].sample(n=500, random_state=42)


# In[193]:


label_prediction = []
for i in llp_statements:
    try:
        model_load = GenAI_article_truth_processing(i,[], [], [])
        gen_score = label_to_score(model_load[0][0][1][0]) * 0.8
        pred_score = label_to_score(model_load[2]) * 0.2
        overall_score = (gen_score + pred_score)
        final_label = score_to_label(overall_score)
        label_prediction.append(final_label)
    except (ValueError, IndexError, KeyError) as e:
        label_prediction.append([model_load[0][0][1][0], model_load[2]])
        continue


# In[194]:


len(label_prediction)


# In[195]:


labels_list = llp_labels.tolist()
len(labels_list)


# In[197]:


count_same_values = 0

for value1, value2 in zip(label_prediction, labels_list):
    if value1 == value2:
        count_same_values += 1


# In[198]:


count_same_values / (len(label_prediction))


# #### GEN AI Model Function

# In[478]:


def GenAI_article_truth_processing(news_article, history, examples, headline):
    history_output = []

        
    #instantiating RAG re-ranking mecahnism
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

    #converting full news article to string
    news_article = f"""{news_article}"""
    example_news_provider = f"""{news_article}"""
    headline = f"""{headline}"""

    #setting up pre-processed examples correctly
    if news_article == "ABC" or news_article == "abc" or news_article == "Abc":
        news_article = abc_content
        headline = abc_headline
    elif news_article == "NPR" or news_article == "npr" or news_article == "Npr":
        news_article = npr_content
        headline = npr_headline
    elif news_article == "FOX" or news_article == "Fox" or news_article == "fox":
        news_article = fox_content
        headline = fox_headline
    overall_synopsis = "Overall Score"
        
    #getting history for context
    history = history or []

    #predictive models
    #IRISAS PREDICTIONS

    #sentiment prediction
    irisa_sentiment = analyzer.polarity_scores(news_article)["compound"]
    
    #quality of writing prediction
    words = news_article.split()
    irisa_qor_ratio = len(set(words)) / len(words)
    
    #sensationalism
    irisa_sensationalism = count_adjectives(news_article)
    
    #adding to df for prediction
    irisa_data = {
        "sentiment": [irisa_sentiment],
        "ttr": [irisa_qor_ratio],
        "adjectives": [irisa_sensationalism]
    }
    
    irisa_pred_df = pd.DataFrame(irisa_data)
    
    #irisa final prediction
    irisa_final_prediction = irisa_clf.predict(irisa_pred_df)[0]
    irisa_final_label_pred = number_to_label(irisa_final_prediction)

    #irisas prediction percentile
    sentiment_percentile = percentileofscore(irisa_X_train['sentiment'], irisa_pred_df['sentiment'])[0]
    ttr_percentile = percentileofscore(irisa_X_train['ttr'], irisa_pred_df['ttr'])[0]
    adjectives_percentile = percentileofscore(irisa_X_train['adjectives'], irisa_pred_df['adjectives'])[0]

    #LOHITS PREDICTIONS
    X_test_instance = [headline + " " + news_article]

    # Vectorize the test instance using the same TF-IDF vectorizer trained on the training data
    X_test_instance_tfidf = tfidf.transform(X_test_instance)
    
    # Make predictions for the test instance
    y_pred_instance = classifier.predict(X_test_instance_tfidf)
    
    y_pred_proba = classifier.predict_proba(X_test_instance_tfidf)
    positive_class_proba = y_pred_proba
    overall_score = (positive_class_proba[0][0] * 0.2) + (positive_class_proba[0][1] * 1) + (positive_class_proba[0][2] * 0.4) + (positive_class_proba[0][3] * 0.6) + (positive_class_proba[0][4] * 0.8) + (positive_class_proba[0][5] * 0.0)
    lohit_ngram_prediction = predict_label(overall_score)
    
    truth_scores = predict_tabular_classification_sample(project="dsc-180a-b09",
                                                         endpoint_id="4607809140427849728",
                                                         instance_dict={"article": news_article})
    reordered_indices = [truth_scores[0]['classes'].index(c) for c in ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']]
    classes_reordered = [truth_scores[0]['classes'][i] for i in reordered_indices]
    scores_reordered = [truth_scores[0]['scores'][i] for i in reordered_indices]

    scaling_dict = {"pants-fire": 0, "false": 0.2, "barely-true": 0.4, "half-true": 0.6, "mostly-true": 0.8, "true": 1}
    zipped_automl = zip(classes_reordered, scores_reordered)
    zipped_list = list(zipped_automl)
    
    #automl is a list of tuples with the label/confidence score
    final_automl_score = scale_and_combine(zipped_list)
    auto_ml_score_label = predict_label(final_automl_score)
    lohit_final_prediction = predict_label(final_automl_score)


    #NICKS PREDICTIONS
    #readability
    r = Readability(news_article)
    fk = r.flesch_kincaid()
    flesch_score = fk.score
    if flesch_score > 12:
        diff = flesch_score - 12
        fk_rating = 100 - (diff * 5)
    elif flesch_score < 8:
        diff = 8 - flesch_score
        fk_rating = 100 - (diff * 5)
    else:
        fk_rating = 100

    #sentiment
    sia = SentimentIntensityAnalyzer()
    moving_sentiment_value = 0
    number_of_paragraphs = 0
    paragraphs = news_article.split('\n\n')
    for i in paragraphs:
        cleaned_text = ' '.join(i.split()).replace("\'", '')
        compound_sentiment_score = sia.polarity_scores(cleaned_text)['compound']
        moving_sentiment_value += compound_sentiment_score
        number_of_paragraphs += 1
    overall_sentiment = moving_sentiment_value / number_of_paragraphs
    overall_sent_score = 100
    if overall_sentiment < -0.2:
        overall_sent_score = 100 + (overall_sentiment * 100)

    #clickbait
    if len(headline) > 0:
        article_title_processed = preprocess_text(headline)
        article_title_vectorized = count_vectorizer.transform([article_title_processed])
        clickbait_probability = clf_best.predict_proba(article_title_vectorized)
        confidence_not_clickbait = clickbait_probability[:, 0]
        confidence_not_clickbait = confidence_not_clickbait[0]
        nick_predicted_label = predict_label(confidence_not_clickbait)
    else:
        confidence_not_clickbait = 0

    if confidence_not_clickbait == 0:
        factors_combined = (overall_sent_score + fk_rating) / 2
        nicks_predicted_label = predict_label(factors_combined)
    else:
        factors_combined = ((confidence_not_clickbait * 100) + overall_sent_score + fk_rating) / 3
        nicks_predicted_label = predict_label(factors_combined)

    #HENRYS PREDICTIONS
    # Tokenize the single text example
    tokenized_news = tokenizer(news, return_tensors="pt", truncation=True, padding=True)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_henry = model_.to(device)
    
    # Move tokenized example to GPU
    tokenized_news_gpu = tokenized_news.to(device)
    
    # Extract BERT embeddings for the tokenized example
    with torch.no_grad():
        model_henry.eval()
        statement_embedding = model_henry(**tokenized_news_gpu).last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Use the RandomForestClassifier to predict the label for the single text example
    y_pred_news = rf_classifier.predict(statement_embedding)
    
    henry_final_prediction = y_pred_news[0]

    #combining all group mates predictive scores
    all_ratings = [irisa_final_label_pred,lohit_final_prediction,nicks_predicted_label,henry_final_prediction]
    labels_as_scores = []
    for i in all_ratings:
        labels_as_scores.append(label_to_score(i))
    pred_models_ave = sum(labels_as_scores) / len(labels_as_scores)
    final_pred_label = score_to_label(pred_models_ave)

    if len(headline) > 0:
        pred_score_output = f"""The overall score created from our predictive models is {pred_models_ave}. This means the article has been evaluated to be {final_pred_label}. The individual scores of the predictive models are as follows.
        Full-text n-gram analysis: {lohit_ngram_prediction}
        Full-text BERT embedding prediction: {henry_final_prediction}
        Google AUTO ML full-text analysis: {auto_ml_score_label}
        Readability Score: {round(fk_rating,2)}
        Not Clickbait Probability: {round(confidence_not_clickbait*100,2)}%
        Quality of Writing Percentile: {round(ttr_percentile,2)}%
        Sensationalism Score Percentile: {round(adjectives_percentile,2)}%
        Sentiment Score Percentile: {round(sentiment_percentile,2)}%"""
    else:
        pred_score_output = f"""The overall score created from our predictive models is {pred_models_ave}. This means the article has been 
        evaluated to be {final_pred_label}. The individual scores of the predictive models are as follows.
        Full-text n-gram analysis: {lohit_ngram_prediction}
        Full-text BERT embedding prediction: {henry_final_prediction}
        Google AUTO ML full-text analysis: {auto_ml_score_label}
        Readability Score: {round(fk_rating,2)}
        Quality of Writing Percentile: {round(ttr_percentile,2)}
        Sensationalism Score Percentile: {round(adjectives_percentile,2)}
        Sentiment Score Percentile: {round(sentiment_percentile,2)}"""
        
    #Pre-processed examples output
    if example_news_provider == "ABC" or example_news_provider == "ABC" or example_news_provider == "Abc":
        match = re.search(r'(\d+(?:\.\d+)?)\/100', abc_final_output)
        score = float(match.group(1))
        overall_score = (pred_models_ave * 0.2) + (score * 0.8)
        overall_score = round(overall_score, 2)
        if overall_score < 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall pants-on-fire rating. Most of, or all of the information in this article is falsified or lacks huge amounts of context and is very misleading. This article should be read with extreme caution. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 33.333 and overall_score > 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall false rating. Most of the information in this article is falsified or lacks context and therefore can be quite misleading. This article should be read with very much caution as much of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 50 and overall_score > 33.333:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall barely-true rating. A majority of the information in this article is falsified or lacks context which could make it misleading. This article should be read with caution as over half of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 66.666 and overall_score > 50:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall half-true rating. This news article contains both truthful and false or misleading content. This article should be read with caution as some of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 83.333 and overall_score > 66.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall mostly-true rating. This news article contains mostly truthful information, but still contains some false or misleading content. This article should be read with caution as a small set of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""   
        else:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall true rating. This news article contains almost entirely truthful information. This article contains highly accurate and non-misleading information and can be reliably trusted. Still use caution though as not every statement here may be completely true. Please also be aware that potentially not all statements in the text could be rated."""
        history_output.append([news_article, abc_final_output])
        return history_output, history_output, pred_score_output, overall_synopsis
    elif example_news_provider == "NPR" or example_news_provider == "NPR" or example_news_provider == "Npr":
        match = re.search(r'(\d+(?:\.\d+)?)\/100', npr_final_output)
        score = float(match.group(1))
        overall_score = (pred_models_ave * 0.2) + (score * 0.8)
        overall_score = round(overall_score, 2)
        if overall_score < 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall pants-on-fire rating. Most of, or all of the information in this article is falsified or lacks huge amounts of context and is very misleading. This article should be read with extreme caution. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 33.333 and overall_score > 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall false rating. Most of the information in this article is falsified or lacks context and therefore can be quite misleading. This article should be read with very much caution as much of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 50 and overall_score > 33.333:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall barely-true rating. A majority of the information in this article is falsified or lacks context which could make it misleading. This article should be read with caution as over half of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 66.666 and overall_score > 50:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall half-true rating. This news article contains both truthful and false or misleading content. This article should be read with caution as some of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 83.333 and overall_score > 66.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall mostly-true rating. This news article contains mostly truthful information, but still contains some false or misleading content. This article should be read with caution as a small set of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""   
        else:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall true rating. This news article contains almost entirely truthful information. This article contains highly accurate and non-misleading information and can be reliably trusted. Still use caution though as not every statement here may be completely true. Please also be aware that potentially not all statements in the text could be rated."""
        history_output.append([news_article, npr_final_output])
        return history_output, history_output, pred_score_output, overall_synopsis
    elif example_news_provider == "FOX" or example_news_provider == "fox" or example_news_provider == "Fox":
        match = re.search(r'(\d+(?:\.\d+)?)\/100', fox_final_output)
        score = float(match.group(1))
        overall_score = (pred_models_ave * 0.2) + (score * 0.8)
        overall_score = round(overall_score, 2)
        if overall_score < 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall pants-on-fire rating. Most of, or all of the information in this article is falsified or lacks huge amounts of context and is very misleading. This article should be read with extreme caution. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 33.333 and overall_score > 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall false rating. Most of the information in this article is falsified or lacks context and therefore can be quite misleading. This article should be read with very much caution as much of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 50 and overall_score > 33.333:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall barely-true rating. A majority of the information in this article is falsified or lacks context which could make it misleading. This article should be read with caution as over half of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 66.666 and overall_score > 50:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall half-true rating. This news article contains both truthful and false or misleading content. This article should be read with caution as some of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 83.333 and overall_score > 66.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall mostly-true rating. This news article contains mostly truthful information, but still contains some false or misleading content. This article should be read with caution as a small set of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""   
        else:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall true rating. This news article contains almost entirely truthful information. This article contains highly accurate and non-misleading information and can be reliably trusted. Still use caution though as not every statement here may be completely true. Please also be aware that potentially not all statements in the text could be rated."""
        history_output.append([news_article, fox_final_output])
        return history_output, history_output, pred_score_output, overall_synopsis
        
    
    #GEN AI
    #instantiating gemini pro model
    PROJECT_ID = "gen-lang-client-0321728687"
    REGION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = generative_models.GenerativeModel("gemini-pro")
    config = {"max_output_tokens": 2048, "temperature": 0.0}
    
    safety_config = {
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    chat = model.start_chat()

    #PerspectiveAPI output check instantiation
    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=PERSPECTIVE_API_KEY,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
        )
    
    #chunking news article for improved processing
    chunked_article_list = tokenize_into_chunks(news_article, 50)
    
    #getting context and fact checks from vector database based on the provided input
    all_response_text = []
    context_list = []
    for i in range(len(chunked_article_list)):
        input = chunked_article_list[i]
        context = RAG_CONTEXT_VDB.query(
            query_texts=[input],
            n_results=7,
        )
        context_list.append(context)
        
    fact_checks_list=[]
    for i in range(len(chunked_article_list)):
        input = chunked_article_list[i]
        fact_checks = RAG_STATEMENTS_VDB.query(
            query_texts=[input],
            n_results=7,
        )
        fact_checks_list.append(fact_checks)

    #creating history list so that gen ai model has additional context when analyzing chunked statements 
    for i in range(len(context_list)):
        input=chunked_article_list[i]
        fact_checks = fact_checks_list[i]
        context = context_list[i]
        
        prev_chunk = chunked_article_list[i - 1] if i > 0 else None
        next_chunk = chunked_article_list[i + 1] if i + 1 < len(chunked_article_list) else None
        history = [prev_chunk, input, next_chunk]
        
        #re-ranking RAG results for fact check statements from RAG_STATEMENTS_VDB
        statement_rerank_list = []
        for j in range(len(fact_checks['ids'][0])):
            reranking_statementSearch = [input, fact_checks['documents'][0][j]]
            statement_rerank_list.append(reranking_statementSearch)
    
        
        scores = reranker.compute_score(statement_rerank_list)
        combined_statement_scores = list(zip(scores, statement_rerank_list, fact_checks['metadatas'][0]))
        sorted_combined_data = sorted(combined_statement_scores, key=lambda x: x[0], reverse=True)
        sorted_statement_scores, sorted_statement_rerank_list, sorted_factCheck_metadata = zip(*sorted_combined_data)
    
        #re-ranking RAG results for context statements from RAG_CONTEXT_VDB
        context_rerank_list = []
        for k in range(len(context['ids'][0])):
            reranking_contextSearch = [input, context['documents'][0][k]]
            context_rerank_list.append(reranking_contextSearch)
            
        scores = reranker.compute_score(context_rerank_list)
        combined_context_scores = list(zip(scores, context_rerank_list, context['metadatas'][0]))
        sorted_combined_data = sorted(combined_context_scores, key=lambda x: x[0], reverse=True)
        sorted_context_scores, sorted_context_rerank_list, sorted_context_metadata = zip(*sorted_combined_data)

        #getting top 3 most relevant pieces of context and fact checks from RAG
        context_window = 3
        prepared_context = []
        prepared_fact_checks = []
        for i in range(context_window):
            prepared_context.append([sorted_context_metadata[i], sorted_context_rerank_list[i][1]])
            prepared_fact_checks.append([sorted_factCheck_metadata[i], sorted_statement_rerank_list[i][1]])

        #Changing chunks from list of strings to one combined string for Gen AI processing
        chunk_history_string = ''
        for chunk in history:
            if chunk != None:
                chunk_history_string += chunk + " "

       

        responses = model.generate_content(f""" Answer the question below marked inside <<<>>> in a full sentence based on the
        knowledge I have provided you below, as well as information you already have access to answer the question.
        Use the additional information I've provided below within the ((())) symbols to help you. 
        (((
        Refer to these fact checked statements as well to determine your answer and be sure to pay close attention to the 
        metadata that is provided: {prepared_fact_checks}.
        Use the following context to help answer the question: {prepared_context}.
        You may also use the chat history provided to help you understand the context better if available: {chunk_history_string}.
        Ensure that you use all this information and think about this question step-by-step using the provided information.
        Make sure you provide a short explanation of why you chose that score.
        )))
        <<<
        Question: How true is the following statement on a scale of 1-100? + {input}. You must provide the score in this format Score:XX., 
        followed by your short explanation.
        >>>
       """,
            generation_config=config,
            stream=True,
            safety_settings=safety_config,                          
        )
        
        #generating initial response with prompt template
       #  responses = model.generate_content(f"""Answer the question below marked inside <<<>>> in a full sentence based on the
       #  knowledge you already have access to answer the question.
    
       #  If you are not very sure of your answer to the question, then use the additional information I've provided below within the 
       #  ((())) symbols to help you.
       #  (((
       #  Refer to these fact checked statements as well to determine your answer and be sure to pay close attention to the 
       #  metadata that is provided: {prepared_fact_checks}.
       #  Use the following context to help answer the question: {prepared_context}.
       #  You may also use the chat history provided to help you understand the context better if available: {chunk_history_string}.
       #  Make sure you provide a short explanation of why you chose that score.
       #  )))
       #  <<<
       #  Question: How true is the following statement on a scale of 1-100? + {input}. You must provide the score in this format Score:XX., 
       #  followed by your short explanation.
       #  >>>
       # """,
       #      generation_config=config,
       #      stream=True,
       #      safety_settings=safety_config,                          
       #  )

        
        #obtaining individual responses
        response_text = ""
        response_text += "Statement: " + input
        for response in responses:
            try:
                response_text += response.text
            except (IndexError, ValueError) as e:
                continue
        response_text = response_text.replace("\n\n", ". ")
        all_response_text.append(response_text)
        
    #combining all responses    
    entire_text_string = ""
    for text in all_response_text:
        entire_text_string += text
    cleaned_text = entire_text_string
    
    #this section is finding and removing the statements that can't be rated by the chatbot
    unratable_sentences = []
    rated_sentences = []

    final_text = ""
    for response in all_response_text:
        if "article does not" in response.lower() or "context does not" in response.lower() or "Statement: Score" in response:
            unratable_sentences.append(response)
        else:
            rated_sentences.append(response)
            final_text += response
    
    not_enough_context = len(unratable_sentences)
    enough_context = len(rated_sentences)
    all_statements_count = len(all_response_text)


    #total score calculation with regex
    pattern = r'\b\s*Score:\s*(\d+)\.'
    total_score = 0
    matches = re.findall(pattern, cleaned_text)
    for match in matches:
        score = int(match)
        total_score += score
    if enough_context == 0:
        average_score = 0
    else:
        average_score = total_score / enough_context
    rounded_average = round(average_score, 1)

    #creating output in nice format for user
    tweaking_output = re.sub(r'(Score:\s*\d+\.)(?!\s*Explanation:)', r'\1 Explanation:', final_text)
    parts = re.split(r"(?=Statement:)", tweaking_output)
    split_parts=[]
    # Clean each part and add to split_parts
    for part in parts:
        cleaned_part = output_clean(part)
        split_parts.append(cleaned_part)
    
    # Initialize variables to store lowest scores and their respective entries
    lowest_scores = [(float('inf'), ''), (float('inf'), ''), (float('inf'), '')]

    # Iterate through each string entry
    statements_rated=0
    for entry in split_parts:
        pattern = r'\b\s*Score:\s*(\d+)\.'
        # Find all occurrences of "Score: " followed by a number until a "."
        scores = re.findall(pattern, entry)
        # Convert scores to integers and update lowest_scores if necessary
        for score in scores:
            statements_rated+=1
            score_int = int(score)
            if score_int < lowest_scores[-1][0]:
                lowest_scores[-1] = (score_int, entry)
                lowest_scores.sort()    
    average_score = total_score / statements_rated
    rounded_average = round(average_score, 1)
    
    # Extract the entries for the three lowest scores
    lowest_entries = [entry for score, entry in lowest_scores]

    #reformatting for better readability
    summary_output = ""
    for statement in lowest_entries:
        # Replace "Statement:", "Score:", and "Explanation:" with a new line followed by the keyword
        formatted_statement = re.sub(r'(Statement:|Score:|Explanation:)', r'\n\1', statement)
        # Append the formatted statement to the output
        summary_output += formatted_statement.strip() + "\n"
    
        # Add a new line after each statement
        summary_output += "\n"

    output_intro = f"""{statements_rated} out of {all_statements_count} statements in the text could be rated. The following score and explanation is based on these {statements_rated} statements. The average truthfulness score from these {all_statements_count} statements is {rounded_average}/100. Some of the lowest rated statements are provided below."""

    output = output_intro + "\n\n" + summary_output

    #Perspective API output safety check
    analyze_request = {
      'comment': { 'text': output},
      'requestedAttributes': requestedAttributes
    }
    response = client.comments().analyze(body=analyze_request).execute()
    
    attributes_surpassed = []
    for key in response['attributeScores']:
        if response['attributeScores'][key]['summaryScore']['value'] > attributeThresholds[key]:
            attributes_surpassed.append((key, response['attributeScores'][key]['summaryScore']['value']))
    
    #crafting output warning message if necessary or regular output message  
    
    if len(attributes_surpassed) == 1:
        attributes_violated = ""
        for i in attributes_surpassed:
            attributes_violated += i[0] + " "
        warning_message = f"""We're sorry, the output message surpasses our threshold for the {attributes_violated}category so we cannot safely provide a response. Please try again with a different input."""
        history_output.append([news_article, warning_message])
        
    elif len(attributes_surpassed) > 1:
        attributes_violated = ""
        counter = 1
        attributes_count = len(attributes_surpassed)
        for i in attributes_surpassed:
            attributes_violated += i[0] + " "
            if counter < attributes_count:
                attributes_violated += "and "
            counter += 1
        warning_message = f"""We're sorry, the output message surpasses our threshold for the {attributes_violated}categories so we cannot safely provide a response. Please try again with a different input."""
        history_output.append([news_article, warning_message])

    else:
        overall_score = (pred_models_ave * 0.2) + (rounded_average * 0.8)
        overall_score = round(overall_score, 2)
        if overall_score < 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall pants-on-fire rating. Most of, or all of the information in this article is falsified or lacks huge amounts of context and is very misleading. This article should be read with extreme caution. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 33.333 and overall_score > 16.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall false rating. Most of the information in this article is falsified or lacks context and therefore can be quite misleading. This article should be read with very much caution as much of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 50 and overall_score > 33.333:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall barely-true rating. A majority of the information in this article is falsified or lacks context which could make it misleading. This article should be read with caution as over half of the information here may be false. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 66.666 and overall_score > 50:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall half-true rating. This news article contains both truthful and false or misleading content. This article should be read with caution as some of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""
        elif overall_score < 83.333 and overall_score > 66.666:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall mostly-true rating. This news article contains mostly truthful information, but still contains some false or misleading content. This article should be read with caution as a small set of the information here may be false or lacking context. Please also be aware that potentially not all statements in the text could be rated."""   
        else:
            overall_synopsis = f"""The overall truthfulness score after combining the Generative and Predictive AI results is {overall_score}/100. This means that the article has an overall true rating. This news article contains almost entirely truthful information. This article contains highly accurate and non-misleading information and can be reliably trusted. Still use caution though as not every statement here may be completely true. Please also be aware that potentially not all statements in the text could be rated."""
        history_output.append([news_article, output])
    return history_output, history_output, pred_score_output, overall_synopsis


# #### Example article load

# In[326]:


#loading live examples
abc_headline, abc_content, abc_link = abc_updated_news()
npr_headline, npr_content, npr_link = npr_updated_news()
fox_headline, fox_content, fox_link = fox_updated_news()


# In[327]:


npr_output = GenAI_article_truth_processing(npr_content, [], [], npr_headline)
npr_output = npr_output[0][0][1]
npr_output = npr_output_parse(npr_output)
npr_final_output = "Here is a link to the analyzed article. " + npr_link + " \n " + npr_output


# In[328]:


fox_output = GenAI_article_truth_processing(fox_content,[], [], fox_headline)
fox_output = fox_output[0][0][1]
fox_output = fox_output_parse(fox_output)
fox_final_output = "Here is a link to the analyzed article. " + fox_link + " \n " + fox_output


# In[329]:


abc_output = GenAI_article_truth_processing(abc_content,[], [], abc_headline)
abc_output = abc_output[0][0][1]
abc_output = abc_output_parse(abc_output)
abc_final_output = "Here is a link to the analyzed article. " + abc_link + " \n " + abc_output


# #### Article testing examples

# In[205]:


news = """Months after leaving the White House, former President Donald Trump began plotting his return to Wall Street. That return, delayed by years of regulatory and legal hurdles, is now on the verge of becoming a reality — and it could make Trump a fortune.

US regulators have finally given the green light to a controversial merger between Truth Social owner Trump Media & Technology Group and a blank-check company. The blessing from the Securities and Exchange Commission removes the last major obstacle holding back the deal.

The merger, if approved by shareholders, would pave the way for Trump Media to become a publicly-traded company — one where Trump will own a dominant stake that could be worth billions.

Digital World Acquisition Corp., the blank-check firm, announced that on Wednesday the SEC signed off on the merger proxy for the deal. A date for a shareholder vote will be set by Friday.

“It does look like this deal is going to reach the finish line now — after more than two years of delays,” said Jay Ritter, a finance professor at the University of Florida.

Trump stake could be worth $4 billion
Shares of Digital World, a special purpose acquisition company, or SPAC, spiked 15% on the major milestone. The stock has nearly tripled this year, fueled by Trump’s political success in the Republican presidential primary, and now the merger progress.

Ritter estimates the merger could pave the way for about $270 million of cash coming into Trump Media, funds the company could fuel Truth Social’s growth.

Trump is set to hold a dominant position in the newly-combined company, owning roughly 79 million shares, according to new SEC filings.

The former president’s stake would be valued at $4 billion based on Digital World’s current trading price of about $50.

Of course, as Ritter notes, it would be very difficult for Trump to translate that paper wealth into actual cash.

Not only would Trump be subject to a lock-up period that would prevent he and other insiders from selling until six months after the merger, but the new company’s fortunes would be closely associated with the former president. That could make it difficult for Trump to sell even after the lock-up period expires.

‘This is a meme stock’
Moreover, there are major questions about the sky-high valuation being placed on this media company.

“This is a meme stock. The valuation is totally divorced from the fundamental value of the company,” said Ritter.

Digital World’s share price values the company at up to about $8 billion on a fully diluted basis, which includes all shares and options that could be converted to common stock, according to Ritter.

He described that valuation as “crazy” because Trump Media is generating little revenue and burning through cash.

New SEC filings indicate Trump Media’s revenue amounted to just $1.1 million during the third quarter. The company posted a loss of $26 million.

Since the merger was first proposed in October 2021, legal, regulatory and financial questions have swirled about the transaction.

In November, accountants warned that Trump Media was burning cash so rapidly that it might not survive unless the long-delayed merger with Digital World is completed soon.

Shareholder vote looms
Now, Trump execs are cheering the green light from the SEC.

“Truth Social was created to serve as a safe harbor for free expression and to give people their voices back,” Trump Media CEO Devin Nunes, a former Republican congressman, said in a statement. “Moving forward, we aim to accelerate our work to build a free speech highway outside the stifling stranglehold of Big Tech.”

Eric Swider, Digital World’s CEO, described the SEC approval as a “significant milestone” and said executives are “immensely proud of the strides we’ve taken towards advancing” the merger.

One of the final remaining hurdles is for Digital World shareholders to approve the merger in an upcoming vote.

The shareholders have enormous incentive to approve the deal because if the merger fails, the blank-check firm would be forced to liquidate. That would leave shareholders with just $10 a share, compared with $50 in the market today.

“Anyone who holds shares and votes against the merger is crazy,” said Ritter, the professor.

“Then again, I might argue that everyone holding DWAC shares is crazy,” he added, referring to the company’s thin revenue and hefty valuation.

Matthew Tuttle, CEO of Tuttle Capital Management, said he’s not surprised by the ups and downs surrounding this merger.

“The thing about Trump and anything related to Trump is, love him or hate him, there is going to be drama,” said Tuttle, who purchased options to buy Digital World shares in his personal account. “Really, I would not have expected anything less.”

Going forward, Tuttle said Trump Media’s share price will live and die by how everything plays out for Trump personally — from his legal troubles to his potential return to the White House.

“Anything bullish for Trump is going to be bullish for the stock,” said Tuttle.

Trump is no stranger to Wall Street, where he has a history, one marked by bankruptcies.

Although Trump has never filed for personal bankruptcy, he has filed four business bankruptcies — all of them linked to casinos he used to own in Atlantic City."""


# In[206]:


text = """ Are Americans paying nearly $500 for an inhaler that would cost just $7 overseas?

U.S. Sen. Tammy Baldwin, D-Wis., says there is a vast difference in the cost of prescriptions in the United States and the rest of the world. 

"Big drug companies charge as little as $7 for an inhaler overseas and nearly $500 for the exact same one here in the US," Baldwin said Feb. 1 in a Facebook post. "That has got to end. We've got to hold Big Pharma accountable for their price-gouging tactics. I won't stop fighting until we do."

That massive cost difference piqued our interest.

How much would patients pay? 
When we asked for backup information, Baldwin’s campaign staff directed us to drug pricing websites, news articles and news releases on the cost of Combivent Respimat (ipratropium bromide and albuterol), a combination medication used to treat chronic obstructive pulmonary disease (COPD). 

Combivent Respimat is available only as a brand-name medication and not available in generic form, according to Medical News Today, which pointed out that the actual price a patient would pay for the medication depends on type of insurance plan, location and pricing at the patient’s pharmacy. Medicare does cover Combivent Respimat. 

According to Drugs.com, a pricing website, Combivent Respimat costs about $525 for a supply of 4 grams, depending on the pharmacy. 

It’s also important to note, that on a practical basis, because of insurance and Medicare coverage, few people in the United States would actually pay $500 out of pocket

"Quoted prices are for cash-paying customers and are not valid with insurance plans," the website says  says. 

Another online drug pricing guide, GoodRx, puts the price of Combivent Respimat between about $477 and $584 at Madison, Wisconsin, pharmacies:

Walgreens —    $508.39 

Walmart —----   $514.45

CVS Pharmacy-$508.14

Hy-Vee —--------$477.97

Costco —---------$584.59

Target —----------$508.14

FEATURED FACT-CHECK

Instagram posts
stated on February 15, 2024 in an Instagram post
Because “17 million immigrants” were “let in” the U.S, “ foot and mouth disease is back. We got rid of that fifty years ago.”
truefalse
By Jeff Cercone • February 16, 2024
Metro Market —-$511.00

Pick ’n Save—---$511.00

So, Baldwin is on target on the cost in the US.

What about overseas?
According to a Jan. 8 news release from U.S. Sen. Bernie Sanders, I-Vt., Combivent Respimat sold for just $7 In France.

Sanders, chairman of the Senate Committee on Health, Education, Labor, and Pensions, sent letters to the CEOs of four pharmaceutical companies announcing an investigation into the high prices the companies are charging for inhalers. Baldwin and Democratic Sens. Ben Ray Luján of New Mexico and Ed Markey of Massachusetts also signed the letters.

The letters were sent to the four biggest manufacturers of inhalers sold in the United States — AstraZeneca, Boehringer Ingelheim, GlaxoSmithKline (GSK) and Teva.

"It is beyond absurd that Boehringer Ingelheim charges $489 for Combivent Respimat in the United States, but just $7 in France," Sanders said in the news release.

The news release said the Committee’s source for the price of Combivent Respimat in France was the Navlin international drug pricing database. 

Baldwin, in the news release, accuses companies of "jacking up prices and turning record profits."

Experts weigh in 
Dr. William B. Feldman noted that Baldwin is referring to list prices here — which are the prices that uninsured patients in the U.S. pay and the prices to which out-of-pocket costs are often tied.

"Manufacturers give sizable (confidential) rebates to insurers, and so the net prices for inhalers in the U.S. are below list prices — but still much higher than the net prices abroad," Feldman said in an email to PolitiFact Wisconsin. 

Feldman, who works at Brigham and Women’s Hospital in Boston and Harvard Medical School, said a key reason inhaler prices remain so high in the U.S. is that there is very little generic competition. 

"Brand-name manufacturers have erected large patent thickets that keep generic competitors off the market," Feldman said. " Inhaler prices are low elsewhere, in part, because governments negotiate prices based on the value of the drugs compared to existing therapies."

David Kreling, professor emeritus in the School of Pharmacy at the University of Wisconsin-Madison, said the U.S. price quoted by Baldwin sounds about right.

"The $500 number may be in the ballpark for U.S. patented (brand-name, newer) drugs," Kreling said in an email to PolitiFact Wisconsin. "That would be consistent with my understanding of market data on sales by firms in the U.S. Things in the $7 range, here, only reside within the off-patent generic drug market (where we have low prices, sometimes at or near lowest in the world)." 

Our ruling
Baldwin said "big drug companies charge as little as $7 for an inhaler overseas and nearly $500 for the exact same one here in the US."

Our review, and that of experts, found the numbers checked out.

Experts cite a variety of reasons for the price differences, including very little generic competition in the United States, and few people in the United States would actually pay $500 out of pocket because of insurance and Medicare coverage. 

For a statement that is accurate but needs clarification or additional information, our rating is Mostly True."""


# ## Gradio (Website) Implementation

# In[479]:


block = gr.Blocks()
prompt_placeholder = "Insert your news article here!"
headline_placeholder = "Paste your news articles headline here!"
with block:
    gr.Markdown("""<h1><center>Generative AI News Article Truthfulness Evaluator</center></h1>
    """)
    examples = gr.Dropdown(["ABC", "NPR", "FOX"], label="News Provider", info="Take any of the news providers in the dropdown below and type the name of the provider exactly how you see it into the Article Content Textbox below to get an up-to-date example articles evaluation!")
    message = gr.Textbox(placeholder=prompt_placeholder, label="Article Content",info="Paste any news article here, or type in the name of one of the news providers in the dropdown above.")
    headline = gr.Textbox(placeholder=headline_placeholder, label = "Headline", info="Paste your news articles headline here if available for improved results.")
    chatbot = gr.Chatbot()
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(make_plot, inputs=[message], outputs=gr.Plot())
    pred_info = gr.Textbox(placeholder="Predictive Analysis", label="Predictive Analysis")
    overall_synopsis = gr.Textbox(placeholder="Overall Rating", label="Overall Rating")
    submit.click(GenAI_article_truth_processing, inputs=[message, state, examples, headline], outputs=[chatbot, state, pred_info, overall_synopsis])
block.launch(share=True, share_server_address="disinformation-destroyers.com:7000")


# In[ ]:




