from google.colab import files
import pandas as pd

uploaded = files.upload()

data = pd.read_csv("news.csv")

print(data.head())

import numpy as np

# sklearn library for tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# sklearn libraries for models and evaulation metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# nltk for preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# matpilotlib and seaborn libraries for visualize
import matplotlib.pyplot as plt
import seaborn as sns

# download nltk datasets
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("wordnet", quiet=True)

print("Libraries and NLTK datasets loaded!")

data = pd.read_csv("news.csv")
data = data.drop(["title", "Unnamed: 0"], axis=1)
data = data.dropna()

# stopwords and lemmatizer assigned
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# text preprocessing for lowercase, tokenize, removal stopwords, punkt and alnum
def preprocess_text(text):
    # lowercase text
    text = text.lower()
    # tokenize text
    tokens = word_tokenize(text)
    # removal stopwords, punkt and only alpha char
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words
    ]
    # reconstruct tokens
    return " ".join(clean_tokens)

# create new col for cleaned text
data["cleaned_text"] = data["text"].apply(preprocess_text)
