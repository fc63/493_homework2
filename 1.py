from google.colab import files
import pandas as pd

uploaded = files.upload()

data = pd.read_csv("news.csv")

print("Succesful upload for news.csv")

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

# we assign data = our dataset and drop the columns and empty rows that we will not use
data = pd.read_csv("news.csv")
data = data.drop(["title", "Unnamed: 0"], axis=1)
data = data.dropna()

# stopwords and lemmatizer assigned
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# text preprocessing for lowercase, tokenize, removal stopwords, punkt and non alpha char
def preprocess_text(text):
    # lowercase text
    text = text.lower()
    # tokenize text
    tokens = word_tokenize(text)
    # removal stopwords, punkt and non alpha char
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words
    ]
    # reconstruct tokens
    return " ".join(clean_tokens)

# create new col for cleaned text
data["cleaned_text"] = data["text"].apply(preprocess_text)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # we get maximum 5000 feature/word for tf idf
X = tfidf_vectorizer.fit_transform(data["cleaned_text"]).toarray()  # TF-IDF feature matrix
y = data["label"].map({"REAL": 0, "FAKE": 1})  # we assigned the labels as 1 and 0, 0 for real, 1 for fake, we will use boolean logic

# we divide 80% of the vectorized data we obtain for the training set and 20% for the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Feature extraction and dataset splitting with TF-IDF completed!")
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# naive bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)  # model training
y_pred_nb = nb_model.predict(X_test)  # system outputs

# logistic regression classifier
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)  # model training
y_pred_lr = lr_model.predict(X_test)  # system outputs

# evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.xlabel("System Output")
    plt.ylabel("Correct Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

results = {
    "Model": ["Naive Bayes", "Logistic Regression"],
    "Accuracy": [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_lr)],
    "Precision": [precision_score(y_test, y_pred_nb), precision_score(y_test, y_pred_lr)],
    "Recall": [recall_score(y_test, y_pred_nb), recall_score(y_test, y_pred_lr)],
    "F1-Score": [f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_lr)],
}

results_df = pd.DataFrame(results)
print(results_df)
print("\n\n")

# naive bayes evaluation
evaluate_model(y_test, y_pred_nb, "Naive Bayes Classifier")

# logistic regression evaluation
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
