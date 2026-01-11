# ===============================
# SMS SPAM CLASSIFIER - FULL PIPELINE
# ===============================

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from wordcloud import WordCloud

# ===============================
# NLTK downloads (Python 3.12 safe)
# ===============================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ===============================
# Load dataset
# ===============================
path ="spam.csv"
df = pd.read_csv(path, encoding='latin-1')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates()

# ===============================
# Feature Engineering
# ===============================
df['num_char'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# ===============================
# Text Preprocessing
# ===============================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    y = []
    for word in words:
        if word.isalnum():
            if word not in stop_words and word not in string.punctuation:
                y.append(ps.stem(word))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

# ===============================
# WordCloud (Spam & Ham)
# ===============================
wc = WordCloud(width=500, height=500, background_color='white')

spam_wc = wc.generate(
    df[df['target'] == 1]['transformed_text'].str.cat(sep=" ")
)

ham_wc = wc.generate(
    df[df['target'] == 0]['transformed_text'].str.cat(sep=" ")
)

# ===============================
# Most common words
# ===============================
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text']:
    spam_corpus.extend(msg.split())

spam_df = pd.DataFrame(
    Counter(spam_corpus).most_common(30),
    columns=['word', 'count']
)

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text']:
    ham_corpus.extend(msg.split())

ham_df = pd.DataFrame(
    Counter(ham_corpus).most_common(30),
    columns=['word', 'count']
)

# ===============================
# Model Building
# ===============================
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# ===============================
# Save Model
# ===============================
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
