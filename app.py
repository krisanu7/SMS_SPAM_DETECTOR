import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---------------- CSS Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    h1, h2, h3 {
        color: #38bdf8;
        text-align: center;
    }

    textarea {
        background-color: #020617 !important;
        color: #f8fafc !important;
        border-radius: 10px;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #38bdf8;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 8px 20px;
    }

    .stButton>button:hover {
        background-color: #0ea5e9;
        color: white;
    }

    /* Spam text style (RED) */
    .spam-text {
        color: red;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
    }

    /* Not spam text style (normal) */
    .not-spam-text {
        color: #f8fafc;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- NLP Setup ----------------
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------------- App UI ----------------
st.title('📩 Spam Message Classifier')

input_sms = st.text_area("✍️ Enter the message")

if st.button('🔍 Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.markdown("<div class='spam-text'>🚨 Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='not-spam-text'>✅ Not Spam</div>", unsafe_allow_html=True)
