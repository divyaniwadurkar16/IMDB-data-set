import streamlit as st
import re
import nltk
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Predictor",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------
# Download NLTK Data
# -------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------
# Text Preprocessing
# -------------------------------
def preprocess_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    vectorizer_path = Path("tfidf_vectorizer.pkl")
    model_path = Path("logistic_regression_model (1).pkl")

    if not vectorizer_path.exists():
        st.error("❌ tfidf_vectorizer.pkl not found.")
        st.stop()

    if not model_path.exists():
        st.error("❌ logistic_regression_model.pkl not found.")
        st.stop()

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    return vectorizer, model

tfidf_vectorizer, model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")

review = st.text_area(
    "Enter your movie review:",
    height=200
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        cleaned = preprocess_text(review)

        vector = tfidf_vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        confidence = model.predict_proba(vector).max() * 100

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"**Confidence:** {confidence:.2f}%")
