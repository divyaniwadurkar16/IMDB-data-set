import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation, numbers, special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("logistic_regression_model.pkl")
        return vectorizer, model
    except FileNotFoundError:
        st.error("❌ Model files not found!")
        st.stop()

tfidf_vectorizer, model = load_models()

# -----------------------------
# App Title
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below and click **Predict Sentiment**."
)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "✍ Enter Movie Review",
    height=200,
    placeholder="Example: This movie was absolutely amazing! The acting was brilliant..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a movie review.")
    else:

        with st.spinner("Analyzing sentiment..."):

            cleaned_input = preprocess_text(user_input)

            input_vector = tfidf_vectorizer.transform([cleaned_input])

            prediction = model.predict(input_vector)[0]

            # Prediction Probability
            probability = model.predict_proba(input_vector)[0]

            confidence = max(probability) * 100

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"### Confidence: **{confidence:.2f}%**")

        with st.expander("Processed Review"):
            st.write(cleaned_input)
