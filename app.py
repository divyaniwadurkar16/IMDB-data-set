import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (only if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Tokenize words
    words = text.split()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load the TF-IDF vectorizer and the trained model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')

# Streamlit App Title
st.title("IMDB Movie Review Sentiment Predictor")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your review here:", height=200)

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_input = preprocess_text(user_input)
        
        # Transform text using the loaded TF-IDF vectorizer
        # Ensure the vectorizer expects a list of documents
        input_vectorized = tfidf_vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(input_vectorized)
        
        # Display result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter a review to predict sentiment.")
