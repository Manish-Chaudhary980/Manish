
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text

st.title("📰 Fake News Detection System")

input_text = st.text_area("Enter News Article Text:", height=200)
if st.button("Predict"):
    if input_text.strip():
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"🧠 Prediction: {'FAKE' if prediction else 'REAL'}")
    else:
        st.warning("Please enter some text.")
