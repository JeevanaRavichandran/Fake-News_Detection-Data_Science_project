import streamlit as st
import joblib
import re

# Load model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

st.title("📰 Fake News Detector")
user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    if pred == 0:
        st.error("🚨 This news seems FAKE.")
    else:
        st.success("✅ This news seems REAL.")
