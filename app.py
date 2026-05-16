import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    return "Fake News" if prediction == 0 else "Real News"

st.title("Fake News Detection System")
st.write("Enter a news article below to check whether it is Fake or Real.")

news_input = st.text_area("Enter News Article")

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        result = predict_news(news_input)

        if result == "Fake News":
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")