import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article to check whether it is FAKE or REAL")

user_input = st.text_area("Paste news content here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        probabilities = model.predict_proba(vector)

        fake_prob = round(probabilities[0][0] * 100, 2)
        real_prob = round(probabilities[0][1] * 100, 2)

        if prediction[0] == 1:
            st.success(f"✅ This news appears to be REAL with {real_prob}% confidence.")
        else:
            st.error(f"❌ This news appears to be FAKE with {fake_prob}% confidence.")
