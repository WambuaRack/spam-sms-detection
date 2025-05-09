import streamlit as st
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

STOPWORDS = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = re.sub(r'\d+', '', text.lower())
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    return ' '.join(tokens)

df['cleaned'] = df['message'].apply(clean_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_sms(message):
    cleaned = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return 'Spam 🚫' if prediction[0] == 1 else 'Not Spam ✅'

# Streamlit UI
st.title("📩 SMS Spam Detection App")
st.write("Enter a message below and find out if it's spam or not!")

user_input = st.text_area("Enter SMS Message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        result = predict_sms(user_input)
        st.success(f"Prediction: **{result}**")

st.markdown("---")
st.caption("Built with ❤️ by shedrack. Powered by Python & Streamlit.")
