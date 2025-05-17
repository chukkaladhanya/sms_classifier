import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [ps.stem(token) for token in tokens]
    return " ".join(tokens)

# Streamlit page configuration
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput>div>div>input, .stTextArea textarea {
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to classify it as **Spam** or **Ham** (Not Spam).")

# User input
with st.form("sms_form"):
    user_input = st.text_area("Enter your message:", height=150)
    submit = st.form_submit_button("Classify")

# Prediction
if submit:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        transformed_text = transform_text(user_input)
        vector_input = vectorizer.transform([transformed_text])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **HAM** (Not Spam).")
