import streamlit as st
import pickle
from helper import text_preprocessing

# Load the model
with open("artifacts/lr.pkl", "rb") as file:
    model = pickle.load(file)

# Set up the Streamlit app
st.set_page_config(page_title="Amazon Review Sentiment Analysis", page_icon=":star:", layout="wide")
st.title("Amazon Review Sentiment Analysis")
st.write("Enter your review below to predict its sentiment (positive or negative).")

# Input text
st.markdown("### Review Input")
text = st.text_area("Enter your review:", height=150)
processed_text = text_preprocessing(text)

# Predict button
if st.button("Predict"):
    with st.spinner('Analyzing the review...'):
        pred = model.predict(processed_text)
        sentiment = "Positive" if pred[0] == 1 else "Negative"
    st.success(f"Prediction: {sentiment}")

# Display additional information
st.sidebar.header("About")
st.sidebar.info("""
This application uses a machine learning model to predict the sentiment of Amazon reviews.
The model was trained on a dataset of Amazon reviews and can classify reviews as positive or negative.
""")

# Add some custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    # Add an illustration of how the app works
st.markdown("### How It Works")
st.write("""
1. **Enter your review**: Type or paste the text of the Amazon review you want to analyze in the text area above.
2. **Click Predict**: Press the 'Predict' button to analyze the sentiment of the review.
3. **View the result**: The app will display whether the review is positive or negative based on the analysis.
""")