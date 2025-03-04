# app.py
import streamlit as st
import joblib
from model.preprocessing import load_model, predict_sentiment

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.stTextInput>div>div>input {
    background-color: white;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🐦 Tweet Sentiment Analyzer")
st.write("Predict the sentiment of a tweet using our pre-trained LightGBM model!")

# Load the model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Input text area
tweet = st.text_area("Enter your tweet here:", 
                     placeholder="Type a tweet to analyze its sentiment...",
                     height=150)

# Prediction button
if st.button("Analyze Sentiment"):
    if tweet.strip():
        # Make prediction
        sentiment = predict_sentiment(tweet, model)
        
        # Display results with styling
        st.markdown(f"""
        ## Prediction Result
        ### Sentiment: **{sentiment}**
        
        #### Tweet Analysis Details:
        - **Original Tweet**: {tweet}
        - **Model Used**: LightGBM Classifier
        """)
        
        # Additional insights
        if "Negative" in sentiment:
            st.warning("This tweet seems to express a negative sentiment.")
        elif "Positive" in sentiment:
            st.success("This tweet appears to have a positive tone.")
        else:
            st.info("The sentiment is neutral or difficult to classify.")
    else:
        st.error("Please enter a tweet to analyze.")

# Model information section
st.sidebar.title("📊 Model Information")
st.sidebar.info("""
### Model Details
- **Type**: LightGBM Classifier
- **Task**: Tweet Sentiment Analysis
- **Classes**: Negative, Positive
- **Accuracy**: ~75.38%
""")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit & LightGBM")