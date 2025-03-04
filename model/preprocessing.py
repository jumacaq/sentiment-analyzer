# model/preprocessing.py
import re
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import emoji

def count_emojis(text):
    """
    Count number of emojis in the text
    """
    return len([char for char in text if char in emoji.EMOJI_DATA])

def count_uppercase_words(text):
    """
    Calculate proportion of uppercase words
    """
    words = text.split()
    if not words:
        return 0
    uppercase_words = sum(1 for word in words if word.isupper())
    return uppercase_words / len(words)

def count_punctuation(text, punct_type):
    """
    Count specific punctuation marks
    """
    if punct_type == 'exclamation':
        return text.count('!')
    elif punct_type == 'question':
        return text.count('?')
    return 0

def count_stopwords(text):
    """
    Count stopwords and calculate ratio
    """
    # Common English stopwords
    stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    
    words = text.lower().split()
    stopwords_count = sum(1 for word in words if word in stopwords)
    
    return {
        'stopwords_count': stopwords_count,
        'stopwords_ratio': stopwords_count / len(words) if words else 0
    }

def detect_sarcasm(text):
    """
    Simple sarcasm detection (very basic)
    """
    # Naive approach: look for contrasting elements
    # More sophisticated methods would require ML models
    contains_positive_words = any(word in text.lower() for word in ['great', 'awesome', 'amazing'])
    contains_negative_context = any(word in text.lower() for word in ['not', 'never', 'hardly'])
    
    return 1 if (contains_positive_words and contains_negative_context) else 0

def extract_features(tweet):
    """
    Extract comprehensive features for sentiment analysis
    
    Args:
        tweet (str): Input tweet text
    
    Returns:
        list: Extracted numerical features
    """
    # Get current datetime
    now = pd.Timestamp.now()
    
    # Text preprocessing
    text = str(tweet).lower().strip()
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(text)
    
    # Feature extraction
    features = {
        'day': now.day,  # Current day of month
        'hour': now.hour,  # Current hour
        'tweet_length': len(text),
        'emoji_count': count_emojis(tweet),
        'exclamation_count': count_punctuation(text, 'exclamation'),
        'question_count': count_punctuation(text, 'question'),
        'uppercase_proportion': count_uppercase_words(tweet),
        
        # Stopwords features
        **count_stopwords(text),
        
        # Repeated words
        'repeated_words_count': len(set(text.split())) - len(set(text.split())),
        'repeated_words_ratio': (len(set(text.split())) - len(set(text.split()))) / len(text.split()) if text.split() else 0,
        
        # Sentiment and subjectivity
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment_score': blob.sentiment.polarity,
        
        # Sarcasm detection
        'sarcasm_score': detect_sarcasm(tweet)
    }
    
    # Convert to list in the same order as training
    feature_order = [
        'day', 'hour', 'tweet_length', 'emoji_count', 'exclamation_count', 
        'question_count', 'uppercase_proportion', 'stopwords_count', 
        'stopwords_ratio', 'repeated_words_count', 'repeated_words_ratio', 
        'subjectivity', 'sentiment_score', 'sarcasm_score'
    ]
    
    return [features[feat] for feat in feature_order]
'''
def load_model(model_path='model/lgbm_model.joblib'):
    """
    Load the pre-trained LightGBM model
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        Trained LightGBM model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
'''
import os
import joblib

def load_model(model_path='model/lgbm_model.joblib'):
    """
    Load the pre-trained LightGBM model with enhanced error checking
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        Trained LightGBM model or None
    """
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file does not exist at path: {model_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Full absolute path: {os.path.abspath(model_path)}")
        return None
    
    try:
        # Attempt to load the model with verbose error handling
        model = joblib.load(model_path)
        
        # Additional validation
        if model is None:
            print(f"ERROR: Model loaded as None from {model_path}")
            return None
        
        # Optional: Add a basic validation if the model has a predict method
        try:
            hasattr(model, 'predict')
        except Exception as attr_error:
            print(f"ERROR: Model does not have a predict method: {attr_error}")
            return None
        
        return model
    
    except FileNotFoundError:
        print(f"File Not Found: {model_path}")
        return None
    except PermissionError:
        print(f"Permission denied when trying to access: {model_path}")
        return None
    except ImportError as ie:
        print(f"Import error - check if required libraries are installed: {ie}")
        return None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def predict_sentiment(tweet, model):
    """
    Predict sentiment of a tweet
    
    Args:
        tweet (str): Input tweet text
        model: Trained LightGBM model
    
    Returns:
        str: Predicted sentiment
    """
    # Extract features
    features = extract_features(tweet)
    
    # Predict
    prediction = model.predict([features])[0]
    
    # Map prediction to sentiment
    if prediction == 0:
        return "Negative 😞"
    elif prediction == 1:
        return "Positive 😊"
    else:
        return "Neutral 😐"