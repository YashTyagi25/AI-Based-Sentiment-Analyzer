#############################################
# Sentiment Analysis Project
# Author: Yash Tyagi
# Description: This project uses a pre-trained LSTM model from 
# TensorFlow Hub to perform sentiment analysis on text data. 
# It classifies text as positive, negative, or neutral.
#############################################

# Import required libraries
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load Pre-trained LSTM Sentiment Model
# Using a pretrained model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
model = hub.load(MODEL_URL)

# Step 2: Define a function to preprocess text data
def preprocess_text(text):
    """
    Preprocess text by removing stopwords and tokenizing.

    :param text: Input text
    :return: Cleaned text as a list of words
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)  # Return as string for the model

# Step 3: Define function for sentiment analysis
def analyze_sentiment(texts):
    """
    Predict sentiments of input text(s) using the pre-trained LSTM model.

    :param texts: List of input text strings
    :return: Sentiment predictions (positive, negative, neutral)
    """
    processed_texts = [preprocess_text(text) for text in texts]  # Preprocess text
    embeddings = model(processed_texts).numpy()  # Convert text to embeddings

    # Simple sentiment classification based on cosine similarity (example logic)
    positive_threshold = 0.5  # Example threshold for positive sentiment
    negative_threshold = -0.5  # Example threshold for negative sentiment

    sentiments = []
    for embed in embeddings:
        score = np.mean(embed)  # Taking the mean of embedding values
        if score > positive_threshold:
            sentiments.append("positive")
        elif score < negative_threshold:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    
    return sentiments

# Step 4: Test with sample data
sample_texts = [
    "I absolutely love this product! It's fantastic.",
    "The movie was terrible and I hated it.",
    "I'm not sure how I feel about this experience.",
    "The service was okay, but the food was excellent.",
    "Terrible customer support ruined my day!"
]

# Predict sentiments
results = analyze_sentiment(sample_texts)

# Display results
print("\nSample Sentiment Analysis Results:")
for text, sentiment in zip(sample_texts, results):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 40)

# Step 5: Load Dataset and Evaluate the Model
# Load dataset (CSV file with 'text' and 'label' columns)
df = pd.read_csv("sentiment_dataset.csv")  # Ensure dataset file is available

# Preprocess text and split dataset
df["processed_text"] = df["text"].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(df["processed_text"], df["label"], test_size=0.2, random_state=42)

# Predict sentiments for the test dataset
predicted_labels = analyze_sentiment(X_test.tolist())

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels, digits=4))
