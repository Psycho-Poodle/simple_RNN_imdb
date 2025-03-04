# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDb dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation function
model = load_model('simple_rnn_model.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Prediction Function
def predict_sentiment(review):
    processed_input = preprocess_text(review)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

# User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.write("Please enter a movie review!")
    else:
        # Make Prediction
        sentiment, score = predict_sentiment(user_input)
        
        # Display Result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.4f}')