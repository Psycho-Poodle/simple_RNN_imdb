#Step 1: Import Libraries and import the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the imdb dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value : key for key, value in word_index.items()}

# print(reversed_word_index)


# Load the pre-trained model with Relu Activation function
model = load_model('Simple_RNN_Imdb/simple_rnn_model.h5')


# Step 2: Helper Functions
def decoded_reviews(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?')for i in encoded_review])

# Function to process user input
def preprocess_text(text):
    
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words ]
    padded_review = sequence.pad_sequence ([encoded_review], maxlen = 500)
    return padded_review

### Step3: Prediction Function
def predict_sentiment(review):
    processed_input = preprocess_text(review)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

### Sreamlit app
import streamlit as st
st.title("IMDB Movie Review sentiment Analysis")
st.write("Enter a Movie review to classify it as Postive Or Negative.")

# User Input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # Make Prediction
    prediction = predict_sentiment(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'


    # Display Result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write(f'Please Enter a Movie Review!')







