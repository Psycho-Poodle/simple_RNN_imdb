#Step 1: Import Libraries and import the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the imdb dataset word index
# word_index = imdb.get_word_index()
# reversed_word_index = {value : key for key, value in word_index.items()}

# print(reversed_word_index)


# Load the pre-trained model with Relu Activation function
model = load_model('Simple_RNN_Imdb/simple_rnn_model.h5')


# Step 2: Helper Functions
def decoded_reviews(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?')for i in encoded_review])


