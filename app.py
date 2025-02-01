import random
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
import string 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten, Conv1D
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelEncoder

# Load the intents file
with open("content.json") as content:
    data1 = json.load(content)

# Prepare the data
inputs = []
tags = []
responses = {}
for intent in data1["intents"]:
    responses[intent["tag"]] = intent["responses"]
    for lines in intent["input"]:
        inputs.append(lines)
        tags.append(intent["tag"])

# Convert to a dataframe
data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Remove punctuations
data["inputs"] = data["inputs"].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data["inputs"] = data["inputs"].apply(lambda wrd: ''.join(wrd))

# Tokenize the data
tokenizer = Tokenizer(num_words=1000)  # Reduce vocabulary size for speed
tokenizer.fit_on_texts(data["inputs"])
train = tokenizer.texts_to_sequences(data["inputs"])

# Apply padding
x_train = pad_sequences(train)

# Encode the outputs
le = LabelEncoder()
y_train = le.fit_transform(data["tags"])

# Define model parameters
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Build the model (Using Conv1D for faster inference)
i = Input(shape=(input_shape,))
z = Embedding(vocabulary + 1, 10)(i)
z = Conv1D(32, 3, activation='relu')(z)  # Replacing LSTM with Conv1D
z = GlobalMaxPooling1D()(z)  # Pooling layer
z = Dense(output_length, activation="softmax")(z)
model = Model(i, z)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Cache the response generation function to speed up repeated predictions
@st.cache(ttl=60*60)  # Cache for 1 hour
def get_response(user_input):
    # Preprocess input
    user_input = ''.join([ch for ch in user_input.lower() if ch not in string.punctuation])
    prediction_input = tokenizer.texts_to_sequences([user_input])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)
    
    # Predict and get response
    output = model.predict(prediction_input)
    output_tag = le.inverse_transform([output.argmax()])[0]
    return random.choice(responses[output_tag])

# Streamlit app
st.title("Chatbot with Streamlit")

# Initialize history container (this will persist across interactions)
if 'history' not in st.session_state:
    st.session_state.history = []

# Display instructions
st.write("Start a conversation with the chatbot! Type your message below and hit 'Enter' or click 'Send'.")

# Input from user (this will be dynamically added)
user_input_placeholder = st.empty()

# Display the conversation as a chat
def display_chat():
    if st.session_state.history:
        chat_messages = ""
        for entry in st.session_state.history:
            chat_messages += f"You: {entry['input']}\n\n"
            chat_messages += f"Bot: {entry['response']}\n\n"
        st.text_area("Chat History", chat_messages, height=300, key="chat_history", disabled=True)

# Process user input
user_input = user_input_placeholder.text_input("You:", "", key="input")
if user_input:
    # Store the new input for the next round
    st.session_state.user_input = user_input

# After processing input, display the conversation
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
    response = get_response(user_input)
    
    # Append user input and response to history
    st.session_state.history.append({"input": user_input, "response": response})
    del st.session_state.user_input  # Clear input field after processing

display_chat()
