import streamlit as st
import numpy as np
import pandas as pd
import json
import nltk
import random
import string
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Load JSON Data
with open("content.json") as content:
    data1 = json.load(content)

# Prepare responses dictionary
responses = {intent["tag"]: intent["responses"] for intent in data1["intents"]}

# Prepare dataset
tags = []
inputs = []
for intent in data1["intents"]:
    for lines in intent["input"]:
        inputs.append(lines)
        tags.append(intent["tag"])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Clean text (lowercase & remove punctuation)
data["inputs"] = data["inputs"].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data["inputs"] = data["inputs"].apply(lambda wrd: "".join(wrd))

# Tokenization with OOV handling
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["inputs"])
train = tokenizer.texts_to_sequences(data["inputs"])

# Padding sequences
x_train = pad_sequences(train, padding="post")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(data["tags"])

# Train-test split (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Model parameters
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

# Build LSTM Model
i = Input(shape=(input_shape,))
z = Embedding(vocabulary + 1, 10)(i)
z = LSTM(10, return_sequences=True)(z)
z = GlobalAveragePooling1D()(z)
z = Dense(output_length, activation="softmax")(z)
model = Model(i, z)

# Compile Model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model with EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stop], verbose=0)

# Save trained model (optional)
model.save("sevak_chatbot.h5")

# Load trained model instead of retraining every time
model = tf.keras.models.load_model("sevak_chatbot.h5")

# Streamlit Chatbot App
st.title("Sevak Chatbot 🤖")
st.write("### Welcome to Sevak! Ask me anything about job postings and workers.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if user_input:
    # Preprocess input text
    text_p = ["".join([letters.lower() for letters in user_input if letters not in string.punctuation])]
    prediction_input = tokenizer.texts_to_sequences(text_p)

    if not prediction_input[0]:  # Handle empty sequence (OOV)
        response = "Sorry, I didn't understand that. Can you try rephrasing?"
    else:
        prediction_input = pad_sequences(prediction_input, maxlen=input_shape, padding="post")
        output = model.predict(prediction_input)
        response_tag = le.inverse_transform([output.argmax()])[0]
        response = random.choice(responses[response_tag])
    
    # Append user input and bot response to history
    st.session_state.chat_history.append((user_input, response))

# Display chat history
st.write("### Chat History")
for user_text, bot_response in st.session_state.chat_history:
    st.markdown(f"**You:** {user_text}")
    st.markdown(f"**Sevak Bot:** {bot_response}")
