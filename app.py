import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
import random
import string

# Load JSON data
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

data["inputs"] = data["inputs"].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data["inputs"] = data["inputs"].apply(lambda wrd: "".join(wrd))

# Tokenization
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data["inputs"])
train = tokenizer.texts_to_sequences(data["inputs"])

# Padding
x_train = pad_sequences(train)

# Encoding labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data["tags"])

input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Build LSTM Model
i = Input(shape=(input_shape,))
z = Embedding(vocabulary + 1, 10)(i)
z = LSTM(10, return_sequences=True)(z)
z = Flatten()(z)
z = Dense(output_length, activation="softmax")(z)
model = Model(i, z)

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(x_train, y_train, epochs=200, verbose=0)

# Streamlit App
st.title("Sevak Chatbot")
st.write("### Welcome to Sevak! Ask me anything about job postings and workers.")

user_input = st.text_input("You:")
if user_input:
    text_p = ["".join([letters.lower() for letters in user_input if letters not in string.punctuation])]
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)

    output = model.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])

    st.markdown(f"### *Sevak Bot:*  ")
    st.markdown(f"<p style='font-size:18px;'>{response}</p>", unsafe_allow_html=True)
