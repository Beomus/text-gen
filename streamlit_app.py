import gdown
import json
import os
import streamlit as st
import torch
from collections import Counter
import zipfile

import lstm_eval
import gpt_eval


@st.cache
def get_checkpoints():
    url = 'https://drive.google.com/uc?id=1-JG4h6ieH98A75pbhpNULtZFJf2x9pcm'
    output = 'checkpoints.zip'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=True)

    if not os.path.exists("./checkpoints"):
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall("./")

if not os.path.exists('checkpoints'):
    get_checkpoints()


st.title('Text Generator')

model_choice = st.selectbox('Select model', ['LSTM', 'GPT'])
data_choice = st.selectbox('Select theme', ['LoTR', 'Harry Potter'])
checkpoint = st.selectbox('Select checkpoint', ['min_loss', 'max_loss'])

data = 'lotr' if data_choice == "LoTR" else 'hp'

st.image(f'panel_{model_choice.lower()}_{data}.png')

model_path = f"checkpoints/{model_choice.lower()}_{data}_{checkpoint}"

if model_choice == 'LSTM':
    with open(f'{model_path}/misc/word_to_id.json') as json_file:
        word_to_id = Counter(json.load(json_file))

    id_to_word = ["<Unknown>"] + [word for word, index in word_to_id.items()]

    net = torch.load(f'{model_path}/last_checkpoint.pth', map_location='cpu')
    net.eval()

    num_sentences = st.number_input('Number of Sentences', min_value=1, max_value=20, value=5)
    user_input = st.text_input('Seed Text (can leave blank)')

    if st.button('Generate Text'):
        generated_text = lstm_eval.prediction(net, word_to_id, id_to_word, user_input, 9, num_sentences)
        st.write(generated_text)
if model_choice == "GPT":
    num_sentences = st.number_input('Number of Sentences', min_value=1, max_value=20, value=5)
    min_length = st.slider('Sentence min length', min_value=1, max_value=50, value=1)
    max_length = st.slider('Sentence max length', min_value=50, max_value=300, value=50)
    user_input = st.text_input('Seed Text (can leave blank)', "<|startoftext|>")

    user_input = "<|startoftext|>" if len(user_input) == 0 else user_input

    if st.button('Generate Text'):
        generated_text = gpt_eval.prediction(
            model_path, 
            user_input,
            num_sentences,
            min_length=min_length,
            max_length=max_length,
        )
        st.write(generated_text)
