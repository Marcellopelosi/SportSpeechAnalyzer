import streamlit as st
import os
import requests

API_TOKEN = st.secrets["huggingface_api_key"]
API_URL = "https://api-inference.huggingface.co/models/oliverguhr/fullstop-punctuation-multilang-large"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def split_text_into_blocks(text, block_size=200):
    words = text.split()
    num_words = len(words)
    start = 0
    blocks = []

    while start < num_words:
        end = min(start + block_size, num_words)
        block = ' '.join(words[start:end])
        blocks.append(block)
        start = end

    return blocks

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def puntuaction_corrector(question):
  data = query(question)
  corrected_text = []
  for dictionary in data:
    if dictionary["entity_group"]!= "0":
      p = dictionary["entity_group"]
    else:
      p = ''
    corrected_text.append(dictionary["word"] + p)
  answer = " ".join(corrected_text)
  return answer

def complete_answer_calculator(question):
  answer = []
  for subquestion in split_text_into_blocks(question):
    answer.append(puntuaction_corrector(subquestion))

  complete_answer = " ".join(answer)
  return complete_answer
