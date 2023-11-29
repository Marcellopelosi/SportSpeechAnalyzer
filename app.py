import streamlit as st
import pandas as pd
import os
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
stemmer = PorterStemmer()

keywords = pd.read_csv("keywords elaborated.csv")

def keywords_to_group(keywords_list):
  stemmed_term_group_dict = dict(zip(keywords["stemmed_term"], keywords["group"]))
  groups = [stemmed_term_group_dict[stemmer.stem(k)] for k in keywords_list]
  return list(set(groups))

def text_to_sentences(text):
    text = text.replace("!", ".")
    strings = [(sentence + ".").strip() for sentence in text.split(".")]
    merged_sentences = []
    current_sentence = ''

    for string in strings:
        # Combine strings until the word count reaches 20-40
        if len(current_sentence.split()) + len(string.split()) <= 40:
            current_sentence += ' ' + string.strip()
        else:
            # If the word count exceeds 40, start a new sentence
            if len(current_sentence.split()) >= 20:
                merged_sentences.append(current_sentence.strip())
            current_sentence = string.strip()

    # Append the last sentence
    merged_sentences.append(current_sentence.strip())

    return merged_sentences
  
def keywords_extractor(text):
  
  keywords_in_sentence = []
  sentences = text_to_sentences(text)
  for sentence in sentences:
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(re.sub(r'[^\w\s]', '', word)) for word in words]
    keywords_found = [words[i] for i,sw in enumerate(stemmed_words) if sw in keywords.stemmed_term.to_list()]
    keywords_in_sentence.append(keywords_found)

  results = pd.DataFrame()
  results["highlights"] = sentences
  results["keywords"] = keywords_in_sentence
  results = results[results["keywords"].apply(len)>0]
  results["group"] = results["keywords"].apply(lambda keywords_list: keywords_to_group(keywords_list))
  return results

import requests

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


API_TOKEN = st.secrets["huggingface_api_key"]
API_URL = "https://api-inference.huggingface.co/models/oliverguhr/fullstop-punctuation-multilang-large"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def punt_corrector(question):
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
    answer.append(punt_corrector(subquestion))

  complete_answer = " ".join(answer)
  return complete_answer

def main():
    st.title("Keywords Extraction Interface")

    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    
    if uploaded_file is not None:
        # Read the uploaded file
        text = uploaded_file.getvalue().decode("utf-8")

        # Display original text
        st.header("Original Text")
        st.text(text)

        # Process text using puntuaction correction model
        text_corrected = complete_answer_calculator(text)
        st.header("Punctuation correction")
        st.text(text_corrected)

        # Process text using Model 2 and display as DataFrame
        st.header("Keywords extraction")
        processed_dataframe = keywords_extractor(text_corrected)
        st.write(processed_dataframe)

if __name__ == "__main__":
    main()
