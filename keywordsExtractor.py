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
from text_to_sentences import text_to_sentences

keywords = pd.read_csv("keywords elaborated.csv")

def keywords_to_group(keywords_list):
  stemmed_term_group_dict = dict(zip(keywords["stemmed_term"], keywords["group"]))
  groups = [stemmed_term_group_dict[stemmer.stem(k)] for k in keywords_list]
  return list(set(groups))
  
def extract_keywords(text):
  
  keywords_in_sentence = []
  sentences = text_to_sentences(text)
  for sentence in sentences:
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(re.sub(r'[^\w\s]', '', word)) for word in words]
    keywords_found = [words[i] for i,sw in enumerate(stemmed_words) if sw in keywords.stemmed_term.to_list()]
    keywords_found = list(set(keywords_found))
    keywords_in_sentence.append(keywords_found)

  results = pd.DataFrame()
  results["highlights"] = sentences
  results["keywords"] = keywords_in_sentence
  results = results[results["keywords"].apply(len)>0]
  results["group"] = results["keywords"].apply(lambda keywords_list: keywords_to_group(keywords_list))
  return results
