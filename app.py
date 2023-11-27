import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
stemmer = PorterStemmer()

keywords = pd.read_csv("keywords elaborated.csv")

def keywords_to_dominant_group(keywords_list):
  stemmed_term_group_dict = dict(zip(keywords["stemmed_term"], keywords["group"]))
  groups = [stemmed_term_group_dict[stemmer.stem(k)] for k in keywords_list]
  domninant_group = max(set(groups), key=groups.count)
  return domninant_group

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
  results["group"] = results["keywords"].apply(lambda keywords_list: keywords_to_dominant_group(keywords_list))
  return results


def text_correction(question):

  os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_api_key"]
  template = """You are a severe grammar teacher. You receive as input a text without punctuation and have to output the same text but with correct punctuation. If you are in doubt between a comma and a full stop, prefer the full stop.
  Question: {question}
  
  Answer:"""
  
  prompt = PromptTemplate(template=template, input_variables=["question"])
  repo_id = "google/flan-t5-xxl"
  llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.1}
  )
  llm_chain = LLMChain(prompt=prompt, llm=llm)

  return llm_chain.run(question)

def main():
    st.title("Text Processing Interface")

    # File upload
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    
    if uploaded_file is not None:
        # Read the uploaded file
        text = uploaded_file.getvalue().decode("utf-8")

        # Display original text
        st.header("Original Text")
        st.text(text)

        # Process text using puntuaction correction model
        text_corrected = text_correction(text)
        st.header("Punctuation correction")
        st.text(text_corrected)

        # Process text using Model 2 and display as DataFrame
        st.header("Keywords extraction")
        processed_dataframe = keywords_extractor(text_corrected)
        st.write(processed_dataframe)

if __name__ == "__main__":
    main()
