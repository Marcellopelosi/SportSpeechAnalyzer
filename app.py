import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain


def text_correction(question):

  os.environ["HUGGINGFACEHUB_API_TOKEN"] == st.secrets["huggingface_api_key"]
  
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

# Function to process text using Model 2 and return a DataFrame
def process_text_model_2(input_text):
    # Replace this with your model's processing logic that returns a pandas DataFrame
    # Here, we're creating a simple DataFrame for demonstration purposes
    data = {
        'Word': input_text.split(),
        'Length': [len(word) for word in input_text.split()]
    }
    df = pd.DataFrame(data)
    return df

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
        st.header("Processed DataFrame from Model 2")
        processed_dataframe = process_text_model_2(processed_text_1)
        st.write(processed_dataframe)

if __name__ == "__main__":
    main()
