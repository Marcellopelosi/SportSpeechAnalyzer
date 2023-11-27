import streamlit as st
import pandas as pd

# Function to process text using Model 1
def process_text_model_1(input_text):
    # Replace this with your model's processing logic
    processed_text = input_text.upper()  # Example: Convert text to uppercase
    return processed_text

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

        # Process text using Model 1
        processed_text_1 = process_text_model_1(text)
        st.header("Processed Text from Model 1")
        st.text(processed_text_1)

        # Process text using Model 2 and display as DataFrame
        st.header("Processed DataFrame from Model 2")
        processed_dataframe = process_text_model_2(processed_text_1)
        st.write(processed_dataframe)

if __name__ == "__main__":
    main()
