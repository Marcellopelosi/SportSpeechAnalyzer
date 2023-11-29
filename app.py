import streamlit as st
import PuntuactionCorrector
import KeywordsExtractor

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
        text_corrected = puntuaction_corrector(text)
        st.header("Punctuation correction")
        st.text(text_corrected)

        # Process text using Model 2 and display as DataFrame
        st.header("Keywords extraction")
        processed_dataframe = keywords_extractor(text_corrected)
        st.write(processed_dataframe)

if __name__ == "__main__":
    main()
