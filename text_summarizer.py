import streamlit as st
import pandas as pd, numpy as np
import PyPDF2, openai

from utils import *

@st.cache_data()
def get_summary(input_text):
    if st.session_state.be=='OpenAI':
        openai.api_key = openai_key
        message = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Summarize this: {input_text}"},
        ]
        response = openai.ChatCompletion.create(model=st.session_state.model, messages=message)
        return response['choices'][0]['Message']['content']
    else:
        return "No implementation yet available"

def clear_text():
    st.session_state.input_text = ""


app_name = 'Text Summarization with ChatGPT'
st.session_state.model = "gpt-3.5-turbo"
st.set_page_config(layout='wide', page_title=app_name)
st.session_state.input_text = ""

st.title(app_name)
with st.sidebar:
    st.title("Select Model")
    st.session_state.be = st.selectbox('LLM model', ['OpenAI', 'HugFace', 'LangChain'])
    

col1, col2 = st.columns(2)

with col1:
    with st.form("Input Text"):
        st.subheader("Input Text")
        st.session_state.input_text = st.text_area(label="Enter text:", \
            value=st.session_state.input_text, max_chars=3000)

        # PDF file uploader or input text
        pdf_file = st.file_uploader("Upload PDF file here.")
        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            st.session_state.input_text = "\n".join([pg.extract_text() for pg in pdf_reader.pages])
        
        submitted = st.form_submit_button("Summarize")

    st.button("Clear", on_click=clear_text)

# summarization output
with col2:
    st.subheader("Output Summary")
    if submitted:
        output = get_summary(st.session_state.input_text)
        st.info(output)
