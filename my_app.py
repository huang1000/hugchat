import streamlit as st
import pandas as pd, numpy as np
import PyPDF2, openai
from hugchat import hugchat
from langchain import document_loaders, embeddings, vectorstores, chains, chat_models, llms, PromptTemplate, HuggingFaceHub
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from utils import *


app_name = 'Text App with LLM'
st.set_page_config(layout='wide', page_title=app_name)
st.title(app_name)

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'model' not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"
if 'api' not in st.session_state:
    st.session_state.api = "HugFace"
if 'task' not in st.session_state:
    st.session_state.task = "Summary"
if 'initialised' not in st.session_state:
    st.session_state.initialised = False

# configure application
with st.sidebar:
    st.title("LLM Config")
    st.session_state.task = st.selectbox('NLP Task', ['Chatbot', 'Summary', 'QA', 'Agent', 'About'])
    st.session_state.api = st.selectbox('API', ['HugFace', 'LangChain', 'OpenAI'])
    st.session_state.model = st.selectbox('LLM Model', ['upstage/Llama-2-70b-instruct-v2', 'stabilityai/StableBeluga-7B', 'google/flan-t5-xl', 'OpenAssistant/oasst-sft-6-llama-30b-xor', 'meta-llama/Llama-2-7b-chat-hf', 'openlm-research/open_llama_3b_v2', 'MBZUAI/LaMini-Neo-1.3B', 'gpt2-medium', 'TurkuNLP/gpt3-finnish-small', 'gpt-3.5-turbo'])
    st.session_state.temperature = st.slider('Temperature :fire:', 0.0, 1.0, 0.8)

    st.subheader("HuggingFace Login")
    hf_email = st.text_input('Enter E-mail:', type='password')
    hf_pass = st.text_input('Enter password:', type='password')


def clear_text():
    st.session_state.input_text = ""

def reset_chatbot():
    # clear chatbot and messages
    if 'messages' in st.session_state.keys():
        st.session_state.pop('messages')
    if 'chatbot' in st.session_state.keys():
        st.session_state.pop('chatbot')

def text_summary():
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
            st.session_state.task = "Summary"
            try:
                output = get_summary(st.session_state.api, st.session_state.model, st.session_state.input_text)
                st.info(output)
                st.session_state.output_text = output
            except Exception as e:
                st.error(e)
        else:
            st.info(st.session_state.output_text)
            
def display_msg():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def chatbot():
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi, how may I help you?"}]

    # log in to Huggingface and grant authorization to hugchat
    if st.session_state.api in ("HugFace", "LangChain"):
        if "cookies" not in st.session_state.keys():
            hugface_login(hf_email, hf_pass)
    elif st.session_state.api == 'OpenAI':
        openai.api_key = openai_key
        
    # create chatbot model
    if st.session_state.api == "HugFace":
        if "chatbot" not in st.session_state.keys():
            hugface_create_chatbot()
    elif st.session_state.api == 'LangChain':
        if "chatbot" not in st.session_state.keys():
            llm = HuggingFaceHub(repo_id=st.session_state.model, model_kwargs={'temperature': st.session_state.temperature})
            st.session_state.chatbot = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=8))
    elif st.session_state.api == "OpenAI":
        if "chatbot" not in st.session_state.keys():
            st.session_state.chatbot = llms.OpenAI(model_name='gpt-3.5-turbo', temperature=st.session_state.temperature)
            #st.session_state.chatbot = chat_models.ChatOpenAI(model='gpt-3.5-turbo', temperature=st.session_state.temperature)
    st.session_state.initialised = True
    
    # Display chat messages
    display_msg()

    # user prompt
    if prompt := st.chat_input(disabled=not st.session_state.initialised):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... :thinking_face:"):
                if st.session_state.api == 'HugFace':
                    st.session_state.chatbot.active_model = st.session_state.model
                    response = st.session_state.chatbot.chat(prompt, temperature=st.session_state.temperature)
                elif st.session_state.api == 'OpenAI':
                    completion = openai.ChatCompletion.create(model=st.session_state.model, \
                                                            messages=st.session_state.messages)
                    response = completion.get("choices")[0].get('message').get('content')
                elif st.session_state.api == 'LangChain':
                    response = st.session_state.chatbot.run(prompt)

                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# tab1, tab2, tab3 = st.tabs(["Chatbot", "Summarize", "Others"])

if st.session_state.task == "Chatbot":
    st.header("LLM-powered Chatbot :robot_face:")
    st.button("Restart :robot_face:", on_click=reset_chatbot())
    chatbot()
elif st.session_state.task == "Summary":
    st.header("LLM-powered text summary")
    text_summary()
elif st.session_state.task == "QA":
    st.header("LLM-powered QA over specific documents")
    st.write("To be available")
elif st.session_state.task == "Agent":
    st.header("LLM-powered Agent")
    st.write("To be available soon")
else:
    st.header("About TextApp powered by LLM model")
    st.markdown(open("README.md", 'r').read())

