import numpy as np, pandas as pd
import streamlit as st
import openai
from hugchat import hugchat
from hugchat.login import Login
import os

def style_number(v, props=''):
    try:
        return props if v<0 else None
    except:
        pass

@st.cache_data()
def load_rand_data(shape=(100,5), columns=list('ACDE')):
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(*shape), columns=columns)
    return df

try:
    openai_key = st.secrets['openai']['api_token']
    hugface_key = st.secrets['hugface']['api_token']
    os.environ['OPENAI_API_KEY'] = openai_key
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugface_key
    hugface_email = st.secrets['hugface']['email']
    hugface_pwd = st.secrets['hugface']['password']
except Exception as error:
    print(error)

@st.cache_data()
def get_summary(api, model, input_text):
    message = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Summarize this: {input_text}"},
    ]
    if api=='OpenAI':
        openai.api_key = openai_key
        response = openai.ChatCompletion.create(model=model, messages=message)
        return response['choices'][0]['Message']['content']
    elif api=='HugFace':
        if "cookies" not in st.session_state.keys():
            hugface_login()
        if "chatbot" not in st.session_state.keys():
            hugface_create_chatbot()
        st.session_state.chatbot.active_model = st.session_state.model
        response = st.session_state.chatbot.chat(f"Summarize this: {input_text}", temperature=st.session_state.temperature)
        return response
    else:
        return "No implementation yet available"
    
def hugface_login(hf_email="", hf_password=""):
    if hf_email == "":
        sign = Login(hugface_email, hugface_pwd)
    else:
        sign = Login(hf_email, hf_password)
    try:
        cookies = sign.login()
        # sign.saveCookies()
        st.session_state.cookies = cookies.get_dict()
    except Exception as e:
        st.write("login failed")
        st.write(e)
        raise(e)

def hugface_create_chatbot():
    st.session_state.chatbot = hugchat.ChatBot(cookies=st.session_state.cookies)
    st.session_state.chatbot.active_model = st.session_state.model
