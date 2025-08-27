import streamlit as st
import os
import openai
from Agent.chatbot import ChatBot


# --- Streamlit config ---
st.set_page_config(page_title= 'NeuroHarshit', layout= 'wide')


@st.cache_resource
def load_agent(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return ChatBot(vector_db_path= os.path.join(base_dir, 'Databases', 'faiss_index'))


# --- Sidebar ---
with st.sidebar:
    st.markdown('## ✨ NeuroHarshit')
    st.markdown('---')

    # API Key input
    openai_api_key = st.text_input(
        'Enter your OpenAI API key',
        key= 'api_key',
        type= 'password',
        placeholder= 'sk-...'
    )

    if openai_api_key:
        st.toast('✅ API Key set successfully')

    st.markdown('---')
    st.caption('[🔑 Get an OpenAI API key](https://platform.openai.com/account/api-keys)') 
    st.caption('[📂 View the source code](https://github.com/Harshit1234G/LLM/tree/main/NeuroHarshit)')


# --- Title & Description ---
st.title('✨ NeuroHarshit')
st.caption('Your personal AI to learn about Harshit in seconds.')

st.markdown(
    """
    **NeuroHarshit** is an AI-powered chatbot that can answer questions about me — my projects, skills, experiences, and interests.  

    💡 Try asking things like:  
    - *What projects has Harshit worked on?*  
    - *What are his technical skills?*  
    - *What are his career goals?*  
    """
)


# --- Chat History ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


# --- Chat Input ---
if question := st.chat_input('Ask anything about Harshit...'):
    if not openai_api_key:
        st.info('⚠️ Please add your OpenAI API key in the sidebar to continue.')
        st.stop()

    try:
        chatbot = load_agent(openai_api_key)

        # User message
        st.session_state.messages.append({'role': 'user', 'content': question})
        st.chat_message('user').write(question)

        # Assistant response
        response = chatbot.run(question)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.chat_message('assistant').write(response)

    except openai.APIConnectionError: 
        st.error('❌ Issue connecting to OpenAI. Check your network/proxy/SSL settings.')
    
    except openai.AuthenticationError: 
        st.error('❌ Your API key was invalid, expired, or revoked. Check your API key and make sure it is correct and active. You may need to generate a new one from your account dashboard.') 
    
    except Exception as e: 
        st.error(f'❌ Something went wrong...\n\n**Error**: {e}')
