import streamlit as st
import os
import openai 
from Agent.chatbot import ChatBot


@st.cache_resource
def load_agent(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return ChatBot(vector_db_path= './Databases/faiss_index')


st.set_page_config(page_title= 'NeuroHarshit', layout= 'wide')


with st.sidebar:
    st.markdown('**✨ NueroHarshit**')
    st.markdown('---')  

    # API Key input
    openai_api_key = st.text_input(
        'Enter your OpenAI API key here',
        key='api_key',
        type='password'
    )

    st.markdown('---')

    '[Get an OpenAI API key](https://platform.openai.com/account/api-keys)' 
    '[View the source code](https://github.com/Harshit1234G/LLM/tree/main/NeuroHarshit)'



st.title('✨ NueroHarshit')


if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if question := st.chat_input():
    if not openai_api_key:
        st.info('Please add your OpenAI API key in the sidebar to continue.')
        st.stop()

    try: 
        chatbot = load_agent(openai_api_key)

        st.session_state.messages.append({'role': 'user', 'content': question})
        st.chat_message('user').write(question)

        response = chatbot.run(question)

        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.chat_message('assistant').write(response)

    except openai.APIConnectionError:
        st.error('Issue connecting to Open AI services.\nSolution: Check your network settings, proxy configuration, SSL certificates, or firewall rules.')

    except openai.AuthenticationError:
        st.error('Your API key or token was invalid, expired, or revoked.\nSolution: Check your API key or token and make sure it is correct and active. You may need to generate a new one from your account dashboard.')

    except Exception as e:
        st.error(f'Something went wrong...\nThe following error caused this issue: {e}')
