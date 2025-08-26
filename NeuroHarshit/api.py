import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import getpass
import os

from Agent.chatbot import ChatBot


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = 'default'


class ChatResponse(BaseModel):
    question: str
    answer: str

try: 
    load_dotenv()
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')

    chatbot = ChatBot(vector_db_path= r'.\Databases\faiss_index')

except Exception as e:
    raise HTTPException(status_code= 500, detail= f'Cannot load chatbot: {e}')


app = FastAPI()


@app.get('/', response_model= dict[str, str])
def root():
    return {'Hello': 'world'}


@app.post('/chat', response_model= ChatResponse)
def generate(data: ChatRequest):
    try:
        answer = chatbot.run(data.question, thread_id= data.thread_id)
        return {'question': data.question, 'answer': answer}
    
    except Exception as e:
        raise HTTPException(status_code= 400, detail= f'Generation failed: {e}')


uvicorn.run(app)
