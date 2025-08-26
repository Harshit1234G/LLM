import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from Agent.chatbot import ChatBot


class Input(BaseModel):
    ...
