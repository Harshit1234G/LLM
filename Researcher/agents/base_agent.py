from typing import Any
from typing_extensions import TypedDict

from langchain_core.prompts import BasePromptTemplate
from langchain_openai import ChatOpenAI

from config import DEFAULT_MODEL, SMALL_MODEL


class ResearchState(TypedDict):
    topic: str
    wikipedia_docs: list[dict[str, Any]]
    arxiv_docs: list[dict[str, Any]]
    news: str


class BaseAgent:
    def __init__(
            self, 
            name: str, 
            instructions: BasePromptTemplate, 
            temperature: float,
            *, 
            use_small_model: bool = False,
            **llm_kwargs
        ) -> None:
        # basic attributes
        self.name = name
        self.instructions = instructions
        self.temperature = temperature
        self.model = SMALL_MODEL if use_small_model else DEFAULT_MODEL

        # core components
        self.llm = ChatOpenAI(
            model= self.model,
            temperature= self.temperature
            **llm_kwargs
        )


    async def run(self, state: ResearchState) -> dict[str, Any]:
        raise NotImplementedError('Subclasses must implement run()')
    

    async def stream(self, state: ResearchState) -> dict[str, Any]:
        return await self.run(state)
