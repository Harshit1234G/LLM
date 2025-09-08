from typing import Any
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import DEFAULT_MODEL, SMALL_MODEL


class ResearchState(TypedDict):
    topic: str
    source: str
    wikipedia_docs: str
    arxiv_docs: str
    knowledge: dict[str, Any]
    report_parts: list[str]
    final_report: str

class BaseAgent:
    def __init__(
            self, 
            name: str, 
            instructions: ChatPromptTemplate, 
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
            temperature= self.temperature,
            **llm_kwargs
        )


    def run(self, state: ResearchState) -> ResearchState:
        raise NotImplementedError('Subclasses must implement run()')
    

    def stream(self, state: ResearchState) -> ResearchState:
        return self.run(state)
