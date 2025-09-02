from langchain_core.prompts import ChatPromptTemplate

from tools import wiki_tool, arxiv_tool
from agents import BaseAgent


class SearcherAgent(BaseAgent):
    def __init__(self):
        prompt = ChatPromptTemplate(
            messages= [
                (
                    "system", 
                    "ROLE: Router for data retrieval.\n"
                    "TASK: Decide which source to use for efficient data retrieval for the user TOPIC from the following sources:\n"
                    "- Wikipedia\n- Arxiv\n- Both\n"
                    "CONSTRAINTS:\n"
                    "- Decide from the above sources only.\n"
                    "- Decision is based on which source can best describe the TOPIC.\n"
                    "- Output exactly one source.\n"
                    "OUTPUT: A single word 'wiki', 'arxiv', or 'both'. No extra fluff/explanation."
                ),
                ('human', 'TOPIC: {topic}')
            ]
        )

        super().__init__(
            name= 'searcher',
            instructions= prompt, 
            temperature= 0.0,
            use_small_model= True
        )


    def __decide_source(self, topic: str) -> str:
        source = self.llm.invoke(
            self.instructions.format_messages(topic= topic)
        ).content.strip().lower()
        return source


    def run(self, state):
        topic = state.get('topic', None)
        if topic is None:
            raise ValueError('No value for "topic" was provided.')

        source = self.__decide_source(topic)
        state['source'] = source

        match source:
            case 'wiki':
                state['wikipedia_docs'] = wiki_tool(topic)
                state['arxiv_docs'] = None

            case 'arxiv':
                state['arxiv_docs'] = arxiv_tool(topic)
                state['wikipedia_docs'] = None

            case 'both':
                state['wikipedia_docs'] = wiki_tool(topic)
                state['arxiv_docs'] = arxiv_tool(topic)

        return state
                