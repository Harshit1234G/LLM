from langchain_core.prompts import ChatPromptTemplate

from tools import wiki_tool, arxiv_tool, google_news_tool
from agents import BaseAgent
from utils import get_logger


class SearcherAgent(BaseAgent):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

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
        self.logger.info('SearcherAgent initialized.')


    def __decide_source(self, topic: str) -> str:
        source = self.llm.invoke(
            self.instructions.format_messages(topic= topic)
        ).content.strip().lower()
        
        self.logger.info(f'Decided source: {source}')
        return source


    def run(self, state):
        self.logger.info('SearcherAgent started.')

        topic = state.get('topic', None)
        if topic is None:
            self.logger.error('No value for "topic" was provided.')
            raise ValueError('No value for "topic" was provided.')

        try:
            source = self.__decide_source(topic)
            state['source'] = source

            match source:
                case 'wiki':
                    state['wikipedia_docs'] = wiki_tool(topic)
                    state['arxiv_docs'] = ''

                case 'arxiv':
                    state['arxiv_docs'] = arxiv_tool(topic)
                    state['wikipedia_docs'] = ''

                case 'both':
                    state['wikipedia_docs'] = wiki_tool(topic)
                    state['arxiv_docs'] = arxiv_tool(topic)

            state['news'] = google_news_tool(topic)

        except Exception as e:
            self.logger.exception(f'Error while retrieving documents: {e}')

        self.logger.info('Successfully loaded the documents.')

        return state
                