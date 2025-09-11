from langchain_core.prompts import ChatPromptTemplate

from tools import wiki_tool, arxiv_tool, google_news_tool
from agents import BaseAgent
from utils import get_logger


class SearcherAgent(BaseAgent):
    def __init__(self):
        """
        Retrieves relevant Wikipedia articles, arXiv research papers, and recent news using specialized tools.
        """
        self.logger = get_logger(self.__class__.__name__)

        prompt = ChatPromptTemplate(
            # for messages I'm using implicit string concatenation, which is used for every prompt in the program
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
        """Decides which source to use for data retrieval: Wikipedia, arXiv, or Both.

        Args:
            topic (str): Topic of report.

        Returns:
            str: Decided source: Wikipedia -> "wiki", arXiv -> "arxiv", Both -> "both"
        """
        source = self.llm.invoke(
            self.instructions.format_messages(topic= topic)
        ).content.strip().lower()
        
        self.logger.info(f'Decided source: {source}')
        return source


    def run(self, state):
        """Retrieves relevant Wikipedia articles, arXiv research papers, and recent news using specialized tools.

        Args:
            state (ResearchState): Current state of the graph.

        Raises:
            ValueError: If state doesn't contain the value for `topic`.

        Returns:
            ResearchState: Updated state with `source`, `wikipedia_docs`, `arxiv_docs` and `news`
        """
        self.logger.info('SearcherAgent started.')

        topic = state.get('topic', None)
        if topic is None:
            self.logger.error('No value for "topic" was provided.')
            raise ValueError('No value for "topic" was provided.')

        try:
            source = self.__decide_source(topic)
            state['source'] = source

            # retrieving data based on the decided source
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

            # retrieving relevant news
            state['news'] = google_news_tool(topic)
            self.logger.info(f'Retrieved recent news on topic: {topic}' if state['news'] != [] else f'No recent news on topic: {topic}')

        except Exception as e:
            self.logger.exception(f'Error while retrieving documents: {e}')

        self.logger.info('Successfully loaded the documents.')
        return state
                