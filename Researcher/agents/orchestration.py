from time import perf_counter
from langgraph.graph import StateGraph

from agents import (
    ResearchState, 
    SearcherAgent, 
    ExtractorAgent, 
    WriterAgent, 
    CriticAgent, 
    AssemblerAgent
)
from utils import get_logger, save_state, sanitize_filename


class ResearchAssistant:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        searcher = SearcherAgent()
        extractor = ExtractorAgent()
        writer = WriterAgent()
        critic = CriticAgent()
        assembler = AssemblerAgent()

        builder = StateGraph(ResearchState)

        builder.add_node(searcher.name, searcher.run)
        builder.add_node(extractor.name, extractor.run)
        builder.add_node(writer.name, writer.run)
        builder.add_node(critic.name, critic.run)
        builder.add_node('assembler', assembler.create_final_pdf)

        builder.set_entry_point(searcher.name)
        builder.add_edge(searcher.name, extractor.name)
        builder.add_edge(extractor.name, writer.name)
        builder.add_conditional_edges(
            writer.name,
            self._criticize_con,
            {
                True: 'assembler',
                False: critic.name
            }
        )
        builder.add_conditional_edges(
            critic.name,
            self._pass_fail,
            {
                'pass': 'assembler',
                'fail': writer.name
            }
        )
        builder.set_finish_point('assembler')

        self.graph = builder.compile()
        self.logger.info('Graph compilation successfull, ResearchAssistant Initialized.')


    def _criticize_con(self, state: ResearchState) -> bool:
        is_criticized = state.get('is_criticized', False)

        if is_criticized:
            self.logger.info('Based on the criticism, rewriting was successfull. Next Node: AssemblerAgent.')

        else:
            self.logger.info('Expanding was successfull. Next Node: CriticAgent')

        return is_criticized


    def _pass_fail(self, state: ResearchState) -> str:
        for criticism in state.get('criticism').values():
            if criticism != 'PASS':
                self.logger.info('One or more report parts failed critic. Next Node: WriterAgent.')
                return 'fail'
        
        self.logger.info('All the report parts passed the critic, skipping rewriting. Next Node: AssemblerAgent.')
        return 'pass'
    

    def run(self, user_input: str) -> ResearchState:
        self.logger.info(f'Starting research on topic "{user_input}"...')
        try:
            start = perf_counter()
            state = {'topic': user_input}
            state = self.graph.invoke(state)
            end = perf_counter()

            elapsed = end - start
            minutes, seconds = divmod(elapsed, 60)
            save_state(state, topic= user_input)

            self.logger.info(f'Total time taken: {int(minutes)}m {seconds:.2f}s')
            self.logger.info(f'Saved final state of the program at .data/{sanitize_filename(user_input)}.json')

            return state
        
        except Exception as e:
            self.logger.exception(f'Error while researching topic {user_input}: {e}')
            save_state(self.graph.get_state(), topic= user_input)
            self.logger.info(f'Saved current state of the program at .data/{sanitize_filename(user_input)}.json')
            exit()
