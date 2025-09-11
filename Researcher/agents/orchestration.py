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
        """
        Orchestrates the end-to-end research pipeline using multiple agents.

        This class builds and executes a LangGraph workflow consisting of the following agents:
            - SearcherAgent: Finds and collects relevant documents (Wikipedia, Arxiv, news).
            - ExtractorAgent: Extracts and structures knowledge from the collected documents.
            - WriterAgent: Expands extracted knowledge into report sections.
            - CriticAgent: Reviews report sections and provides feedback for rewriting if needed.
            - AssemblerAgent: Assembles all validated parts into a final PDF report.

        Initializes the ResearchAssistant by:
            - Creating agent instances.
            - Defining a StateGraph workflow with nodes and conditional edges.
            - Setting entry, finish points, and conditional routing logic.
            - Compiling the graph into an executable pipeline.

        The pipeline ensures research reports are accurate, complete, and properly formatted.
        """
        self.logger = get_logger(self.__class__.__name__)

        # initializing agents
        searcher = SearcherAgent()
        extractor = ExtractorAgent()
        writer = WriterAgent()
        critic = CriticAgent()
        assembler = AssemblerAgent()

        # building graph
        builder = StateGraph(ResearchState)

        # adding nodes
        builder.add_node(searcher.name, searcher.run)
        builder.add_node(extractor.name, extractor.run)
        builder.add_node(writer.name, writer.run)
        builder.add_node(critic.name, critic.run)
        builder.add_node('assembler', assembler.create_final_pdf)

        # adding edges and coditional edges
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

        # compiling grahp
        self.graph = builder.compile()
        self.logger.info('Graph compilation successfull, ResearchAssistant Initialized.')


    def _criticize_con(self, state: ResearchState) -> bool:
        """
        Determines whether the workflow should proceed to the CriticAgent or directly to the AssemblerAgent.

        Args:
            state (ResearchState): Current state of the graph.

        Returns:
            bool: 
                - True -> WriterAgent successfully applied criticism and the report can proceed to assembly.
                - False -> Report expansion finished but requires criticism.
        """
        is_criticized = state.get('is_criticized', False)

        if is_criticized:
            self.logger.info('Based on the criticism, rewriting was successfull. Next Node: AssemblerAgent.')

        else:
            self.logger.info('Expanding was successfull. Next Node: CriticAgent')

        return is_criticized


    def _pass_fail(self, state: ResearchState) -> str:
        """
        Evaluates the outcome of the CriticAgent and decides whether to send the report back for rewriting or proceed to final assembly.

        Args:
            state (ResearchState): Current state of the graph.

        Returns:
            str:
                - 'fail' -> One or more report parts failed review, reroute to WriterAgent for rewriting.
                - 'pass' -> All report parts passed review, proceed to AssemblerAgent.
        """
        for criticism in state.get('criticism').values():
            if criticism != 'PASS':
                self.logger.info('One or more report parts failed critic. Next Node: WriterAgent.')
                return 'fail'
        
        self.logger.info('All the report parts passed the critic, skipping rewriting. Next Node: AssemblerAgent.')
        return 'pass'
    

    def run(self, user_input: str) -> ResearchState:
        """
        Executes the full research pipeline for a given topic.

        ## Steps:
            1. Initializes state with user topic.
            2. Invokes the compiled workflow graph.
            3. Records runtime performance.
            4. Saves the final state upon success or partial state upon failure.

        Args:
            user_input (str): The research topic to investigate.

        Returns:
            ResearchState: Final state containing the complete research report and metadata.
        """
        self.logger.info(f'Starting research on topic "{user_input}"...')
        try:
            # invoking graph and starting performance counter
            start = perf_counter()
            state = {'topic': user_input}
            state = self.graph.invoke(state)
            end = perf_counter()

            # calculating minutes and seconds
            elapsed = end - start
            minutes, seconds = divmod(elapsed, 60)

            # saving the final state
            save_state(state, topic= user_input)

            self.logger.info(f'Total time taken: {int(minutes)}m {seconds:.2f}s')
            self.logger.info(f'Saved final state of the program at .data/{sanitize_filename(user_input)}.json')

            return state
        
        except Exception as e:
            self.logger.exception(f'Error while researching topic {user_input}: {e}')

            save_state(self.graph.get_state(), topic= user_input)
            self.logger.info(f'Saved current state of the program at .data/{sanitize_filename(user_input)}.json')
