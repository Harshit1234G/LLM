from langchain_core.prompts import ChatPromptTemplate

from agents import BaseAgent, ResearchState
from utils import get_logger


class WriterAgent(BaseAgent):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    'ROLE: You are a professional research writer with expertise in creating academic-style reports.\n'
                    'TASK: Expand the given INPUT JSON containing topic, subtopic, summaries and references into a clear, detailed, and well-structured section for a research report.\n'

                    'CITATION RULES:\n'
                    '- Use only the provided references.\n'
                    '- Inline citations must appear immediately after the statements they support, in the format [id].\n'
                    '- If multiple sources support the same statement, include them as [id1][id2].\n'
                    '- Do not invent or include new references.\n'

                    'REWRITING RULES:\n'
                    '- These rules apply ONLY if CRITICISM and PREVIOUS RESPONSE are provided.\n'
                    '- The Writer must strictly follow the instructions in CRITICISM when rewriting.\n'
                    '- If the criticism points out issues such as lack of clarity, poor flow, missing details, or incorrect structure:\n'
                    '  - Revise the PREVIOUS RESPONSE accordingly.\n'
                    '  - Ensure all mentioned issues are fully corrected.\n'
                    '- If the criticism requests specific improvements (e.g., expand a section, simplify tone, fix structure):\n'
                    '  - Implement those changes exactly as stated.\n'
                    '- If the criticism says the PREVIOUS RESPONSE is completely off-topic or unusable:\n'
                    '  - Discard the response entirely and output a new line character (\\n).\n'
                    '- Do not ignore or override the criticism under any circumstances.\n'
                    '- Always preserve the original GUIDELINES for academic tone, structure, citations, and formatting while applying the criticism.\n'
                    
                    'GUIDELINES:\n'
                    '- Use the summaries as the sole factual basis; do not introduce external information.\n'
                    '- Expand, explain, and elaborate the summaries into 2-4 coherent paragraphs per topic.\n'
                    '- For subtopics, write 1-3 paragraphs each, structured as subsections within the topic.\n'
                    '- Ensure that every provided subtopic is addressed in detail under its own subsection.\n'
                    '- Merge overlapping points naturally to avoid repetition.\n'
                    '- Maintain a formal, academic, third-person tone throughout.\n'
                    '- Ensure logical paragraph flow with smooth transitions.\n'
                    '- Do not add introductions or conclusions beyond what is directly relevant.\n'

                    'OUTPUT:\n'
                    '- Output must be in valid Markdown.\n'
                    '- Use "##" for topic headings and "###" for subtopic headings.\n'
                    '- Write the expanded content in paragraphs under each heading.\n'
                    '- Do not include anything outside the Markdown structure.\n'
                ),
                ('human', 'INPUT JSON: {input_json}\n\nCRITICISM: {criticism}\n\nPREVIOUS RESPONSE: {prev_response}')
            ]
        )

        super().__init__(
            name= 'writer',
            instructions= prompt,
            temperature= 0.3
        )
        self.logger.info('WriterAgent initialized.')


    def _expand_topic(self, state: ResearchState) -> list[str]:
        knowledge = state.get('knowledge')

        if knowledge is None:
            self.logger.error('No value for knowledge is provided.')
            raise ValueError('No value for knowledge is provided.')
        
        report_parts = []
        topics = knowledge.get('topics', [])

        for index, topic in enumerate(topics):
            title = topic.get('title', f'Untitled-{index}')
            self.logger.info(f'Expanding topic [{index + 1}]: {title}')

            try:
                prompt = self.instructions.format_messages(
                    input_json= topic,
                    criticism= '',
                    prev_response= ''
                )
                text = self.llm.invoke(prompt).content.strip()
                report_parts.append(text)

                self.logger.info(f'Successfully expanded topic [{index + 1}]: {title}')

            except Exception as e:
                self.logger.exception(f'Error while expanding topic [{index + 1}] {title}: {e}')
                report_parts.append('')

        return report_parts
    

    def _rewrite_topic(self, state: ResearchState) -> ResearchState:
        criticisms = state.get('criticism', {})
        topics = state.get('knowledge', {}).get('topics', [])

        for index, critique in criticisms.items():
            if critique == 'PASS':
                self.logger.info(f'Topic [{index + 1}] passed without changes.')
                continue

            title = topics[index].get('title', f'Untitled-{index}')
            self.logger.info(f'Rewriting topic [{index + 1}]: {title}')

            try:
                prompt = self.instructions.format_messages(
                    input_json= topics[index],
                    criticism= critique,
                    prev_response= state['report_parts'][index]
                )
                text = self.llm.invoke(prompt).content.strip()
                state['report_parts'][index] = text
                self.logger.info(f'Successfully rewrote topic [{index + 1}]: {title}')

            except Exception as e:
                self.logger.exception(f'Error while rewriting topic [{index + 1}] {title}: {e}')

        return state
        

    def run(self, state):
        self.logger.info('WriterAgent started.')
        
        if state.get('is_criticized', False):
            updated_state = self._rewrite_topic(state)
            self.logger.info('WriterAgent finished rewritting.')
            return updated_state

        report_parts = self._expand_topic(state)        

        self.logger.info('WriterAgent finished expanding.')
        return {'report_parts': report_parts}
