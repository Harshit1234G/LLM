from langchain_core.prompts import ChatPromptTemplate

from agents import BaseAgent
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
                    '- If multiple sources support the same statement, include them as [id1, id2].\n'
                    '- Do not invent or include new references.\n'
                    
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
                ('human', 'INPUT JSON: {input_json}\n')
            ]
        )

        super().__init__(
            name= 'writer',
            instructions= prompt,
            temperature= 0.5
        )
        self.logger.info('WriterAgent initialized.')


    def run(self, state):
        self.logger.info('WriterAgent started.')
        knowledge = state.get('knowledge', None)

        if knowledge is None:
            self.logger.error('No value for knowledge is provided.')
            raise ValueError('No value for knowledge is provided.')
        
        state['report_parts'] = []

        for topic in knowledge.get('topics'):
            self.logger.info(f'Expanding topic: {topic.get("title")}')
            try:
                text = self.llm.invoke(
                    self.instructions.format_messages(input_json= topic)
                ).content.strip()
                state['report_parts'].append(text)

                self.logger.info(f'Successfully expanded topic: {topic.get("title")}')

            except Exception as e:
                self.logger.exception(f'Error while expanding topic {topic.get("title")}: {e}')

        self.logger.info('WriterAgent finished.')
        return state
