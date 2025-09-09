from langchain_core.prompts import ChatPromptTemplate

from agents import BaseAgent
from utils import get_logger


class CriticAgent(BaseAgent):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    'ROLE: You are a strict academic fact-checker and critic.\n'
                    'TASK: Review the given INPUT JSON (knowledge base) and the WRITER OUTPUT (expanded section).\n'
                    'Determine whether the WRITER OUTPUT strictly adheres to the facts in the INPUT JSON without introducing hallucinations.\n'

                    'CRITICISM RULES:\n'
                    '- Compare every statement in the WRITER OUTPUT against the INPUT JSON.\n'
                    '- If the content is factually correct and consistent with the JSON, respond only with the word "PASS".\n'
                    '- If any hallucinations, inaccuracies, or unsupported claims are found, provide a detailed criticism:\n'
                    '   * Identify the incorrect or unsupported statements.\n'
                    '   * Explain why they are incorrect, vague, or hallucinated.\n'
                    '   * Refer to the specific topic/subtopic/summary point in the INPUT JSON that contradicts it.\n'
                    '   * Do not rewrite the section yourself, only provide criticism.\n'

                    'GUIDELINES:\n'
                    '- Be strict: even minor factual drift should be flagged.\n'
                    '- Do not invent new information outside of the JSON.\n'
                    '- Maintain a concise and clear tone in your criticism.\n'
                    '- Ensure your response is actionable for rewriting.\n'

                    'OUTPUT:\n'
                    '- If everything is valid, respond exactly with "PASS".\n'
                    '- If issues exist, respond with a structured criticism (full sentences, plain text).\n'
                ),
                ('human', 'INPUT JSON: {input_json}\n\nWRITER OUTPUT: {writer_output}')
            ]
        )


        super().__init__(
            name= 'critic',
            instructions= prompt,
            temperature= 0.1,
            use_small_model= True
        )
        self.logger.info('CriticAgent initialized.')


    def run(self, state):
        self.logger.info('CriticAgent started.')
        report_parts = state.get('report_parts', None)

        if report_parts is None:
            self.logger.error('No value for report_parts is provided.')
            raise ValueError('No value for report_parts is provided.')
        
        criticism = {}

        for index, part in enumerate(report_parts):
            self.logger.info(f'Criticizing part: {index + 1}')

            try:
                prompt = self.instructions.format_messages(
                    input_json= state.get('knowledge'), 
                    writer_output= part
                )
                response = self.llm.invoke(prompt).content.strip()
                criticism[index] = response

                self.logger.info(f'Successfully criticized part {index + 1}, Status: {response if response == 'PASS' else 'FAIL'}')

            except Exception as e:
                self.logger.exception(f'Error while criticising part {index + 1}: {e}')
                return

        self.logger.info('CriticAgent finished.')
        return {
            'criticism': criticism,
            'is_criticized': True
        }
