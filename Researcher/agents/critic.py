from langchain_core.prompts import ChatPromptTemplate

from agents import BaseAgent
from utils import get_logger


class CriticAgent(BaseAgent):
    def __init__(self):
        """Reviews the Writer's output against the knowledge base. Detects hallucinations, unsupported claims, or factual drift. Provides corrective feedback or validates correctness.
        """
        self.logger = get_logger(self.__class__.__name__)

        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    'ROLE: You are an academic fact-checker and critic with adaptive strictness.\n'
                    'TASK: Review the given INPUT JSON (knowledge base) and the WRITER OUTPUT (expanded section).\n'
                    'Your strictness should depend on the severity of issues:\n'
                    '- Minor issues (stylistic drift, harmless rephrasing) -> flag softly and suggest improvement only if necessary.\n'
                    '- Moderate issues (slightly vague, missing details, unclear flow) -> provide constructive criticism.\n'
                    '- Severe issues (hallucinations, contradictions, unsupported claims) -> respond with very strict criticism.\n'

                    'CRITICISM RULES:\n'
                    '- Compare every statement in the WRITER OUTPUT against the INPUT JSON.\n'
                    '- If the content is factually correct and consistent with the JSON, respond only with "PASS".\n'
                    '- If issues exist, provide a structured criticism:\n'
                    '   * Identify the incorrect/unsupported statements.\n'
                    '   * Explain why they are problematic.\n'
                    '   * Refer to the specific topic/subtopic/summary point in the INPUT JSON.\n'
                    '- Do not rewrite the section yourself.\n'

                    'GUIDELINES:\n'
                    '- Be situational: soft for small issues, harsh for serious factual errors.\n'
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
            temperature= 0.2
        )
        self.logger.info('CriticAgent initialized.')


    def run(self, state):
        """Reviews the Writer's output against the knowledge base. Detects hallucinations, unsupported claims, or factual drift. Provides corrective feedback or validates correctness.

        Args:
            state (ResearchState): Current state of the graph.

        Raises:
            ValueError: If no value for `report_parts` is provided.

        Returns:
            ResearchState: Updated state with `criticism` and `is_criticized`
        """
        self.logger.info('CriticAgent started.')
        report_parts = state.get('report_parts', None)

        if report_parts is None:
            self.logger.error('No value for report_parts is provided.')
            raise ValueError('No value for report_parts is provided.')
        
        criticism = {}

        # criticizing each part one-by-one
        for index, part in enumerate(report_parts):
            self.logger.info(f'Criticizing part: {index + 1}')

            try:
                prompt = self.instructions.format_messages(
                    input_json= state.get('knowledge'), 
                    writer_output= part
                )
                response = self.llm.invoke(prompt).content.strip()
                criticism[index] = response

                # status printing in log
                self.logger.info(f'Successfully criticized part {index + 1}, Status: {response if response == 'PASS' else 'FAIL'}')

            except Exception as e:
                self.logger.exception(f'Error while criticising part {index + 1}: {e}')
                return

        self.logger.info('CriticAgent finished.')
        return {
            'criticism': criticism,
            'is_criticized': True
        }
