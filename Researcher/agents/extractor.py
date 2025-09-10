import json
from langchain_core.prompts import ChatPromptTemplate

from agents import BaseAgent
from utils import get_logger


class ExtractorAgent(BaseAgent):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    'ROLE: You are an expert summarizer and knowledge extractor.\n'
                    'TASK: Read the DOCUMENTS for the given TOPIC and extract a structured knowledge base as provided JSON SCHEMA.\n'

                    'GUIDELINES:\n'
                    '- Focus on capturing all key ideas, concepts, results, methods, and implications.\n'
                    '- Do not copy everything, condense where possible, but never drop unique insights.\n'
                    '- Aim for clarity, completeness, and conciseness.\n'
                    '- Merge overlapping information from different documents, avoid redundancy.\n'
                    '- Keep comprehensive and detailed summary.\n'
                    '- If technical methods/equations appear, explain them clearly.\n'
                    '- Only add details that are relevant to the TOPIC.\n'
                    '- Do not add any introductory or concluding sentences.\n'

                    'OUTPUT FORMAT RULES:\n'
                    '- Your output must be a valid JSON object that strictly follows the schema below.\n'
                    '- Do not include any text outside the JSON.\n'
                    '- Add sources for all the documents.\n'
                    '- Each summary_points list must contain as many items as needed to capture all insights.\n'
                    '- Each topic may contain subtopics with their own summaries.\n'
                    '- Include references for every topic/subtopic by linking to source IDs.\n'
                    '- Add an abstract of around 150-300 words which summarizes the purpose, methods, key findings, and conclusions.'
                    '- Finally add a conclusion, don\'t ask any follow-up question.\n'

                    'JSON SCHEMA:\n'
                    '{{\n'
                    '  "topic": "string (non-empty)",\n'
                    '  "sources": [\n'
                    '    {{\n'
                    '      "id": "integer (unique, required)",\n'
                    '      "title": "string (non-empty)",\n'
                    '      "authors": ["string (non-empty)", "..."],\n'
                    '      "source": "string (Wikipedia or arXiv, non-empty)",\n'
                    '      "url": "string (valid URL)"\n'
                    '    }}\n'
                    '  ],\n'
                    '  "topics": [\n'
                    '    {{\n'
                    '      "id": "string matching pattern: t<number>",\n'
                    '      "title": "string (non-empty)",\n'
                    '      "summary_points": ["string (non-empty)", "... (at least 1 required)"],\n'
                    '      "subtopics": [\n'
                    '        {{\n'
                    '          "id": "string matching pattern: t<number>.<number>",\n'
                    '          "title": "string (non-empty)",\n'
                    '          "summary_points": ["string (non-empty)", "... (at least 1 required)"],\n'
                    '          "references": ["integer (must match a valid source id)", "... (at least 1 required)"]\n'
                    '        }}\n'
                    '      ],\n'
                    '      "references": ["integer (must match a valid source id)", "... (at least 1 required)"]\n'
                    '    }}\n'
                    '  ],\n'
                    '  "abstract": "string (non-empty)",\n'
                    '  "conclusion": "string (non-empty)"\n'
                    '}}\n'
                ),
                ('human', 'TOPIC: {topic}\n\nDOCUMENTS: {docs}')
            ]
        )

        super().__init__(
            name= 'extractor',
            instructions= prompt,
            temperature= 0.0
        )
        self.logger.info('ExtractorAgent initialized.')


    def run(self, state):
        topic = state.get('topic')
        self.logger.info(f'Starting extraction for topic: "{topic}"')

        docs = state.get('wikipedia_docs', '') + state.get('arxiv_docs', '')
        self.logger.info('Combined the Wikipedia and arXiv documents.')

        try:
            response = self.llm.invoke(
                self.instructions.format_messages(topic= topic, docs= docs)
            ).content.strip()
            self.logger.info('LLM response received.')

            knowledge = json.loads(response)
            self.logger.info(f'Successfully parsed knowledge JSON for topic "{topic}"')

        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse LLM output as JSON: {e}')
            self.logger.debug(f'Raw LLM output:\n{response}')

        return {'knowledge': knowledge}
