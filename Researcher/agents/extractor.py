import json
from langchain_core.prompts import ChatPromptTemplate
from agents import BaseAgent


class ExtractorAgent(BaseAgent):
    def __init__(self):
        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    'ROLE: You are an expert summarizer and knowledge extractor.\n'
                    'TASK: Read the DOCUMENTS for the given TOPIC and extract a structured knowledge base as provided JSON SCHEMA.\n'

                    'GUIDELINES:\n'
                    '- Focus on capturing all key ideas, concepts, results, methods, and implications.\n'
                    '- Do no copy everything, condense where possible, but never drop unique insights.\n'
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
                    '- Finally add a conclusion, don\'t ask any follow-up question.\n'

                    'JSON SCHEMA:\n'
                    '{{\n'
                    '  "topic": "string",\n'
                    '  "sources": [\n'
                    '    {{"id": "string", "title": "string", "authors": ["string", "..."], "source": "string", "url": "string"}}\n'
                    '  ],\n'
                    '  "topics": [\n'
                    '    {{\n'
                    '      "id": "t<number>",\n'
                    '      "title": "string",\n'
                    '      "summary_points": ["string", "..."],\n'
                    '      "subtopics": [\n'
                    '        {{\n'
                    '          "id": "t<number>.<number>",\n'
                    '          "title": "string",\n'
                    '          "summary_points": ["string", "..."],\n'
                    '          "references": ["source_id_1", "source_id_2", "..."]\n'
                    '        }}\n'
                    '      ],\n'
                    '      "references": ["source_id_1", "source_id_2", "..."]\n'
                    '    }}\n'
                    '  ],\n'
                    '  "conclusion": "string"\n'
                    '}}\n'
                ),
                ('human', 'TOPIC: {topic}\n\nDOCUMENTS: {docs}')
            ]
        )

        super().__init__(
            name= 'extractor',
            instructions= prompt,
            temperature= 0.0,
            use_small_model= True
        )


    def run(self, state):
        docs = state.get('wikipedia_docs') + state.get('arxiv_docs')
        knowledge = self.llm.invoke(
            self.instructions.format_messages(topic= state['topic'], docs= docs)
        ).content.strip()

        return {'knowledge': json.loads(knowledge)}
