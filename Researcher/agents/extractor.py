from langchain_core.prompts import ChatPromptTemplate
from agents import BaseAgent


class ExtractorAgent(BaseAgent):
    def __init__(self):
        prompt = ChatPromptTemplate(
            messages= [
                (
                    'system',
                    "ROLE: You are an expert in extracting insights.\n"
                    "TASK: Create a comprehensive summarization of the following DOCUMENTS.\n"
                    "GUIDELINES:\n"
                    "- Focus on capturing all key ideas, concepts, results, methods, and implicaitons.\n"
                    "- Do no copy everything, condense where possible, but never drop unique insights.\n"
                    "- Aim for clarity, completeness, and conciseness.\n"
                    "- Organize output in a structured, pointwise format with headings and subpoints.\n"
                    "- Merge overlapping information from different documents, avoid redundancy.\n"
                    "- Keep explanations detailed enough for future use as a knowledge base.\n"
                    "- If technical methods/equations appear, explain them clearly.\n"
                    "- Do not add any introductory or concluding sentences.\n"
                    "OUTPUT: A detailed, structured, knowledge-rich summary of all the provided documents, balancing completeness with readability."
                ),
                ('human', 'DOCUMENTS: {docs}')
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
        summary = self.llm.invoke(
            self.instructions.format_messages(docs= docs)
        ).content.strip()

        return {'summary': summary}
