from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

class ChatBot:
    MODEL: str = 'gpt-4o-mini'
    EMBEDDING_MODEL: str = 'text-embedding-3-large'

    def __init__(self, vector_db_path: str, *, temperature: int = 0.2) -> None:
        self.vector_store = self._load_faiss_index(vector_db_path)
        self.llm = ChatOpenAI(
            model= self.MODEL,
            temperature= temperature
        )
        self.retriever = self._create_retrieve()


    def _load_faiss_index(self, path: str) -> FAISS:
        #! A dangerous deserializaion of pickle file activated, although it is completely safe for non-server applications.
        vector_store = FAISS.load_local(
            folder_path= path,
            embeddings= OpenAIEmbeddings(model= self.EMBEDDING_MODEL),
            allow_dangerous_deserialization= True
        )
        return vector_store
    

    def _create_retrieve(self) -> MultiQueryRetriever:
        QUERY_PROMPT = PromptTemplate.from_template(
            template= """ROLE: You improve search queries for document retrieval.
TASK: Generate exactly 4 topic-style search queries (not questions) for the user's question.

CONSTRAINTS:
- Preserve all key entities/terms from the user question.
- Provide: 1 broader, 1 narrower, and 2 specific/technical variants.
- Each line must be distinct, ≤12 words, noun-heavy, no filler words.
- Use at least one domain-specific/jargon synonym where natural.
- Maintain the original intent.

USER QUESTION: {question}

OUTPUT FORMAT: Return exactly 4 lines, one query per line. No numbering, bullets, quotes, or extra text.
"""
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever= self.vector_store.as_retriever(
                search_type= 'mmr', 
                search_kwargs= {'k': 5, 'fetch_k': 20}
            ),
            llm= self.llm,
            prompt= QUERY_PROMPT,
            include_original= True
        )

        return retriever
    

if __name__ == '__main__':
    # Set logging for the queries
    import logging

    logging.basicConfig()
    print('Generated Queries: ')
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    chat_bot = ChatBot(
        vector_db_path= r"E:\Python\LLM\NeuroHarshit\Databases\faiss_index"
    )
    retrieved_data = chat_bot.retriever.invoke('What is The Joy of Computing using Python?')
    
    print('Retrieved Chunks: ')
    for index, data in enumerate(retrieved_data, start= 1):
        print(f'{index}: {data.page_content}\n\n')
