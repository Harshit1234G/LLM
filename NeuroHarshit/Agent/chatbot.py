from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class ChatBot:
    MODEL: str = 'gpt-4o-mini'
    EMBEDDING_MODEL: str = 'text-embedding-3-large'

    def __init__(
            self, 
            vector_db_path: str, 
            *, 
            temperature: int = 0.2
        ) -> None:
        self.vector_store = self._load_faiss_index(vector_db_path)
        self.llm = ChatOpenAI(
            model= self.MODEL,
            temperature= temperature
        )


    def _load_faiss_index(self, path: str) -> FAISS:
        #! A dangerous deserializaion of pickle file activated, although it is completely safe for non-server applications.
        vector_store = FAISS.load_local(
            folder_path= path,
            embeddings= OpenAIEmbeddings(self.EMBEDDING_MODEL),
            allow_dangerous_deserialization= True
        )
        return vector_store
