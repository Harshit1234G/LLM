from typing_extensions import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str


class ChatBot:
    def __init__(
            self, 
            *, 
            model: str = 'gpt-4o-mini', 
            embedding_model: str = 'text-embedding-3-large',
            temperature: float = 0.3, 
            vector_db_path: str = r'.\faiss_index'
        ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.temperature = temperature

        self.vector_db = self.__load_faiss_index()

        self.llm = ChatOpenAI(
            model= self.model,
            temperature= self.temperature,
        )
        self.retriever = self.__create_retriever()
        self.graph = self.__build_graph()


    def __load_faiss_index(self) -> FAISS:
        vector_store = FAISS.load_local(
            folder_path= self.vector_db_path,
            embeddings= OpenAIEmbeddings(model= self.embedding_model),   # using open ai embeddings
            allow_dangerous_deserialization= True                        #! A dangerous deserializaion of pickle file activated, although it is completely safe for non-server applications.
        )

        return vector_store
    

    def __create_retriever(self) -> MultiQueryRetriever:
        # prompt for generating different versions of the query
        QUERY_PROMPT = PromptTemplate.from_template(
            template= """ROLE: You improve search queries for document retrieval.
            TASK: Generate exactly 4 topic-style search queries (not questions) for the user's question.
            CONSTRAINTS:
            - Preserve all key entities/terms from the user question.
            - Provide: 1 broader, 1 narrower, and 2 specific variants.
            - Each line must be distinct, <=12 words, noun-heavy, no filler words.
            - Use at least one domain-specific/jargon synonym where natural.
            - Maintain the original intent.
            USER QUESTION: {question}
            OUTPUT FORMAT: Return exactly 4 lines, one query per line. No numbering, bullets, quotes, or extra text.
            """
        )

        # creating the MultiQueryRetriever from the llm and prompt
        retriever = MultiQueryRetriever.from_llm(
            retriever= self.vector_db.as_retriever(
                search_type= 'mmr',                    # maximal margin relevance
                search_kwargs= {'k': 5, 'fetch_k': 20} # fetches 20 documents, select only 5
            ),
            llm= self.llm,
            prompt= QUERY_PROMPT,
            include_original= True                     # including original for question related retrieval
        )

        return retriever


    def context_generator(self, state: ChatState) -> ChatState:
        user_question = state['messages'][-1].content
        docs = self.retriever.invoke(user_question)
        context = '\n\n'.join(d.page_content for d in docs)
        return {'context': context}
    

    def chatbot(self, state: ChatState) -> ChatState:
        user_question = state['messages'][-1].content
        context = state.get('context', '')

        system_message = SystemMessage(
            content= """ROLE: You are an AI clone of Harshit, created to represent him in professional conversations.
            TASK: Answer questions about Harshit's background, education, experience, projects and skills strictly based on the provided context.
            AUDIENCE: Primarily HR professionals and recruiters.
            STYLE: Maintain a human-like, professional, friendly, and conversational tone.
            CONSTRAINTS: 
            - Answer strictly using the context.
            - Use bullet points if natural.
            - No speculation or external facts.
            - If context is very short, give a brief 1-2 sentence answer.  
            - If context is rich, answer in <=200 words unless user explicitly asks for detail.  
            - If context is missing or irrelevant, reply strictly with: "Sorry! I can only talk about Harshit's Portfolio.".
            - End every response with a natural follow-up question that encourages engagement.
            """
        )

        rag_prompt = [
            system_message,
            HumanMessage(content= f'Question: {user_question}\n\nContext:\n{context}')
        ]

        response = self.llm.invoke(rag_prompt)
        return {'messages': [response]}

    

    def __build_graph(self) -> StateGraph:
        builder = StateGraph(ChatState)

        builder.add_node('retriever', self.context_generator)
        builder.add_node('chatbot', self.chatbot)

        builder.set_entry_point('retriever')
        builder.add_edge('retriever', 'chatbot')
        builder.set_finish_point('chatbot')

        return builder.compile()
    

    def run(self, user_message: str) -> str:
        state = {'messages': [HumanMessage(content= user_message)]}
        result = self.graph.invoke(state)
        state = result
        return result['messages'][-1].content
    

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()


    chat_bot = ChatBot(vector_db_path= r"E:\Python\LLM\NeuroHarshit\Databases\faiss_index")
    while True:
        question = input('\nQuestion: ')

        if question == 'quit':
            break
        
        response = chat_bot.run(question)
        print('\nResponse:', response)
