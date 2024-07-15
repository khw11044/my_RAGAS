# RAGPipeLine.py
from fastapi import FastAPI, HTTPException, Request
import logging

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# Retriever 기법 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ParentDocumentRetriever

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

from langchain.chains.query_constructor.base import AttributeInfo

from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


# from langchain.retrievers import WebResearchRetriever
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
# from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool


# from langchain.chains import RetrievalQAWithSourcesChain

from utils.update import split_document, convert_file_to_documents
from utils.prompt import *
from utils.redis_utils import save_message_to_redis, get_messages_from_redis
from core.redis_config import redis_conn  # Redis 설정 임포트

from langchain.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv
load_dotenv()

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The category of the documents. One of ['1.법률 및 규제', '2.경제 및 시장 분석', '3.정책 및 무역', '4.컨퍼런스 및 박람회, 전시회']",
        type="string",
    ),
    AttributeInfo(
        name="subcategory", 
        description="The Table of Contents of the documents.", 
        type="string"
    ),
    AttributeInfo(
        name="filename",
        description="The name of the document",
        type="string",
    ),
    AttributeInfo(
        name="page_no",
        description="The page of the document",
        type="int",
    ),
    AttributeInfo(
        name="datetimes",
        description="The Date the document was uploaded",
        type="string",
    ),
    
    AttributeInfo(
        name="keyword", 
        description="Keywords in the content", 
        type="string"
    ),
]

from utils.config import config


class Ragpipeline:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = config["similarity_k"]
        self.llm = ChatOpenAI(
            model       = config['llm_predictor']['model_name'],
            temperature = config['llm_predictor']['temperature'],
            streaming   = True,
            callbacks=[StreamCallback()]
        )
        self.vector_store   = self.init_vectorDB()
        self.retriever      = self.init_retriever()
        self.chain          = self.init_chat_chain()
        self.web_retriever  = self.init_web_research_retriever()  
        self.mq_retriever   = self.init_multi_query_retriever()
        self.sq_retriever   = self.init_self_query_retriever()
        self.pd_retriever   = self.init_parent_document_retriever()
        self.web_chain      = self.init_web_chat_chain()
        self.title_chain    = self.init_title_chain()
        self.text_chain     = self.init_text_chain()
        self.session_histories = {}
        self.current_user_email = None
        self.current_session_id = None

    def init_vectorDB(self, persist_dir=config["chroma"]["persist_dir"]):
        embeddings = OpenAIEmbeddings(model=config['embed_model']['model_name'])
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print(f"[초기화] vector_store 초기화 완료")
        return vector_store

    def init_retriever(self):
        
        retriever = self.vector_store.as_retriever(
            search_kwargs = {"score_threshold": 0.5, "k": config["retriever_k"]},
            search_type   = "similarity_score_threshold"
        )
        
        print(f"[초기화] retriever 초기화 완료")
        return retriever
    
    def init_multi_query_retriever(self):
        """사용자의 질문을 여러 개의 유사 질문으로 재생성 """
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=self.retriever, llm=self.llm
        )
        
        return retriever_from_llm
    
    def init_self_query_retriever(self):
        """metadata를 이용해서 필터링해서 정보를 반환"""
        document_content_description = "Brief summary or Report or Explain"
        retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vector_store,
            document_content_description,
            metadata_field_info,
            verbose = True
        )
        
        return retriever
    
    def init_parent_document_retriever(self):
        """청크 유사도를 통해 원본 Document-즉, Parent Document를 모두 참고하는 방법"""
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=300)        # 280 
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)     # 최소 200 이상
        store = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=store,
            child_splitter=child_splitter,
            # parent_splitter=parent_splitter,
        )
        
        return retriever
    
    def init_web_research_retriever(self):
        """ Web Research Retriever 초기화 """            
        search = GoogleSearchAPIWrapper()
        
        # self.vector_store를 써버리면 web search한 내용이 들어가버린다. 
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),
                     persist_directory="./temp_web_db")
        web_retriever = WebResearchRetriever.from_llm(
                    vectorstore=vectorstore, # self.vector_store,
                    llm=self.llm , 
                    search=search, 
                )
        return web_retriever

    def init_chat_chain(self):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        question_answer_chain = create_stuff_documents_chain(
            self.llm, qa_prompt)
        
        rag_chat_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)
        
        print("[초기화] RAG chain 초기화 완료")
        return rag_chat_chain
    
    def init_web_chat_chain(self):
        # 1. 사용자의 질문 문맥화 <- 프롬프트 엔지니어링
        history_aware_retriever = create_history_aware_retriever(                           # 대화 기록을 가져온 다음 이를 사용하여 검색 쿼리를 생성하고 이를 기본 리트리버에 전달
            self.llm, self.web_retriever, contextualize_q_prompt
        )
        
        # 2. 응답 생성 + 프롬프트 엔지니어링
        question_answer_chain = create_stuff_documents_chain(self.llm, web_qa_prompt)           # 문서 목록을 가져와서 모두 프롬프트로 포맷한 다음 해당 프롬프트를 LLM에 전달합니다.
        
        # 3. 최종 체인 생성
        rag_chat_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # 사용자 문의를 받아 리트리버로 전달하여 관련 문서를 가져옵니다. 그런 다음 해당 문서(및 원본 입력)는 LLM으로 전달되어 응답을 생성

        return rag_chat_chain
    
    def init_title_chain(self):
        question_answer_chain = create_stuff_documents_chain(
            self.llm, title_generator_prompt)
        rag_title_chain = create_retrieval_chain(
            self.retriever, question_answer_chain)
        print("[초기화] RAG title chain 초기화 완료")
        return rag_title_chain
    # title은 post를 이용해서 만드는 것도..?
    
    def init_text_chain(self):
        question_answer_chain = create_stuff_documents_chain(
            self.llm, text_generator_prompt)
        rag_text_chain = create_retrieval_chain(
            self.retriever, question_answer_chain)
        print("[초기화] RAG post chain 초기화 완료")
        return rag_text_chain
        
    

    def chat_generation(self, question: str) -> dict:
        def get_session_history(session_id=None, user_email=None):
            session_id = session_id if session_id else self.current_session_id
            user_email = user_email if user_email else self.current_user_email
            if session_id not in self.session_histories:
                self.session_histories[session_id] = ChatMessageHistory()
                # Redis에서 세션 히스토리 불러오기
                
                # history_messages = get_messages_from_redis(user_email, session_id)
                # for message in history_messages:
                #     self.session_histories[session_id].add_message(HumanMessage(content=message))
                    
                print(f"[히스토리 생성] 새로운 히스토리를 생성합니다. 세션 ID: {session_id}, 유저: {user_email}")
            return self.session_histories[session_id]

        results = self.vector_store.similarity_search_with_score(question, k=3)
        
        print(results[0][1])
        
        if results[0][1] > 0.3:
            print('제가 잘 모르는 내용이라서, 검색한 내용을 알려 드릴께요.')
            final_chain = self.web_chain
        else:
            final_chain = self.chain
        
        conversational_rag_chain = RunnableWithMessageHistory(
                final_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": self.current_session_id}}
        )

        # Redis에 세션 히스토리 저장
        save_message_to_redis(self.current_user_email,
                              self.current_session_id, question)
        save_message_to_redis(self.current_user_email,
                              self.current_session_id, response["answer"])

        # print(f'[응답 생성] 실제 모델 응답: response => \n{response}\n')
        # print(response["answer"])
        # print(type(response["answer"]))
        return response
    
    def title_generation(self, question: str):
        title_chain = self.title_chain # title prompt + retrieval chain 선언
        response = title_chain.invoke({'input':question, 'num':str(3)})
        # print(response)
        return response
        
    def text_generation(self, question: str):
        
        text_chain = self.text_chain # title prompt + retrieval chain 선언
        response = text_chain.invoke({'input':question})
        # print(response)
        
        return response

    def update_vector_db(self, file) -> bool:
        """
        벡터 스토어 업데이트: 새로운 문서 추가 시 호출 
        PDF파일 또는 CSV파일 또는 hwp파일, word 파일 등 
        기존 DB에 유사도 검사를 통해 중복되는 내용은 추가하지 않음 
        """
        upload_documents = convert_file_to_documents(self.vector_store, file, self.SIMILARITY_THRESHOLD)

        if upload_documents:
            self.vector_store.add_documents(upload_documents)
            print(f"Added {len(upload_documents)} new documents to the vector store")
            return True
        else:
            print('모두 유사한 청크로 판단되어 해당 문서가 저장되지 않음')
            return False

    def delete_vector_db_by_doc_id(self, doc_id):
        """
        주어진 문서 ID에 해당하는 벡터 임베딩을 삭제
        """
        # 벡터 데이터베이스에서 모든 문서 가져오기
        all_documents = self.vector_store._collection.get(include=["metadatas"])
        documents_to_delete = [doc_id for i, metadata in enumerate(all_documents["metadatas"]) if metadata.get("doc_id") == doc_id]
        if documents_to_delete:
            self.vector_store._collection.delete(ids=documents_to_delete)
            print(f"[벡터 DB 삭제] 문서 ID [{doc_id}]의 임베딩을 벡터 DB에서 삭제했습니다.")
        else:
            print(f"[벡터 DB 삭제 실패] 문서 ID [{doc_id}]에 대한 임베딩을 찾을 수 없습니다.")
    
    
    def print_text(self, question: str):
        
        
        return question + question