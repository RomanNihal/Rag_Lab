import os
from typing import List
from pydantic import SecretStr

# 1. Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Vector Store & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 3. The New Chain Import (Guaranteed to work in v0.3+)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 4. Providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.base import ChatbotInterface

class SimpleRAGBot(ChatbotInterface):
    def __init__(self, api_key: str, provider: str, model_name: str):
        super().__init__(api_key, provider, model_name)
        
        self.rag_chain = None
        self.api_key = api_key
        secret_key = SecretStr(self.api_key)

        # --- DYNAMIC LLM INITIALIZATION ---
        if self.provider == "groq":
            self.llm = ChatGroq(api_key=secret_key, model=self.model_name)
        elif self.provider == "openai":
            self.llm = ChatOpenAI(api_key=secret_key, model=self.model_name)
        elif self.provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(google_api_key=secret_key, model=self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @staticmethod
    def get_name() -> str:
        return "Simple RAG (Modern LCEL)"

    def process_files(self, file_paths: List[str]) -> str:
        documents = []
        for path in file_paths:
            try:
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                elif path.endswith(".txt"):
                    loader = TextLoader(path)
                    documents.extend(loader.load())
            except Exception as e:
                return f"Error loading {path}: {e}"
        
        if not documents:
            return "No valid documents found."

        # Split & Store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            collection_name="rag_collection"
        )
        
        # --- NEW: Create Chain using LCEL (No RetrievalQA) ---
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 1. Create a prompt that tells the LLM how to answer
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 2. Build the chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return f"Processed {len(documents)} pages using {self.model_name}."

    def chat(self, user_query: str) -> str:
        if not self.rag_chain:
            return "⚠️ Please upload documents first."
        try:
            # The new chain expects 'input', not 'query'
            response = self.rag_chain.invoke({"input": user_query})
            return response['answer']
        except Exception as e:
            return f"Error: {str(e)}"