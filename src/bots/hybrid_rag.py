import os
from typing import List
from pydantic import SecretStr

# 1. Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- FIXED IMPORTS FOR V1.0+ ---
# We use langchain_classic for components that were moved out of core
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# -------------------------------

# 3. Providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.base import ChatbotInterface

class HybridRAGBot(ChatbotInterface):
    def __init__(self, api_key: str, provider: str, model_name: str):
        super().__init__(api_key, provider, model_name)
        
        self.rag_chain = None
        self.api_key = api_key
        # Wrap key for Pydantic
        secret_key = SecretStr(self.api_key)

        # Initialize LLM
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
        return "Hybrid RAG (Vector + Keyword)"

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

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # --- RETRIEVER 1: Vector Search (Semantic) ---
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            collection_name="hybrid_collection"
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # --- RETRIEVER 2: BM25 (Keyword) ---
        # Note: rank_bm25 must be installed: 'uv add rank_bm25'
        keyword_retriever = BM25Retriever.from_documents(splits)
        keyword_retriever.k = 3

        # --- COMBINE: Ensemble Retriever ---
        # Weights: 0.5 for Vector, 0.5 for Keyword
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )
        
        # Build Chain
        system_prompt = (
            "You are a precise assistant. Use the retrieved context below "
            "to answer the question. If the context doesn't contain the answer, "
            "say you don't know."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

        return f"Processed {len(documents)} pages using Hybrid Search."

    def chat(self, user_query: str) -> str:
        if not self.rag_chain:
            return "⚠️ Please upload documents first."
        try:
            response = self.rag_chain.invoke({"input": user_query})
            return response['answer']
        except Exception as e:
            return f"Error: {str(e)}"