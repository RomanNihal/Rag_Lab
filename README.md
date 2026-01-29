# 🧪 Modular RAG Experimentation Lab

A modular, extensible playground for comparing different Retrieval-Augmented Generation (RAG) architectures and LLMs side-by-side. 

Built with **Streamlit**, **LangChain**, and **uv**.

## 🚀 Features

* **Modular Architecture:** Easily swap "Brains" (RAG strategies) without changing the UI.
* **Multi-Provider Support:** Switch instantly between **Groq**, **OpenAI**, and **Google Gemini** to compare cost vs. quality.
* **Dynamic Model Selection:** Choose specific models (e.g., `llama3-70b` vs `mixtral-8x7b`) on the fly.
* **Local Embeddings:** Uses `all-MiniLM-L6-v2` locally (via HuggingFace) for free, private vector embedding.
* **Modern Stack:** Powered by `uv` for lightning-fast dependency management.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain v0.3
* **Vector DB:** ChromaDB (Local)
* **Package Manager:** uv

## 🧪 Usage

1.  Open the app in your browser.
2.  **Sidebar Config:**
    * **Architecture:** Select the RAG strategy (e.g., "Simple RAG").
    * **Provider:** Choose your LLM provider (Groq, OpenAI, Gemini).
    * **Model:** Select the specific model to test.
    * **API Key:** Paste your API key (keys are not stored persistently).
3.  **Upload Data:** Drag and drop PDF or TXT files.
4.  **Build:** Click "🚀 Build Chatbot" to ingest and index the data.
5.  **Chat:** Ask questions to compare how different models handle your specific documents.

## 🗺️ Roadmap

* [x] Simple RAG (Vector Search)
* [ ] Multi-Modal RAG (Image/Graph support)
* [ ] Agentic RAG (LangGraph with Web Search)
* [ ] Evaluation Metrics (Ragas integration)