import streamlit as st
import os
import shutil
from src.bots.simple_rag import SimpleRAGBot
from src.bots.hybrid_rag import HybridRAGBot

PAGE_TITLE = "Modular RAG Lab"
TEMP_DIR = "temp_data"

# --- CONFIGURATION MAPS ---
AVAILABLE_BOTS = {
    SimpleRAGBot.get_name(): SimpleRAGBot,
    HybridRAGBot.get_name(): HybridRAGBot,
}

# The Map: Provider -> List of Models
PROVIDER_MODELS = {
    "Groq": [
        "llama3-8b-8192", 
        "llama3-70b-8192", 
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ],
    "OpenAI": [
        "gpt-4o-mini", 
        "gpt-4o", 
        "gpt-3.5-turbo"
    ],
    "Gemini": [
        "gemini-2.5-flash", 
        "gemini-2.5-pro"
    ]
}

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_bot" not in st.session_state:
    st.session_state.active_bot = None

with st.sidebar:
    st.header("⚙️ Lab Settings")
    
    # 1. Architecture Selection
    selected_bot_name = st.selectbox("1. RAG Architecture", list(AVAILABLE_BOTS.keys()))

    # 2. Provider Selection
    selected_provider = st.selectbox("2. LLM Provider", list(PROVIDER_MODELS.keys()))

    # 3. Model Selection (Dynamic based on Provider)
    available_models = PROVIDER_MODELS[selected_provider]
    selected_model = st.selectbox("3. Specific Model", available_models)
    
    # 4. API Key
    api_key = st.text_input(f"Enter {selected_provider} API Key", type="password")
    
    st.divider()
    
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "txt"])
    
    if st.button("🚀 Build Chatbot"):
        if not api_key:
            st.error("API Key is required!")
        elif not uploaded_files:
            st.error("Upload files first!")
        else:
            with st.spinner(f"Initializing {selected_model}..."):
                if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR)
                
                saved_paths = []
                for f in uploaded_files:
                    path = os.path.join(TEMP_DIR, f.name)
                    with open(path, "wb") as file: file.write(f.getbuffer())
                    saved_paths.append(path)
                
                # Initialize the specific bot with ALL parameters
                bot_class = AVAILABLE_BOTS[selected_bot_name]
                st.session_state.active_bot = bot_class(
                    api_key=api_key, 
                    provider=selected_provider.lower(),
                    model_name=selected_model
                )
                
                status = st.session_state.active_bot.process_files(saved_paths)
                st.success(status)
                st.session_state.messages = []

# --- CHAT INTERFACE ---
st.title(f"🤖 {PAGE_TITLE}")

if st.session_state.active_bot:
    st.caption(f"Architecture: **{selected_bot_name}** | Model: **{selected_model}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.active_bot:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = st.session_state.active_bot.chat(prompt)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})