import os
import tempfile
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import requests
import json
import time
from typing import Dict, List, Optional

from document_processor import DocumentProcessor
from vector_store import VectorStore

load_dotenv()

st.set_page_config(
    page_title="Custom Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    h1 {
        color: white !important;
        text-align: center;
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .stButton > button[kind="secondary"] {
        background: linear-gradient(45deg, #6c757d, #495057);
    }
    .chat-message {
        padding: 15px 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        animation: fadeIn 0.4s;
        max-width: 85%;
        line-height: 1.5;
    }
    .user-message {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        margin-left: auto;
        margin-right: 10px;
    }
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        margin-right: auto;
        margin-left: 10px;
        border: 1px solid #e9ecef;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 5px 0;
    }
    .badge-document {
        background: #2196f3;
        color: white;
    }
    .badge-openrouter {
        background: #4caf50;
        color: white;
    }
    .badge-mixed {
        background: #ff9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    if "uploaded_file_size" not in st.session_state:
        st.session_state.uploaded_file_size = None
    
    if "openrouter_configured" not in st.session_state:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            st.session_state.openrouter_configured = True
        else:
            st.session_state.openrouter_configured = False
    
    if "answering_mode" not in st.session_state:
        st.session_state.answering_mode = "Smart (Document + AI)"
    
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "questions_asked": 0,
            "document_answers": 0,
            "ai_answers": 0,
            "total_chunks": 0
        }

initialize_session_state()

def call_openrouter_api(messages: List[Dict], temperature: float = 0.3) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    
    if not api_key:
        return "Error: API key not configured. Please check your .env file."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Document Chatbot"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 800
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            if response.status_code == 429:
                error_msg = "Rate limit exceeded. Please wait a moment and try again."
            elif response.status_code == 401:
                error_msg = "Invalid API key. Please check your .env file."
            return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "Error: Connection failed. Please check your internet connection."
    except Exception as e:
        return f"Error calling API: {str(e)}"

def get_smart_answer(query: str, context: Optional[str] = None) -> Dict:
    messages = []
    
    if context:
        system_prompt = """You are a smart document assistant. Your task:

1. FIRST, answer based ONLY on the provided document context
2. If the answer is COMPLETELY in the context, provide it clearly
3. If the answer is PARTIALLY in the context, use what's there and supplement with general knowledge
4. If the answer is NOT in the context, provide a helpful general answer
5. ALWAYS indicate your sources clearly

Format your response:
- Start with "Based on your document:" if using document
- Start with "Based on general knowledge:" if not in document
- Use "Mixed sources:" for partial matches
- Be accurate, concise, and helpful"""
        
        user_content = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Please provide the best possible answer following the instructions above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
    else:
        system_prompt = "You are a helpful AI assistant. Provide accurate, helpful answers to the user's questions."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    
    if st.session_state.conversation_history:
        history = st.session_state.conversation_history[-4:]
        for msg in history:
            messages.insert(-1, msg)
    
    answer = call_openrouter_api(messages, temperature=0.2)
    
    if context:
        if "Based on your document" in answer:
            source = "document"
            st.session_state.stats["document_answers"] += 1
        elif "Based on general knowledge" in answer:
            source = "openrouter"
            st.session_state.stats["ai_answers"] += 1
        elif "Mixed sources" in answer:
            source = "mixed"
            st.session_state.stats["document_answers"] += 0.5
            st.session_state.stats["ai_answers"] += 0.5
        else:
            source = "openrouter"
            st.session_state.stats["ai_answers"] += 1
    else:
        source = "openrouter"
        st.session_state.stats["ai_answers"] += 1
    
    return {
        "answer": answer,
        "source": source,
        "context_used": bool(context)
    }

def process_uploaded_document(uploaded_file) -> bool:
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.session_state.uploaded_file_size = f"{file_size_mb:.2f} MB"
        
        processor = DocumentProcessor()
        chunks = processor.process_document(tmp_path)
        
        if not chunks:
            st.error("Could not extract readable text from this document.")
            os.unlink(tmp_path)
            return False
        
        metadata = [{
            "source_file": uploaded_file.name,
            "chunk_id": i,
            "word_count": len(chunk.split()),
            "char_count": len(chunk),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        } for i, chunk in enumerate(chunks)]
        
        st.session_state.vector_store.clear()
        st.session_state.vector_store.add_documents(chunks, metadata)
        
        st.session_state.document_processed = True
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.stats["total_chunks"] = len(chunks)
        
        st.session_state.messages = []
        st.session_state.conversation_history = []
        
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def render_sidebar():
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")
        
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            help="Supported: PDF, Word, Text, Markdown",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"File: {uploaded_file.name}\nSize: {file_size:.1f} KB")
        
        col1, col2 = st.columns(2)
        with col1:
            if uploaded_file and not st.session_state.processing:
                if st.button("Process", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = process_uploaded_document(uploaded_file)
                        if success:
                            st.success("Processed successfully!")
                    st.session_state.processing = False
                    st.rerun()
        
        with col2:
            if st.session_state.document_processed:
                if st.button("Clear", type="secondary", use_container_width=True):
                    st.session_state.vector_store.clear()
                    st.session_state.document_processed = False
                    st.session_state.uploaded_file_name = None
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.session_state.stats["total_chunks"] = 0
                    st.success("Document cleared!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Document Status")
        
        if st.session_state.document_processed:
            info = st.session_state.vector_store.get_info()
            
            with st.expander(f"{st.session_state.uploaded_file_name}", expanded=False):
                st.write(f"Chunks: {info['total_documents']:,}")
                st.write(f"Total Words: {info['total_words']:,}")
                st.write(f"Embeddings: {info['embedding_type']}")
        else:
            st.info("No document uploaded yet")
        
        st.markdown("---")
        st.markdown("### Answering Mode")
        
        mode = st.radio(
            "Select answering strategy:",
            ["Smart (Document + AI)", "Document Only", "AI Only"],
            index=0,
            label_visibility="collapsed",
            key="mode_selector"
        )
        st.session_state.answering_mode = mode
        
        st.markdown("---")
        st.markdown("### API Status")
        
        if st.session_state.openrouter_configured:
            st.success("AI API Connected")
            st.info("Ready to answer questions")
        else:
            st.error("API Key Missing")
            st.info("Add API key to .env file")
        
        st.markdown("---")
        st.markdown("### Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.stats["questions_asked"])
        with col2:
            if st.session_state.document_processed:
                st.metric("Chunks", st.session_state.stats["total_chunks"])

def render_chat_interface():
    st.markdown("# Custom Document Chatbot")
    
    mode = st.session_state.answering_mode
    if mode == "Smart (Document + AI)":
        st.success("Smart Mode Active: Using document + AI knowledge")
    elif mode == "Document Only":
        st.warning("Document Only: Answers strictly from uploaded document")
    else:
        st.info("AI Only: General AI responses")
    
    if st.session_state.document_processed:
        st.info(f"Active Document: {st.session_state.uploaded_file_name}")
    else:
        st.warning("Upload a document to enable document-based answers")
    
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                if "source" in msg:
                    source = msg["source"]
                    if source == "document":
                        st.markdown(f'<div class="source-badge badge-document">From Document</div>', 
                                   unsafe_allow_html=True)
                    elif source == "openrouter":
                        st.markdown(f'<div class="source-badge badge-openrouter">AI Knowledge</div>', 
                                   unsafe_allow_html=True)
                    elif source == "mixed":
                        st.markdown(f'<div class="source-badge badge-mixed">Mixed Sources</div>', 
                                   unsafe_allow_html=True)
                
                message(msg["content"], is_user=False, key=f"assistant_{i}")
    
    query = st.chat_input("Ask me anything about your document or any topic...")
    
    if query:
        st.session_state.stats["questions_asked"] += 1
        
        st.session_state.messages.append({"role": "user", "content": query})
        
        mode = st.session_state.answering_mode
        
        if mode == "AI Only" or not st.session_state.document_processed:
            with st.spinner("Thinking..."):
                response = get_smart_answer(query, context=None)
        
        elif mode == "Document Only":
            with st.spinner("Searching document..."):
                search_results = st.session_state.vector_store.search(query, k=3)
                
                if search_results:
                    context = "\n\n".join([f"[Chunk {r['rank']}] {r['content']}" 
                                         for r in search_results])
                    response = get_smart_answer(query, context=context)
                else:
                    response = {
                        "answer": "No relevant information found in your document.\n\nPlease ask something related to the document content, or switch to Smart/AI mode for general answers.",
                        "source": "document",
                        "context_used": True
                    }
        
        else:
            with st.spinner("Searching document..."):
                search_results = st.session_state.vector_store.search(query, k=3)
                
                if search_results:
                    best_score = max([r.get('score', 0) for r in search_results])
                    
                    if best_score > 0.2:
                        context = "\n\n".join([f"[Chunk {r['rank']}] {r['content']}" 
                                             for r in search_results])
                        
                        with st.spinner("Combining document & AI knowledge..."):
                            response = get_smart_answer(query, context=context)
                        
                        if best_score < 0.4:
                            response["answer"] = f"Note: This topic has weak relevance to your document (similarity: {best_score:.1%})\n\n" + response["answer"]
                    
                    else:
                        context = "\n\n".join([f"[Chunk {r['rank']}] {r['content']}" 
                                             for r in search_results[:1]])
                        
                        with st.spinner("Getting AI answer with weak document context..."):
                            response = get_smart_answer(query, context=context)
                        
                        response["answer"] = f"Note: Weak document relevance ({best_score:.1%} similarity)\n\n" + response["answer"]
                
                else:
                    with st.spinner("Getting general AI answer..."):
                        response = get_smart_answer(query, context=None)
                    
                    response["answer"] = "Note: No relevant content found in your document. Here's a general answer:\n\n" + response["answer"]
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"],
            "source": response.get("source", "openrouter")
        })
        
        st.session_state.conversation_history.append({"role": "user", "content": query})
        st.session_state.conversation_history.append({"role": "assistant", "content": response["answer"]})
        
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        
        st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Questions Asked", st.session_state.stats["questions_asked"])
    
    with col2:
        if st.session_state.document_processed:
            info = st.session_state.vector_store.get_info()
            st.metric("Document Chunks", info['total_documents'])
    
    with col3:
        doc_answers = int(st.session_state.stats["document_answers"])
        ai_answers = int(st.session_state.stats["ai_answers"])
        st.metric("Document/AI Answers", f"{doc_answers}/{ai_answers}")

def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("""
        API Key Required
        
        Please create a .env file in your project folder with:
        OPENROUTER_API_KEY=your_key_here
        
        Get your free API key from: https://openrouter.ai/
        
        Then restart the application.
        """)
        st.stop()
    
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()