"""
Semantic Search Engine - Streamlit Application

A modular RAG (Retrieval Augmented Generation) application that enables semantic search
over PDF documents using ChromaDB vector store and OpenAI embeddings.

Features:
    - PDF document upload and processing
    - Text chunking with configurable parameters
    - Vector embeddings using OpenAI text-embedding-3-large
    - Persistent ChromaDB vector store
    - Question answering with GPT-4o-mini
    - Context display for transparency
    - Modular architecture with separation of concerns

Environment Variables Required:
    OPENAI_API_KEY: OpenAI API key for embeddings and chat

Usage:
    streamlit run app.py

Author: Harsh
Repository: https://github.com/shrimpy8/semantic-serach
"""

import streamlit as st
import logging
from config_loader import load_config
from core import DocumentProcessor, VectorStoreManager, QAChain
from utils import add_documents_with_retry, stream_llm_with_retry

# Load configuration
config = load_config()

# Configure structured logging
logging_config = config.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config["level"]),
    format=logging_config["format"],
    handlers=[
        logging.FileHandler(logging_config["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = False
if 'vector_store_manager' not in st.session_state:
    # Initialize vector store manager
    st.session_state.vector_store_manager = VectorStoreManager(
        embedding_model_name=config.get_embedding_model(),
        collection_name=config.get_collection_name(),
        persist_directory=config.get_persist_directory()
    )
if 'qa_chain' not in st.session_state:
    # Initialize QA chain
    retriever = st.session_state.vector_store_manager.get_retriever(
        search_type=config.get_search_type(),
        search_k=config.get_search_k()
    )
    st.session_state.qa_chain = QAChain(
        model_name=config.get_chat_model(),
        temperature=config.get_chat_temperature(),
        retriever=retriever,
        system_prompt=config.get_qa_system_prompt()
    )

# UI Setup
st.title("🔍 Semantic Search Engine")
st.header("Upload a PDF file to get started")

# Sidebar - Vector Store Management
st.sidebar.title("📚 Database Management")

# Display current database status
try:
    collection_count = st.session_state.vector_store_manager.get_collection_count()
    if collection_count > 0:
        st.sidebar.success(f"Database contains **{collection_count}** document chunks")
    else:
        st.sidebar.info("Database is empty. Upload a document to begin.")
except Exception as e:
    st.sidebar.warning(f"Could not check database status: {str(e)}")
    logger.error(f"Error checking database status: {e}")

# Clear vector store option
if st.sidebar.button("🗑️ Clear All Documents"):
    try:
        st.session_state.vector_store_manager.clear_collection()
        st.session_state.processed_file = False
        st.sidebar.success("✅ Database cleared successfully!")
        logger.info("Vector store cleared by user")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing database: {str(e)}")
        logger.error(f"Error clearing vector store: {e}", exc_info=True)

# Sidebar - Configuration Info
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.markdown(f"- **Embedding Model**: {config.get_embedding_model()}")
st.sidebar.markdown(f"- **Chat Model**: {config.get_chat_model()}")
st.sidebar.markdown(f"- **Chunk Size**: {config.get_chunk_size()}")
st.sidebar.markdown(f"- **Retrieval K**: {config.get_search_k()}")

# Main Content - File Upload
st.markdown("---")
uploaded_file = st.file_uploader(
    "📄 Select a PDF file",
    type=['pdf'],
    help="Upload a PDF document to enable semantic search"
)

if uploaded_file is not None:
    # Validate file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.error("❌ Only PDF files are currently supported.")
        logger.warning(f"Invalid file type uploaded: {uploaded_file.name}")
    else:
        with st.spinner("🔄 Processing document..."):
            try:
                # Initialize document processor
                doc_processor = DocumentProcessor(
                    chunk_size=config.get_chunk_size(),
                    chunk_overlap=config.get_chunk_overlap(),
                    add_start_index=config.get_add_start_index()
                )

                # Process the uploaded file
                st.info(f"📖 Processing: **{uploaded_file.name}**")
                chunks = doc_processor.process_uploaded_file(uploaded_file)
                st.success(f"✅ Document split into **{len(chunks)}** chunks")

                # Display chunk information
                with st.expander("📊 View chunk details", expanded=False):
                    chunk_info = doc_processor.get_chunk_info(chunks)
                    for info in chunk_info:
                        st.write(f"Chunk {info['index']}: {info['size']} characters")

                # Index embeddings with retry logic
                with st.spinner("🔗 Creating embeddings and indexing..."):
                    chroma_ids = add_documents_with_retry(
                        st.session_state.vector_store_manager.vector_store,
                        chunks
                    )
                    st.success(f"✅ Indexed **{len(chroma_ids)}** embeddings successfully!")

                # Update session state
                st.session_state.processed_file = True
                logger.info(f"File processing complete: {uploaded_file.name}")

            except ValueError as e:
                st.error(f"❌ Validation error: {str(e)}")
                logger.error(f"Validation error: {e}")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.error(f"Error processing file {uploaded_file.name}: {e}", exc_info=True)

# Question Answering Interface
st.markdown("---")
st.subheader("💬 Ask Questions About Your Document")

# Chat input
if prompt := st.chat_input("Type your question here..."):
    if not st.session_state.processed_file:
        st.error("⚠️ Please upload a document before asking questions.")
    else:
        # Display user question
        st.write(f"**❓ Question:** {prompt}")

        with st.spinner("🔍 Searching for answer..."):
            try:
                # Retrieve context
                docs_retrieved = st.session_state.qa_chain.retrieve_context(prompt)

                if not docs_retrieved:
                    logger.warning("No relevant information found for query")
                    st.warning("⚠️ No relevant information found. Try rephrasing your question.")
                else:
                    # Display retrieved context
                    with st.expander("📝 View context used for answering", expanded=False):
                        for i, doc in enumerate(docs_retrieved):
                            st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content}\n```")

                    # Format context
                    context = st.session_state.qa_chain.format_context(docs_retrieved)

                    # Generate and stream answer
                    st.subheader("✨ Answer:")
                    answer_placeholder = st.empty()

                    full_answer = ""
                    for chunk in stream_llm_with_retry(
                        st.session_state.qa_chain.llm_model,
                        st.session_state.qa_chain.prompt_template.invoke({
                            "question": prompt,
                            "document": context
                        })
                    ):
                        full_answer += chunk.content
                        answer_placeholder.write(full_answer)

                    logger.info(f"Answer generated: {len(full_answer)} characters")

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                logger.error(f"Error generating answer: {e}", exc_info=True)

# Help Section
with st.expander("ℹ️ How this app works", expanded=False):
    st.markdown("""
    ### RAG Pipeline Overview

    1. **📄 Upload PDF**: The app processes your PDF and splits it into smaller chunks.
    2. **🔗 Document Indexing**: Each chunk is converted into a vector embedding and stored in ChromaDB.
    3. **🔍 Similarity Search**: When you ask a question, the app finds the most relevant chunks.
    4. **✨ Answer Generation**: GPT-4o-mini generates an answer based only on the retrieved chunks.

    ### Tips for Better Results

    - ✅ Ask specific questions that might be answered in the document
    - ✅ If you get "I can't answer" responses, try rephrasing
    - ✅ For complex documents, clear the database before uploading new files
    - ✅ Check the context expander to see what information was used

    ### Technology Stack

    - **Embeddings**: OpenAI text-embedding-3-large
    - **Vector Store**: ChromaDB (persistent)
    - **LLM**: GPT-4o-mini
    - **Framework**: LangChain + Streamlit
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with LangChain, ChromaDB, and OpenAI | "
    f"<a href='https://github.com/shrimpy8/semantic-serach'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
