"""
Semantic Search Engine - Streamlit Application

A modular RAG (Retrieval Augmented Generation) application that enables semantic search
over PDF documents using ChromaDB vector store and OpenAI embeddings.

Features:
    - PDF document upload and processing
    - Text chunking with configurable parameters
    - Vector embeddings using OpenAI text-embedding-3-large
    - Persistent ChromaDB vector store
    - Hybrid retrieval (BM25 + semantic search)
    - Re-ranking with Cohere/Jina
    - Conversation history with follow-up optimization
    - A/B testing framework for retrieval methods
    - Question answering with GPT-4o-mini
    - Context display for transparency

Environment Variables Required:
    OPENAI_API_KEY: OpenAI API key for embeddings and chat
    COHERE_API_KEY: (Optional) Cohere API key for re-ranking

Usage:
    streamlit run app.py

Author: Harsh
Repository: https://github.com/shrimpy8/semantic-serach
"""

import time
import streamlit as st
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config_loader import load_config
from core import (
    DocumentProcessor,
    VectorStoreManager,
    QAChain,
    HybridRetriever,
    RetrievalMethod,
    create_hybrid_retriever,
    ConversationManager,
    ABTestingManager,
    TestVariant
)
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


def initialize_session_state():
    """Initialize all session state variables."""
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = False

    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            embedding_model_name=config.get_embedding_model(),
            collection_name=config.get_collection_name(),
            persist_directory=config.get_persist_directory(),
            use_docker=config.use_chroma_docker(),
            chroma_host=config.get_chroma_host(),
            chroma_port=config.get_chroma_port()
        )

    if 'qa_chain' not in st.session_state:
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

    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None

    if 'documents' not in st.session_state:
        st.session_state.documents = []

    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(
            storage_dir=config.get_conversation_storage_dir(),
            max_history=config.get_max_conversation_history()
        )

    if 'ab_testing_manager' not in st.session_state:
        st.session_state.ab_testing_manager = ABTestingManager(
            storage_dir=config.get_ab_testing_storage_dir()
        )

    if 'current_retrieval_method' not in st.session_state:
        st.session_state.current_retrieval_method = config.get_default_retrieval_method()


def render_sidebar():
    """Render sidebar with configuration and management options."""
    st.sidebar.title("Semantic Search Engine")

    # Retrieval Settings Section (moved to top)
    st.sidebar.markdown("### Retrieval Settings")

    # Retrieval Profile (Presets)
    presets = config.get_retrieval_presets()
    preset_names = list(presets.keys()) + ["custom"]

    # Initialize preset in session state
    if 'current_preset' not in st.session_state:
        st.session_state.current_preset = config.get_default_preset()

    # Format function for preset display
    def format_preset(preset_name: str) -> str:
        if preset_name == "custom":
            return "⚙️ Custom"
        preset = presets.get(preset_name, {})
        icon = preset.get("icon", "")
        display = preset.get("display_name", preset_name)
        return f"{icon} {display}"

    selected_preset = st.sidebar.selectbox(
        "Retrieval Profile",
        options=preset_names,
        index=preset_names.index(st.session_state.current_preset) if st.session_state.current_preset in preset_names else 0,
        format_func=format_preset,
        help="Pre-configured retrieval settings for different use cases"
    )
    st.session_state.current_preset = selected_preset

    # Apply preset values or show custom controls
    if selected_preset != "custom":
        preset = presets[selected_preset]

        # Show preset description
        st.sidebar.info(f"{preset.get('icon', '')} {preset.get('description', '')}")

        # Apply preset values to session state
        st.session_state.search_k = preset.get("k", 5)
        st.session_state.hybrid_alpha = preset.get("alpha", 0.5)
        st.session_state.use_reranking = preset.get("rerank", True)
        st.session_state.current_retrieval_method = preset.get("method", "hybrid")

        # Show current settings (read-only)
        with st.sidebar.expander("Preset Settings", expanded=False):
            st.markdown(f"**Results**: {st.session_state.search_k}")
            st.markdown(f"**Alpha**: {st.session_state.hybrid_alpha}")
            st.markdown(f"**Re-ranking**: {'On' if st.session_state.use_reranking else 'Off'}")
            st.markdown(f"**Method**: {st.session_state.current_retrieval_method}")

    else:
        # Custom mode - show all controls
        retrieval_methods = {
            "Semantic Only": "semantic",
            "BM25 Only": "bm25",
            "Hybrid (BM25 + Semantic)": "hybrid"
        }

        selected_method = st.sidebar.selectbox(
            "Retrieval Method",
            options=list(retrieval_methods.keys()),
            index=list(retrieval_methods.values()).index(
                st.session_state.current_retrieval_method
            ) if st.session_state.current_retrieval_method in retrieval_methods.values() else 2,
            help="Choose retrieval strategy"
        )
        st.session_state.current_retrieval_method = retrieval_methods[selected_method]

        # Alpha slider for hybrid mode
        if st.session_state.current_retrieval_method == "hybrid":
            st.session_state.hybrid_alpha = st.sidebar.slider(
                "Semantic Weight (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=getattr(st.session_state, 'hybrid_alpha', config.get_hybrid_alpha()),
                step=0.1,
                help="0 = BM25 only, 1 = Semantic only"
            )

        # Re-ranking toggle
        st.session_state.use_reranking = st.sidebar.checkbox(
            "Enable Re-ranking",
            value=getattr(st.session_state, 'use_reranking', config.is_reranking_enabled()),
            help="Apply cross-encoder re-ranking for better accuracy"
        )

        # Number of results
        st.session_state.search_k = st.sidebar.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=getattr(st.session_state, 'search_k', config.get_search_k()),
            help="Number of document chunks to retrieve"
        )

    st.sidebar.markdown("---")

    # Configuration Display
    with st.sidebar.expander("Configuration", expanded=False):
        st.markdown(f"**Embedding**: {config.get_embedding_model()}")
        st.markdown(f"**Chat Model**: {config.get_chat_model()}")
        st.markdown(f"**Chunk Size**: {config.get_chunk_size()}")
        st.markdown(f"**Re-ranker**: {config.get_reranker_provider()}")


@st.dialog("Clear All Documents")
def confirm_clear_documents():
    """Confirmation dialog for clearing all documents."""
    st.warning("⚠️ This will permanently delete all indexed documents from the database.")
    st.markdown("Are you sure you want to continue?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Clear All", type="primary", use_container_width=True):
            try:
                st.session_state.vector_store_manager.clear_collection()
                st.session_state.processed_file = False
                st.session_state.documents = []
                st.session_state.hybrid_retriever = None
                logger.info("Vector store cleared by user")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
                logger.error(f"Error clearing vector store: {e}", exc_info=True)
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_database_management():
    """Render database management section at bottom of sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Management")

    try:
        collection_count = st.session_state.vector_store_manager.get_collection_count()
        if collection_count > 0:
            st.sidebar.success(f"**{collection_count}** document chunks indexed")
        else:
            st.sidebar.info("Database is empty. Upload a document to begin.")
    except Exception as e:
        st.sidebar.warning(f"Could not check database status: {str(e)}")
        logger.error(f"Error checking database status: {e}")

    if st.sidebar.button("Clear All Documents"):
        confirm_clear_documents()


@st.dialog("Clear Conversation History")
def confirm_clear_history():
    """Confirmation dialog for clearing conversation history."""
    st.warning("⚠️ This will delete the current conversation history.")
    st.markdown("Are you sure you want to continue?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Clear", type="primary", use_container_width=True):
            if st.session_state.conversation_manager.current_session:
                st.session_state.conversation_manager.delete_session(
                    st.session_state.conversation_manager.current_session.session_id
                )
                st.session_state.conversation_manager.start_session()
                logger.info("Conversation history cleared by user")
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_conversation_history():
    """Render conversation history panel."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Conversation History")

    if st.session_state.conversation_manager.current_session:
        history = st.session_state.conversation_manager.get_query_history(n=5)
        if history:
            with st.sidebar.expander(f"Recent Queries ({len(history)})", expanded=False):
                for i, item in enumerate(reversed(history)):
                    st.markdown(f"**Q{len(history)-i}:** {item['query'][:50]}...")
                    st.caption(f"Method: {item['retrieval_method']}")
                    st.markdown("---")

    # Session management
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New Session"):
            st.session_state.conversation_manager.start_session(
                document_name=getattr(st.session_state, 'current_doc_name', None)
            )
            st.rerun()
    with col2:
        if st.button("Clear History"):
            confirm_clear_history()


def render_ab_testing_panel():
    """Render A/B testing panel."""
    if not config.is_ab_testing_enabled():
        return

    with st.expander("A/B Testing", expanded=False):
        st.markdown("### Compare Retrieval Methods")

        if st.session_state.processed_file and st.session_state.hybrid_retriever:
            test_query = st.text_input(
                "Test Query",
                placeholder="Enter a query to compare methods...",
                key="ab_test_query"
            )

            if st.button("Run Comparison") and test_query:
                with st.spinner("Running A/B comparison..."):
                    run_ab_comparison(test_query)

            # Show comparison results
            if st.session_state.ab_testing_manager.current_experiment:
                summary = st.session_state.ab_testing_manager.get_comparison_summary()
                if summary.get("total_tests", 0) > 0:
                    st.markdown("### Results")

                    # Create comparison table
                    cols = st.columns(len(summary.get("variants", {})))
                    for i, (variant, stats) in enumerate(summary.get("variants", {}).items()):
                        if stats:
                            with cols[i]:
                                st.metric(
                                    label=variant.upper(),
                                    value=f"{stats.get('avg_score', {}).get('mean', 0):.3f}",
                                    delta=f"{stats.get('latency', {}).get('mean', 0):.0f}ms"
                                )

                    if summary.get("recommendation", {}).get("best_variant"):
                        st.success(
                            f"Recommended: **{summary['recommendation']['best_variant']}** "
                            f"(avg score: {summary['recommendation']['best_avg_score']:.3f})"
                        )

                    # Export button
                    if st.button("Export Results (CSV)"):
                        csv_data = st.session_state.ab_testing_manager.export_results("csv")
                        if csv_data:
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name="ab_test_results.csv",
                                mime="text/csv"
                            )
        else:
            st.info("Upload a document to enable A/B testing.")


def run_ab_comparison(query: str):
    """Run A/B comparison test for a query."""
    if not st.session_state.ab_testing_manager.current_experiment:
        st.session_state.ab_testing_manager.create_experiment(
            name=f"Comparison - {query[:30]}",
            description="Automated A/B comparison"
        )

    def retriever_func(q, method, k):
        method_map = {
            "semantic": RetrievalMethod.SEMANTIC,
            "bm25": RetrievalMethod.BM25,
            "hybrid": RetrievalMethod.HYBRID,
            "hybrid_rerank": RetrievalMethod.HYBRID
        }
        use_rerank = method == "hybrid_rerank"
        return st.session_state.hybrid_retriever.retrieve(
            q, k=k,
            method=method_map.get(method, RetrievalMethod.HYBRID),
            use_reranker=use_rerank
        )

    variants = [TestVariant.CONTROL, TestVariant.VARIANT_A, TestVariant.VARIANT_B]
    if st.session_state.hybrid_retriever.reranker:
        variants.append(TestVariant.VARIANT_C)

    st.session_state.ab_testing_manager.run_comparison(
        query=query,
        retriever_func=retriever_func,
        variants=variants,
        k=st.session_state.search_k
    )


def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file."""
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.error("Only PDF files are currently supported.")
        logger.warning(f"Invalid file type uploaded: {uploaded_file.name}")
        return

    with st.spinner("Processing document..."):
        try:
            # Initialize document processor
            doc_processor = DocumentProcessor(
                chunk_size=config.get_chunk_size(),
                chunk_overlap=config.get_chunk_overlap(),
                add_start_index=config.get_add_start_index()
            )

            # Process the uploaded file
            st.info(f"Processing: **{uploaded_file.name}**")
            chunks = doc_processor.process_uploaded_file(uploaded_file)

            # Store documents for BM25
            st.session_state.documents = chunks
            st.session_state.current_doc_name = uploaded_file.name

            # Index embeddings with retry logic
            with st.spinner("Creating embeddings and indexing..."):
                chroma_ids = add_documents_with_retry(
                    st.session_state.vector_store_manager.vector_store,
                    chunks
                )

            # Display chunk information in three-column layout
            chunk_info = doc_processor.get_chunk_info(chunks)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"**{len(chunks)}** chunks created")

            with col2:
                with st.expander("View chunk details", expanded=False):
                    for info in chunk_info[:5]:  # Show first 5
                        st.write(f"Chunk {info['index']}: {info['size']} chars")
                    if len(chunk_info) > 5:
                        st.write(f"... and {len(chunk_info) - 5} more")

            with col3:
                st.success(f"**{len(chroma_ids)}** embeddings indexed")

            # Initialize hybrid retriever
            semantic_retriever = st.session_state.vector_store_manager.get_retriever(
                search_k=st.session_state.search_k * config.get_fetch_k_multiplier()
            )

            st.session_state.hybrid_retriever = create_hybrid_retriever(
                semantic_retriever=semantic_retriever,
                documents=chunks,
                enable_reranker=config.is_reranking_enabled(),
                reranker_provider=config.get_reranker_provider(),
                alpha=config.get_hybrid_alpha(),
                rrf_k=config.get_rrf_k(),
                bm25_k1=config.get_bm25_k1(),
                bm25_b=config.get_bm25_b()
            )

            # Start conversation session
            st.session_state.conversation_manager.start_session(
                document_name=uploaded_file.name
            )

            # Update session state
            st.session_state.processed_file = True
            logger.info(f"File processing complete: {uploaded_file.name}")

        except ValueError as e:
            st.error(f"Validation error: {str(e)}")
            logger.error(f"Validation error: {e}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error processing file {uploaded_file.name}: {e}", exc_info=True)


def handle_question(prompt: str):
    """Handle a user question and generate response."""
    if not st.session_state.processed_file:
        st.error("Please upload a document before asking questions.")
        return

    # Display user question
    st.markdown(f"**Question:** {prompt}")

    # Optimize query for follow-ups if enabled
    optimized_query = prompt
    if config.is_follow_up_optimization_enabled():
        optimized_query = st.session_state.conversation_manager.optimize_follow_up_query(
            prompt,
            include_context=True
        )

    with st.spinner("Searching for answer..."):
        try:
            start_time = time.perf_counter()

            # Use hybrid retriever if available, otherwise fall back to QA chain
            if st.session_state.hybrid_retriever:
                method_map = {
                    "semantic": RetrievalMethod.SEMANTIC,
                    "bm25": RetrievalMethod.BM25,
                    "hybrid": RetrievalMethod.HYBRID
                }
                method = method_map.get(
                    st.session_state.current_retrieval_method,
                    RetrievalMethod.HYBRID
                )

                # Get alpha from session state for hybrid mode
                alpha = getattr(st.session_state, 'hybrid_alpha', config.get_hybrid_alpha())

                results = st.session_state.hybrid_retriever.retrieve(
                    optimized_query,
                    k=st.session_state.search_k,
                    method=method,
                    alpha=alpha,
                    use_reranker=st.session_state.use_reranking
                )

                retrieval_time = (time.perf_counter() - start_time) * 1000

                if not results:
                    logger.warning("No relevant information found for query")
                    st.warning("No relevant information found. Try rephrasing your question.")
                    return

                docs_retrieved = [r.document for r in results]
                scores = [r.final_score for r in results]

                # Display retrieved context with detailed scores
                with st.expander("Context used for answering", expanded=False):
                    # Summary header
                    preset_info = f"Profile: **{st.session_state.current_preset}**" if st.session_state.current_preset != "custom" else "Profile: **Custom**"
                    st.caption(f"{preset_info} | Method: **{method.value}** | Time: {retrieval_time:.0f}ms")

                    for i, result in enumerate(results):
                        # Create score breakdown
                        score_parts = []
                        if result.semantic_score is not None:
                            score_parts.append(f"Semantic: {result.semantic_score:.3f}")
                        if result.bm25_score is not None:
                            score_parts.append(f"BM25: {result.bm25_score:.3f}")
                        if result.rerank_score is not None:
                            score_parts.append(f"Rerank: {result.rerank_score:.3f}")

                        # Display chunk with score breakdown
                        st.markdown(f"**Chunk {i+1}** (Final: {result.final_score:.4f})")

                        # Show score breakdown in a compact format
                        if score_parts:
                            st.caption(" | ".join(score_parts))

                        # Show content
                        st.code(result.document.page_content, language=None)

            else:
                # Fall back to original QA chain
                docs_retrieved = st.session_state.qa_chain.retrieve_context(prompt)
                scores = [1.0 / (i + 1) for i in range(len(docs_retrieved))]
                retrieval_time = 0

                if not docs_retrieved:
                    logger.warning("No relevant information found for query")
                    st.warning("No relevant information found. Try rephrasing your question.")
                    return

                with st.expander("Context used for answering", expanded=False):
                    for i, doc in enumerate(docs_retrieved):
                        st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content}\n```")

            # Format context
            context = st.session_state.qa_chain.format_context(docs_retrieved)

            # Generate and stream answer
            st.subheader("Answer:")
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

            # Record in conversation history
            st.session_state.conversation_manager.add_query(
                query=prompt,
                answer=full_answer,
                retrieved_docs=docs_retrieved,
                scores=scores,
                retrieval_method=st.session_state.current_retrieval_method
            )

            logger.info(f"Answer generated: {len(full_answer)} characters")

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            logger.error(f"Error generating answer: {e}", exc_info=True)


def render_help_section():
    """Render help section with 2-column layout and learn more link."""
    with st.expander("How this app works", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### RAG Pipeline Overview

            1. **Upload PDF**: The app processes your PDF and splits it into smaller chunks.
            2. **Document Indexing**: Each chunk is converted into a vector embedding and stored in ChromaDB.
            3. **Hybrid Search**: When you ask a question, the app finds relevant chunks using:
               - **Semantic search**: Finds conceptually similar content
               - **BM25 (keyword)**: Finds exact term matches
               - **Re-ranking**: Optionally re-scores results for better accuracy
            4. **Answer Generation**: GPT-4o-mini generates an answer based only on the retrieved chunks.

            ---

            ### Retrieval Methods

            - **Semantic Only**: Uses embedding similarity (best for conceptual questions)
            - **BM25 Only**: Uses keyword matching (best for exact terms)
            - **Hybrid**: Combines both methods (recommended for general use)
            """)

        with col2:
            st.markdown("""
            ### Tips for Better Results

            - Ask specific questions that might be answered in the document
            - If you get "I can't answer" responses, try rephrasing
            - Use the alpha slider to tune the balance between semantic and keyword search
            - Check the context expander to see what information was used
            - Follow-up questions automatically use conversation context

            ---

            ### Technology Stack

            - **Embeddings**: OpenAI text-embedding-3-large
            - **Vector Store**: ChromaDB (persistent)
            - **LLM**: GPT-4o-mini
            - **Re-ranking**: Cohere / Jina (optional)
            - **Framework**: LangChain + Streamlit
            """)

        # Learn more link
        st.markdown("---")
        st.page_link(
            "pages/1_How_It_Works.py",
            label="📚 Learn more about optimizing your semantic search",
            icon="🔗"
        )


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Page configuration
    st.set_page_config(
        page_title="Semantic Search Engine",
        page_icon="magnifying_glass_tilted_left:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Hide the default page navigation in sidebar
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Custom CSS for larger font sizes
    st.markdown("""
        <style>
        /* Increase base font size */
        html, body, [class*="css"] {
            font-size: 18px;
        }

        /* Main content text */
        .stMarkdown, .stText, p, li {
            font-size: 18px !important;
        }

        /* Headers */
        h1 {
            font-size: 2.5rem !important;
        }
        h2 {
            font-size: 2rem !important;
        }
        h3 {
            font-size: 1.6rem !important;
        }

        /* Sidebar text */
        .css-1d391kg, [data-testid="stSidebar"] {
            font-size: 17px !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            font-size: 17px !important;
        }

        /* Input fields and buttons */
        .stTextInput input, .stTextArea textarea {
            font-size: 17px !important;
        }
        .stButton button {
            font-size: 17px !important;
        }

        /* Selectbox and other widgets */
        .stSelectbox, .stMultiSelect, .stSlider {
            font-size: 17px !important;
        }

        /* Expander text */
        .streamlit-expanderHeader {
            font-size: 18px !important;
        }

        /* Code blocks */
        code, .stCode {
            font-size: 15px !important;
        }

        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            font-size: 17px !important;
        }

        /* Table text */
        .stDataFrame, .stTable {
            font-size: 16px !important;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            font-size: 17px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()
    render_conversation_history()
    render_database_management()

    # Main content
    st.title("Semantic Search Engine")
    st.markdown("Upload a PDF document and ask questions using hybrid semantic search.")

    # Help Section (moved to top for visibility)
    render_help_section()

    # Always visible "Learn more" link outside the expander
    st.page_link(
        "pages/1_How_It_Works.py",
        label="📚 Learn more about optimizing your semantic search →",
        icon=None
    )

    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Select a PDF file",
        type=['pdf'],
        help="Upload a PDF document to enable semantic search"
    )

    if uploaded_file is not None:
        # Check if file already exists in database
        if st.session_state.vector_store_manager.document_exists(uploaded_file.name):
            st.warning(
                f"⚠️ **'{uploaded_file.name}'** already exists in the database. "
                "You can clear the database from the sidebar to re-index, or upload a different file."
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Process Anyway", type="secondary"):
                    process_uploaded_file(uploaded_file)
        else:
            process_uploaded_file(uploaded_file)

    # A/B Testing Panel
    render_ab_testing_panel()

    # Question Answering Interface
    st.markdown("---")
    st.subheader("Ask Questions About Your Document")

    # Inline question input (directly under header)
    col1, col2 = st.columns([5, 1])
    with col1:
        question_input = st.text_input(
            "Your question",
            placeholder="Type your question here...",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    if ask_button and question_input:
        handle_question(question_input)
    elif ask_button and not question_input:
        st.warning("Please enter a question.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with LangChain, ChromaDB, and OpenAI | "
        f"<a href='https://github.com/shrimpy8/semantic-serach'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
