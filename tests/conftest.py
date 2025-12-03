"""
Pytest configuration and shared fixtures for semantic search tests.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-456")


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first chunk of text about machine learning.",
            metadata={"page": 1, "source": "test.pdf"}
        ),
        Document(
            page_content="This is the second chunk discussing artificial intelligence.",
            metadata={"page": 1, "source": "test.pdf"}
        ),
        Document(
            page_content="This is the third chunk about deep learning and neural networks.",
            metadata={"page": 2, "source": "test.pdf"}
        )
    ]


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_content = """
models:
  embedding:
    name: "text-embedding-3-small"
    provider: "openai"
  chat:
    name: "gpt-3.5-turbo"
    provider: "openai"
    temperature: 0.0

document_processing:
  chunk_size: 500
  chunk_overlap: 100
  add_start_index: true

vector_store:
  provider: "chroma"
  collection_name: "test_collection"
  persist_directory: "./test_chroma_db"
  search_type: "similarity"
  search_k: 2

retry:
  max_attempts: 2
  min_wait: 1
  max_wait: 5
  multiplier: 1

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "test_semantic_search.log"

prompts:
  qa_system: "Answer the question {question} using {document}."
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_uploaded_file():
    """Create a mock Streamlit uploaded file."""
    mock_file = Mock()
    mock_file.name = "test_document.pdf"
    mock_file.size = 1024 * 50  # 50KB
    mock_file.getbuffer.return_value = b"Mock PDF content"
    return mock_file


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for testing without API calls."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 1536 for _ in range(3)]
    mock.embed_query.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI for testing without API calls."""
    mock = MagicMock()

    # Mock invoke response
    mock_response = Mock()
    mock_response.content = "This is a test answer based on the provided context."
    mock.invoke.return_value = mock_response

    # Mock stream response
    mock_chunk_1 = Mock()
    mock_chunk_1.content = "This is "
    mock_chunk_2 = Mock()
    mock_chunk_2.content = "a test answer."
    mock.stream.return_value = [mock_chunk_1, mock_chunk_2]

    return mock


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield

    # Cleanup test directories
    test_dirs = ["./test_chroma_db", "./chroma_test"]
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)

    # Cleanup test log files
    test_logs = ["test_semantic_search.log", "semantic_search.log"]
    for log_file in test_logs:
        if os.path.exists(log_file):
            os.remove(log_file)
