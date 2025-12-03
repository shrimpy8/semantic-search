# 🔍 Semantic Search Engine

A production-ready RAG (Retrieval Augmented Generation) application built with LangChain, ChromaDB, and Streamlit. Upload PDF documents and ask natural language questions to retrieve contextually relevant answers powered by OpenAI.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **📄 PDF Document Processing**: Upload and parse PDF files with automatic text extraction
- **🔗 Semantic Chunking**: Intelligent text splitting with configurable chunk sizes and overlap
- **🎯 Vector Embeddings**: High-quality embeddings using OpenAI text-embedding-3-large
- **💾 Persistent Vector Store**: ChromaDB for efficient similarity search with disk persistence
- **💬 Natural Language Queries**: Ask questions in plain language about your documents
- **✨ Contextual Answers**: GPT-4o-mini generates answers based solely on document content
- **🔄 Real-time Streaming**: Token-by-token answer generation for responsive UX
- **📊 Context Transparency**: View exactly which document chunks were used for answers
- **🛡️ Robust Error Handling**: Automatic retry logic with exponential backoff for API resilience
- **📝 Structured Logging**: Comprehensive logging for debugging and monitoring
- **⚙️ YAML Configuration**: Easily customizable settings without code changes
- **🎨 Modular Architecture**: Clean separation of concerns for maintainability

## 🏗️ Architecture

### Project Structure

```
semantic-search/
├── app.py                      # Streamlit UI application
├── config.yaml                 # Centralized configuration
├── config_loader.py            # Configuration management
├── core/                       # Core business logic
│   ├── __init__.py
│   ├── document_processor.py  # PDF loading and chunking
│   ├── vector_store.py        # ChromaDB management
│   └── qa_chain.py            # Question answering pipeline
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── retry_utils.py         # API retry decorators
├── tests/                      # Test suite (41 passing tests)
│   ├── __init__.py
│   ├── conftest.py            # pytest fixtures
│   ├── test_config_loader.py  # Configuration tests
│   ├── test_document_processor.py  # Document processing tests
│   ├── test_vector_store.py   # Vector store tests
│   ├── test_qa_chain.py       # QA chain tests
│   └── test_retry_utils.py    # Retry logic tests
├── pytest.ini                  # pytest configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### RAG Pipeline

```
┌──────────────┐
│  PDF Upload  │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Document        │
│ Processor       │ (PDF → Pages → Chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Store    │
│ Manager         │ (Chunks → Embeddings → ChromaDB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ User Question   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ QA Chain        │ (Retrieve → Format → Generate)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamed Answer │
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 2GB+ free disk space (for ChromaDB vector store)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shrimpy8/semantic-serach.git
cd semantic-serach
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

**⚠️ SECURITY NOTE**: Never commit your `.env` file to version control. It's already in `.gitignore`.

## 💡 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Using the Semantic Search Engine

1. **Upload PDF**: Click "Select a PDF file" and choose your document
2. **Wait for Processing**: The app will extract text, create chunks, and generate embeddings
3. **View Database Status**: Check the sidebar to see how many chunks are indexed
4. **Ask Questions**: Type your question in the chat input at the bottom
5. **View Context**: Expand "View context used for answering" to see retrieved chunks
6. **Clear Database**: Use the sidebar button to remove all documents and start fresh

## ⚙️ Configuration

All settings are managed in `config.yaml`. Customize without changing code:

### Model Configuration

```yaml
models:
  embedding:
    name: "text-embedding-3-large"  # OpenAI embedding model
  chat:
    name: "gpt-4o-mini"              # Chat model
    temperature: 0.0                 # 0.0 = deterministic
```

### Document Processing

```yaml
document_processing:
  chunk_size: 1000        # Characters per chunk
  chunk_overlap: 200      # Overlap between chunks
  add_start_index: true   # Add metadata for chunk position
```

### Vector Store Settings

```yaml
vector_store:
  collection_name: "semantic_search_docs"
  persist_directory: "./chroma/db"
  search_k: 3             # Number of chunks to retrieve
```

### Retry Configuration

```yaml
retry:
  max_attempts: 3         # Retry attempts for API calls
  min_wait: 2             # Minimum wait time (seconds)
  max_wait: 10            # Maximum wait time (seconds)
```

## 🛠️ Development

### Module Overview

#### `core/document_processor.py`
- PDF loading with PyPDFLoader
- Text chunking with RecursiveCharacterTextSplitter
- Temporary file management
- Chunk statistics and metadata

#### `core/vector_store.py`
- ChromaDB initialization and management
- Document embedding and indexing
- Similarity search and retrieval
- Collection management (clear, count)

#### `core/qa_chain.py`
- Retrieval-augmented generation pipeline
- Context formatting
- Prompt template management
- Streaming answer generation

#### `utils/retry_utils.py`
- Exponential backoff retry decorators
- Handles `RateLimitError`, `APIConnectionError`, `APIError`
- Configurable retry parameters

### Adding New Features

1. **New Embedding Model**: Update `config.yaml` models section
2. **Different Chunk Size**: Modify `document_processing` in config
3. **Custom System Prompt**: Edit `prompts.qa_system` in config
4. **New Document Processor**: Extend `DocumentProcessor` class
5. **UI Changes**: Modify `app.py`

### Code Style

- Follow PEP 8 guidelines
- Comprehensive docstrings for all classes and functions
- Type hints for function parameters and returns
- Structured logging throughout

### Testing

The project includes a comprehensive test suite with 41 passing tests covering all core functionality:

**Run all tests**:
```bash
pytest tests/ -v
```

**Run specific test categories**:
```bash
# Unit tests only
pytest tests/ -v -m unit

# Integration tests only
pytest tests/ -v -m integration

# Slow tests (marked separately)
pytest tests/ -v -m slow
```

**Test coverage**:
```bash
pytest tests/ --cov=. --cov-report=html
```

**Test Structure**:
- `tests/test_config_loader.py` - Configuration loading and validation
- `tests/test_document_processor.py` - PDF processing and chunking
- `tests/test_vector_store.py` - ChromaDB operations
- `tests/test_qa_chain.py` - Question answering pipeline
- `tests/test_retry_utils.py` - Retry logic and error handling
- `tests/conftest.py` - Shared fixtures and mocks

**Test Statistics**:
- Total tests: 41
- Test duration: <5 seconds
- All tests use mocking for external dependencies (no API calls)

## 📝 Logging

Application logs are written to:
- **Console**: Real-time output during execution
- **File**: `semantic_search.log` (persistent logs)

Log format:
```
2025-11-30 15:30:45,123 - __main__ - INFO - Processing file: document.pdf, size: 1048576 bytes
```

Adjust logging level in `config.yaml`:
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 🔒 Security

### API Key Management

1. **Never commit** `.env` files to version control
2. **Rotate keys immediately** if accidentally exposed
3. **Use environment variables** for all sensitive data
4. **Review** `SECURITY_NOTICE.md` for detailed security guidelines

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
if git diff --cached --name-only | grep -E '\\.env$'; then
    echo "❌ ERROR: Attempting to commit .env file!"
    exit 1
fi
exit 0
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## 🐛 Troubleshooting

### Common Issues

**Error: "Configuration file not found"**
- Ensure `config.yaml` exists in the project directory
- Check file permissions

**Error: "OpenAI API key not found"**
- Verify `.env` file exists with valid `OPENAI_API_KEY`
- Ensure `python-dotenv` is installed

**Error: "Rate limit exceeded"**
- Wait for rate limit reset
- Retry logic should handle this automatically
- Check API quota/billing

**ChromaDB Permission Error**
- Ensure write permissions for `./chroma/db` directory
- Try deleting the directory and restarting

**Slow Embedding Generation**
- Large documents take time to process
- Consider reducing chunk_size in config
- Monitor OpenAI API usage dashboard

## 📚 How It Works

### 1. Document Upload & Processing

When you upload a PDF:
1. File is temporarily saved to disk
2. PyPDFLoader extracts text from all pages
3. RecursiveCharacterTextSplitter creates overlapping chunks
4. Temporary file is cleaned up

### 2. Embedding & Indexing

For each chunk:
1. OpenAI text-embedding-3-large generates 1536-dimension vector
2. Vector + metadata stored in ChromaDB
3. Collection persisted to disk for future sessions

### 3. Question Answering

When you ask a question:
1. Question is embedded using same model
2. ChromaDB performs similarity search (cosine similarity)
3. Top-K most relevant chunks retrieved
4. Chunks + question formatted into prompt
5. GPT-4o-mini generates answer based on context
6. Answer streamed token-by-token to UI

## 💡 Tips for Better Results

- ✅ Ask specific questions that can be answered from the document
- ✅ Upload well-structured PDFs with clear text (avoid scanned images)
- ✅ For multi-topic documents, ask focused questions
- ✅ Use the context viewer to understand what information was retrieved
- ✅ Clear the database between different documents to avoid confusion
- ✅ Rephrase questions if you get "I can't answer" responses
- ✅ Check chunk size if answers seem incomplete or too broad

## 🚧 Limitations

- **PDF Only**: Currently supports PDF files only (no DOCX, TXT, etc.)
- **Text Extraction**: Quality depends on PDF structure (scanned PDFs won't work)
- **Context Window**: Limited to top-K chunks (may miss relevant information)
- **English Focused**: Best performance with English documents
- **Memory Storage**: Vector store persisted locally (not suitable for multi-user production)

## 🗺️ Roadmap

**Completed** ✅:
- [x] Unit and integration tests (pytest) - 41 passing tests

**Future enhancements planned**:
- [ ] Support for multiple file formats (DOCX, TXT, HTML, Markdown)
- [ ] Multi-document collections with isolated searches
- [ ] Hybrid search (BM25 + semantic) with re-ranking
- [ ] Document similarity and comparison features
- [ ] Export search results to CSV/JSON
- [ ] User authentication and multi-user support
- [ ] Cloud vector store integration (Pinecone, Weaviate)
- [ ] Advanced chunking strategies (sentence-based, semantic)
- [ ] Citation tracking (show exact page/paragraph references)
- [ ] Conversation history and follow-up questions
- [ ] Docker containerization
- [ ] API endpoint for programmatic access

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenAI](https://openai.com/) - Embeddings and chat models

## 👤 Author

**Harsh**
- GitHub: [@shrimpy8](https://github.com/shrimpy8)
- Repository: [semantic-serach](https://github.com/shrimpy8/semantic-serach)

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review existing [GitHub Issues](https://github.com/shrimpy8/semantic-serach/issues)
3. Create a new issue with detailed description

---

**Made with ❤️ using LangChain, ChromaDB, and OpenAI**
