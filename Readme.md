# Semantic Search Engine

A Streamlit application that allows users to upload PDF documents, analyze them using vector embeddings, and ask natural language questions to retrieve contextually relevant information from the documents.

## Features

- **PDF Document Processing**: Upload and process PDF files
- **Semantic Chunking**: Split documents into manageable chunks for better analysis
- **Vector Embeddings**: Create and store vector representations of document chunks
- **Natural Language Queries**: Ask questions in plain language about your documents
- **Contextual Answers**: Get answers based specifically on the content of your documents
- **Real-time Processing**: Watch as your documents are processed and indexed
- **Context Transparency**: See exactly which parts of your document were used to generate answers

## Technologies Used

- **Streamlit**: Web interface framework
- **LangChain**: Document processing, vector storage, and prompt management
- **OpenAI Models**: LLM and embedding models for natural language processing
- **ChromaDB**: Vector database for semantic search functionality

## Requirements

- Python 3.8+
- OpenAI API key (for GPT-4o-mini and text-embedding-3-large models)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/shrimpy8/semantic-search.git
   cd semantic-search-engine
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a PDF document using the file uploader

4. Wait for the document to be processed and indexed

5. Enter your questions in the chat input at the bottom of the page

6. View the AI-generated answers based on the content of your document

## How It Works

1. **Document Processing**: When you upload a PDF, the application:
   - Creates a temporary copy of the file
   - Extracts text from all pages
   - Splits the text into manageable chunks

2. **Vector Embedding**: Each text chunk is converted into a vector representation using OpenAI's text-embedding-3-large model

3. **Database Storage**: The vectors are stored in a ChromaDB database for efficient semantic search

4. **Question Answering**: When you ask a question:
   - Your question is converted to a vector
   - The system finds the most similar chunks from your document
   - The relevant chunks are sent to GPT-4o-mini along with your question
   - The AI generates an answer based only on the provided context

## Tips for Best Results

- Ask specific questions that are likely to be answered in your document
- For large documents, consider how your question relates to the document content
- If you get "I can't answer that question" responses, try rephrasing your query
- Use the "View context used for answering" feature to see what information the AI is working with
- For new documents, use the "Clear existing documents" checkbox to remove previous document data

## Limitations

- Currently only supports PDF files
- Text extraction quality depends on the PDF structure and formatting
- Very large documents may take longer to process
- Questions must be answerable from the document content

