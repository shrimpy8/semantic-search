"""
Document Processor Module

Handles PDF document loading and text chunking for semantic search.
"""

import os
import uuid
import tempfile
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles PDF document loading and text chunking.

    This class manages the entire document processing pipeline including:
    - Temporary file handling for uploaded PDFs
    - PDF parsing and text extraction
    - Text chunking with configurable parameters

    Attributes:
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between consecutive chunks
        add_start_index: Whether to add start index metadata to chunks

    Example:
        >>> processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        >>> chunks = processor.process_uploaded_file(uploaded_file)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, add_start_index: bool = True):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks
            add_start_index: Whether to add start index metadata
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index
        )

        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Process an uploaded PDF file into text chunks.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            List of LangChain Document objects (chunks)

        Raises:
            ValueError: If file is not a PDF
            Exception: If PDF processing fails

        Example:
            >>> chunks = processor.process_uploaded_file(uploaded_file)
            >>> print(f"Created {len(chunks)} chunks")
        """
        if not uploaded_file.name.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")

        logger.info(f"Processing file: {uploaded_file.name}, size: {uploaded_file.size} bytes")

        # Create temporary file
        temp_file_path = self._create_temp_file(uploaded_file)

        try:
            # Load PDF
            docs = self._load_pdf(temp_file_path)
            logger.info(f"Loaded {len(docs)} pages from PDF")

            # Split into chunks
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Document split into {len(chunks)} chunks")

            # Log chunk statistics
            self._log_chunk_stats(chunks)

            return chunks

        finally:
            # Clean up temporary file
            self._cleanup_temp_file(temp_file_path)

    def _create_temp_file(self, uploaded_file) -> str:
        """
        Create a temporary file from uploaded file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Path to temporary file
        """
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Created temporary file: {temp_file_path}")
        return temp_file_path

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF document using PyPDFLoader.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects (one per page)

        Raises:
            Exception: If PDF loading fails
        """
        logger.info(f"Loading PDF from: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _log_chunk_stats(self, chunks: List[Document]) -> None:
        """
        Log statistics about document chunks.

        Args:
            chunks: List of document chunks
        """
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i}: {len(chunk.page_content)} characters")

        if chunks:
            avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_size:.0f} characters")

    def _cleanup_temp_file(self, file_path: str) -> None:
        """
        Remove temporary file.

        Args:
            file_path: Path to temporary file
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")

    def get_chunk_info(self, chunks: List[Document]) -> List[dict]:
        """
        Get information about each chunk.

        Args:
            chunks: List of document chunks

        Returns:
            List of dictionaries with chunk metadata

        Example:
            >>> info = processor.get_chunk_info(chunks)
            >>> for item in info:
            ...     print(f"Chunk {item['index']}: {item['size']} chars")
        """
        return [
            {
                "index": i + 1,
                "size": len(chunk.page_content),
                "metadata": chunk.metadata
            }
            for i, chunk in enumerate(chunks)
        ]
