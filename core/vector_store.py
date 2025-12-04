"""
Vector Store Manager Module

Handles ChromaDB vector store operations including document indexing and retrieval.
Supports both local persistent storage and remote Docker/HTTP server modes.
"""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.

    This class handles all interactions with the ChromaDB vector store including:
    - Vector store initialization and persistence
    - Document embedding and indexing
    - Similarity search and retrieval
    - Collection management (clear, delete)

    Supports two modes:
    - Local: Uses persistent directory storage (default)
    - Docker/HTTP: Connects to ChromaDB server via HTTP

    Attributes:
        embedding_model: OpenAI embeddings model
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory for persistent storage (local mode)
        chroma_host: ChromaDB server host (Docker mode)
        chroma_port: ChromaDB server port (Docker mode)
        vector_store: ChromaDB vector store instance

    Example:
        >>> # Local mode
        >>> manager = VectorStoreManager(
        ...     embedding_model_name="text-embedding-3-large",
        ...     collection_name="my_docs"
        ... )
        >>> # Docker mode
        >>> manager = VectorStoreManager(
        ...     embedding_model_name="text-embedding-3-large",
        ...     use_docker=True,
        ...     chroma_host="localhost",
        ...     chroma_port=8000
        ... )
        >>> ids = manager.add_documents(chunks)
        >>> retriever = manager.get_retriever(search_k=3)
    """

    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-large",
        collection_name: str = "semantic_search_docs",
        persist_directory: str = "./chroma/db",
        use_docker: bool = False,
        chroma_host: str = "localhost",
        chroma_port: int = 8000
    ):
        """
        Initialize the vector store manager.

        Args:
            embedding_model_name: Name of OpenAI embedding model
            collection_name: Name for ChromaDB collection
            persist_directory: Directory path for persistence (local mode)
            use_docker: If True, connect to ChromaDB Docker server
            chroma_host: ChromaDB server hostname (Docker mode)
            chroma_port: ChromaDB server port (Docker mode)
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_docker = use_docker
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self._chroma_client = None

        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        logger.info(f"Initialized OpenAI embeddings: {embedding_model_name}")

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        mode = "Docker" if use_docker else "Local"
        logger.info(f"Vector store initialized ({mode} mode): collection={collection_name}")

    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize ChromaDB vector store.

        Supports two modes:
        - Local: Uses persistent directory with SQLite backend
        - Docker: Connects to ChromaDB server via HTTP client

        Returns:
            Chroma vector store instance

        Raises:
            ConnectionError: If Docker mode enabled but server unreachable
        """
        if self.use_docker:
            # Docker/HTTP client mode - connects to ChromaDB server
            logger.info(f"Connecting to ChromaDB server at {self.chroma_host}:{self.chroma_port}")
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                # Test connection
                self._chroma_client.heartbeat()
                logger.info("Successfully connected to ChromaDB server")

                return Chroma(
                    client=self._chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB server: {e}")
                raise ConnectionError(
                    f"Cannot connect to ChromaDB at {self.chroma_host}:{self.chroma_port}. "
                    "Ensure the Docker container is running: docker run -p 8000:8000 chromadb/chroma"
                ) from e
        else:
            # Local persistent mode
            logger.info(f"Using local ChromaDB at {self.persist_directory}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store with embeddings.

        Args:
            documents: List of LangChain Document objects to index

        Returns:
            List of document IDs

        Raises:
            Exception: If indexing fails

        Example:
            >>> ids = manager.add_documents(chunks)
            >>> print(f"Indexed {len(ids)} documents")
        """
        logger.info(f"Adding {len(documents)} documents to vector store...")
        ids = self.vector_store.add_documents(documents=documents)
        logger.info(f"Successfully indexed {len(ids)} documents")
        return ids

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_k: int = 3
    ) -> VectorStoreRetriever:
        """
        Get a retriever for similarity search.

        Args:
            search_type: Type of search ("similarity" or "mmr")
            search_k: Number of documents to retrieve

        Returns:
            Configured retriever instance

        Example:
            >>> retriever = manager.get_retriever(search_k=5)
            >>> docs = retriever.invoke("What is the main topic?")
        """
        logger.info(f"Creating retriever: type={search_type}, k={search_k}")
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": search_k}
        )

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents in vector store

        Example:
            >>> count = manager.get_collection_count()
            >>> print(f"Database contains {count} chunks")
        """
        try:
            count = self.vector_store._collection.count()
            logger.debug(f"Collection count: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def clear_collection(self) -> None:
        """
        Clear all documents from the vector store.

        This deletes the collection and recreates it empty.

        Raises:
            Exception: If clearing fails

        Example:
            >>> manager.clear_collection()
            >>> # Collection is now empty
        """
        try:
            logger.info("Clearing vector store collection...")
            self.vector_store.delete_collection()

            # Recreate empty collection
            self.vector_store = self._initialize_vector_store()
            logger.info("Vector store cleared and recreated")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}", exc_info=True)
            raise

    def get_indexed_documents(self) -> List[str]:
        """
        Get list of unique document sources in the vector store.

        Returns:
            List of unique source filenames

        Example:
            >>> docs = manager.get_indexed_documents()
            >>> print(f"Indexed documents: {docs}")
        """
        try:
            # Get all metadata from the collection
            collection = self.vector_store._collection
            result = collection.get(include=["metadatas"])

            # Extract unique source values
            sources = set()
            for metadata in result.get("metadatas", []):
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

            logger.debug(f"Found {len(sources)} indexed documents")
            return sorted(list(sources))

        except Exception as e:
            logger.error(f"Error getting indexed documents: {e}")
            return []

    def document_exists(self, filename: str) -> bool:
        """
        Check if a document with the given filename exists in the vector store.

        Args:
            filename: Name of the file to check

        Returns:
            True if document exists, False otherwise

        Example:
            >>> exists = manager.document_exists("report.pdf")
            >>> if exists:
            ...     print("Document already indexed!")
        """
        indexed_docs = self.get_indexed_documents()
        return filename in indexed_docs

    def search_similar(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of similar Document objects

        Example:
            >>> docs = manager.search_similar("machine learning", k=5)
            >>> for doc in docs:
            ...     print(doc.page_content[:100])
        """
        logger.info(f"Searching for similar documents: query='{query[:50]}...', k={k}")
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} similar documents")
        return results
