"""
Vector Store Manager Module

Handles ChromaDB vector store operations including document indexing and retrieval.
"""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.

    This class handles all interactions with the ChromaDB vector store including:
    - Vector store initialization and persistence
    - Document embedding and indexing
    - Similarity search and retrieval
    - Collection management (clear, delete)

    Attributes:
        embedding_model: OpenAI embeddings model
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory for persistent storage
        vector_store: ChromaDB vector store instance

    Example:
        >>> manager = VectorStoreManager(
        ...     embedding_model_name="text-embedding-3-large",
        ...     collection_name="my_docs"
        ... )
        >>> ids = manager.add_documents(chunks)
        >>> retriever = manager.get_retriever(search_k=3)
    """

    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-large",
        collection_name: str = "semantic_search_docs",
        persist_directory: str = "./chroma/db"
    ):
        """
        Initialize the vector store manager.

        Args:
            embedding_model_name: Name of OpenAI embedding model
            collection_name: Name for ChromaDB collection
            persist_directory: Directory path for persistence
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        logger.info(f"Initialized OpenAI embeddings: {embedding_model_name}")

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        logger.info(f"Vector store initialized: collection={collection_name}")

    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize ChromaDB vector store.

        Returns:
            Chroma vector store instance
        """
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
