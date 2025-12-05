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
        search_k: int = 3,
        filter: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """
        Get a retriever for similarity search.

        Args:
            search_type: Type of search ("similarity" or "mmr")
            search_k: Number of documents to retrieve
            filter: Optional ChromaDB filter dict for scoped search

        Returns:
            Configured retriever instance

        Example:
            >>> # Basic retriever
            >>> retriever = manager.get_retriever(search_k=5)

            >>> # Collection-scoped retriever
            >>> retriever = manager.get_retriever(
            ...     search_k=5,
            ...     filter={"collection_id": "abc123"}
            ... )

            >>> # Document-scoped retriever
            >>> retriever = manager.get_retriever(
            ...     search_k=5,
            ...     filter={"document_id": {"$in": ["doc1", "doc2"]}}
            ... )
        """
        search_kwargs = {"k": search_k}

        if filter:
            search_kwargs["filter"] = filter
            logger.info(f"Creating retriever: type={search_type}, k={search_k}, filter={filter}")
        else:
            logger.info(f"Creating retriever: type={search_type}, k={search_k}")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
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

    def search_similar(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional ChromaDB filter for scoped search

        Returns:
            List of similar Document objects

        Example:
            >>> # Basic search
            >>> docs = manager.search_similar("machine learning", k=5)

            >>> # Collection-scoped search
            >>> docs = manager.search_similar(
            ...     "machine learning",
            ...     k=5,
            ...     filter={"collection_id": "abc123"}
            ... )
        """
        query_preview = query[:50] if len(query) > 50 else query
        logger.info(f"Searching for similar documents: query='{query_preview}...', k={k}, filter={filter}")

        if filter:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
        else:
            results = self.vector_store.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} similar documents")
        return results

    def search_by_collection(
        self,
        query: str,
        collection_id: str,
        k: int = 3
    ) -> List[Document]:
        """
        Search within a specific collection.

        Args:
            query: Search query text
            collection_id: Collection to search within
            k: Number of results to return

        Returns:
            List of similar Document objects from the collection

        Example:
            >>> docs = manager.search_by_collection(
            ...     "machine learning",
            ...     collection_id="abc123",
            ...     k=5
            ... )
        """
        return self.search_similar(
            query=query,
            k=k,
            filter={"collection_id": collection_id}
        )

    def search_by_documents(
        self,
        query: str,
        document_ids: List[str],
        k: int = 3
    ) -> List[Document]:
        """
        Search within specific documents.

        Args:
            query: Search query text
            document_ids: List of document IDs to search within
            k: Number of results to return

        Returns:
            List of similar Document objects from the specified documents

        Example:
            >>> docs = manager.search_by_documents(
            ...     "machine learning",
            ...     document_ids=["doc1", "doc2"],
            ...     k=5
            ... )
        """
        if len(document_ids) == 1:
            filter_dict = {"document_id": document_ids[0]}
        else:
            filter_dict = {"document_id": {"$in": document_ids}}

        return self.search_similar(query=query, k=k, filter=filter_dict)

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks for a document from the vector store.

        Args:
            document_id: Document ID whose chunks to delete

        Returns:
            Number of chunks deleted

        Example:
            >>> deleted = manager.delete_by_document_id("xyz789")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            collection = self.vector_store._collection

            # Get IDs of chunks with this document_id
            results = collection.get(
                where={"document_id": document_id},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"No chunks found for document {document_id}")
                return 0

            # Delete the chunks
            collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")
            return 0

    def delete_by_collection_id(self, collection_id: str) -> int:
        """
        Delete all chunks for a collection from the vector store.

        Args:
            collection_id: Collection ID whose chunks to delete

        Returns:
            Number of chunks deleted

        Example:
            >>> deleted = manager.delete_by_collection_id("abc123")
            >>> print(f"Deleted {deleted} chunks")
        """
        try:
            collection = self.vector_store._collection

            # Get IDs of chunks with this collection_id
            results = collection.get(
                where={"collection_id": collection_id},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"No chunks found for collection {collection_id}")
                return 0

            # Delete the chunks
            collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for collection {collection_id}")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Error deleting chunks for collection {collection_id}: {e}")
            return 0

    def get_chunks_by_document(self, document_id: str) -> List[Document]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID to get chunks for

        Returns:
            List of Document chunks

        Example:
            >>> chunks = manager.get_chunks_by_document("xyz789")
            >>> print(f"Found {len(chunks)} chunks")
        """
        try:
            collection = self.vector_store._collection
            results = collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )

            chunks = []
            for i, content in enumerate(results.get("documents", [])):
                metadata = results.get("metadatas", [])[i] if results.get("metadatas") else {}
                chunks.append(Document(page_content=content, metadata=metadata))

            logger.debug(f"Found {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []

    def clear_non_collection_documents(self) -> int:
        """
        Clear all documents that are NOT part of any collection.

        These are legacy documents uploaded via the home page file uploader
        that don't have a collection_id in their metadata.

        Returns:
            Number of chunks deleted

        Example:
            >>> deleted = manager.clear_non_collection_documents()
            >>> print(f"Deleted {deleted} legacy chunks")
        """
        try:
            collection = self.vector_store._collection

            # Get all documents
            all_results = collection.get(include=["metadatas"])
            all_ids = all_results.get("ids", [])
            all_metadatas = all_results.get("metadatas", [])

            # Find IDs without collection_id
            ids_to_delete = []
            for i, metadata in enumerate(all_metadatas):
                if not metadata or "collection_id" not in metadata:
                    ids_to_delete.append(all_ids[i])

            if not ids_to_delete:
                logger.info("No non-collection documents found to delete")
                return 0

            # Delete in batches (ChromaDB has limits)
            batch_size = 5000
            total_deleted = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                collection.delete(ids=batch)
                total_deleted += len(batch)

            logger.info(f"Deleted {total_deleted} non-collection chunks")
            return total_deleted

        except Exception as e:
            logger.error(f"Error clearing non-collection documents: {e}", exc_info=True)
            return 0

    def clear_all_collection_documents(self) -> int:
        """
        Clear all documents that ARE part of collections.

        These are documents with a collection_id in their metadata,
        uploaded via the Collections page.

        Returns:
            Number of chunks deleted

        Example:
            >>> deleted = manager.clear_all_collection_documents()
            >>> print(f"Deleted {deleted} collection chunks")
        """
        try:
            collection = self.vector_store._collection

            # Get all documents
            all_results = collection.get(include=["metadatas"])
            all_ids = all_results.get("ids", [])
            all_metadatas = all_results.get("metadatas", [])

            # Find IDs with collection_id
            ids_to_delete = []
            for i, metadata in enumerate(all_metadatas):
                if metadata and "collection_id" in metadata:
                    ids_to_delete.append(all_ids[i])

            if not ids_to_delete:
                logger.info("No collection documents found to delete")
                return 0

            # Delete in batches (ChromaDB has limits)
            batch_size = 5000
            total_deleted = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                collection.delete(ids=batch)
                total_deleted += len(batch)

            logger.info(f"Deleted {total_deleted} collection chunks")
            return total_deleted

        except Exception as e:
            logger.error(f"Error clearing collection documents: {e}", exc_info=True)
            return 0

    def get_non_collection_count(self) -> int:
        """
        Get count of chunks NOT in any collection.

        Returns:
            Number of non-collection chunks
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["metadatas"])
            all_metadatas = all_results.get("metadatas", [])

            count = sum(1 for m in all_metadatas if not m or "collection_id" not in m)
            return count
        except Exception as e:
            logger.error(f"Error getting non-collection count: {e}")
            return 0

    def get_collection_documents_count(self) -> int:
        """
        Get count of chunks IN collections.

        Returns:
            Number of collection chunks
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["metadatas"])
            all_metadatas = all_results.get("metadatas", [])

            count = sum(1 for m in all_metadatas if m and "collection_id" in m)
            return count
        except Exception as e:
            logger.error(f"Error getting collection documents count: {e}")
            return 0

    def get_all_documents(self, collection_id: Optional[str] = None) -> List[Document]:
        """
        Get all documents from the vector store.

        Args:
            collection_id: Optional collection ID to filter by.
                          If None, returns non-collection documents.

        Returns:
            List of Document objects
        """
        try:
            collection = self.vector_store._collection
            all_results = collection.get(include=["documents", "metadatas"])

            documents = []
            contents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])

            for content, metadata in zip(contents, metadatas):
                if collection_id:
                    # Filter by specific collection
                    if metadata and metadata.get("collection_id") == collection_id:
                        documents.append(Document(page_content=content, metadata=metadata or {}))
                else:
                    # Get non-collection documents only
                    if not metadata or "collection_id" not in metadata:
                        documents.append(Document(page_content=content, metadata=metadata or {}))

            logger.info(f"Retrieved {len(documents)} documents from vector store")
            return documents

        except Exception as e:
            logger.error(f"Error getting all documents: {e}", exc_info=True)
            return []
