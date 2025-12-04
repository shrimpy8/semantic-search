"""
Core Module for Semantic Search

This module contains the core functionality for document processing,
vector storage, question-answering, hybrid retrieval, re-ranking,
conversation history, and A/B testing.
"""

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .qa_chain import QAChain
from .bm25_retriever import BM25Retriever, BM25Result
from .reranker import (
    BaseReranker,
    CohereReranker,
    JinaReranker,
    RerankerFactory,
    RerankResult
)
from .hybrid_retriever import (
    HybridRetriever,
    HybridResult,
    RetrievalMethod,
    create_hybrid_retriever
)
from .conversation import (
    ConversationManager,
    ConversationSession,
    ConversationMessage,
    QueryRecord
)
from .ab_testing import (
    ABTestingManager,
    ABTestExperiment,
    ABTestResult,
    TestVariant,
    RetrievalMetrics
)

__all__ = [
    # Original exports
    'DocumentProcessor',
    'VectorStoreManager',
    'QAChain',
    # BM25 Retriever
    'BM25Retriever',
    'BM25Result',
    # Re-rankers
    'BaseReranker',
    'CohereReranker',
    'JinaReranker',
    'RerankerFactory',
    'RerankResult',
    # Hybrid Retriever
    'HybridRetriever',
    'HybridResult',
    'RetrievalMethod',
    'create_hybrid_retriever',
    # Conversation History
    'ConversationManager',
    'ConversationSession',
    'ConversationMessage',
    'QueryRecord',
    # A/B Testing
    'ABTestingManager',
    'ABTestExperiment',
    'ABTestResult',
    'TestVariant',
    'RetrievalMetrics'
]
