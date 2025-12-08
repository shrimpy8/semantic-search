"""
Configuration Loader Module

Handles loading and parsing of YAML configuration files for the semantic search application.
Provides type-safe access to configuration values with validation.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for semantic search application.

    Loads YAML configuration files and provides type-safe access to configuration
    values. Validates configuration structure and provides helpful error messages.

    Attributes:
        config_path: Path to the YAML configuration file
        config: Loaded configuration dictionary

    Example:
        >>> config = ConfigLoader("config.yaml")
        >>> chunk_size = config.get_chunk_size()
        >>> embedding_model = config.get_embedding_model()
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the YAML configuration file (default: config.yaml)

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML file is malformed
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure config.yaml exists in the project directory."
            )

        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise

    def get_embedding_model(self) -> str:
        """Get embedding model name."""
        return self.config.get("models", {}).get("embedding", {}).get("name", "text-embedding-3-large")

    def get_chat_model(self) -> str:
        """Get chat model name."""
        return self.config.get("models", {}).get("chat", {}).get("name", "gpt-4o-mini")

    def get_chat_temperature(self) -> float:
        """Get chat model temperature."""
        return self.config.get("models", {}).get("chat", {}).get("temperature", 0.0)

    def get_chunk_size(self) -> int:
        """Get document chunk size."""
        return self.config.get("document_processing", {}).get("chunk_size", 1000)

    def get_chunk_overlap(self) -> int:
        """Get document chunk overlap."""
        return self.config.get("document_processing", {}).get("chunk_overlap", 200)

    def get_add_start_index(self) -> bool:
        """Get whether to add start index to chunks."""
        return self.config.get("document_processing", {}).get("add_start_index", True)

    def get_collection_name(self) -> str:
        """Get ChromaDB collection name."""
        return self.config.get("vector_store", {}).get("collection_name", "semantic_search_docs_streamlit")

    def get_persist_directory(self) -> str:
        """Get ChromaDB persistence directory."""
        return self.config.get("vector_store", {}).get("persist_directory", "./chroma/db")

    def use_chroma_docker(self) -> bool:
        """Check if ChromaDB Docker mode is enabled."""
        return self.config.get("vector_store", {}).get("use_docker", False)

    def get_chroma_host(self) -> str:
        """Get ChromaDB server hostname (Docker mode)."""
        return self.config.get("vector_store", {}).get("chroma_host", "localhost")

    def get_chroma_port(self) -> int:
        """Get ChromaDB server port (Docker mode)."""
        return self.config.get("vector_store", {}).get("chroma_port", 8000)

    def get_search_type(self) -> str:
        """Get vector search type."""
        return self.config.get("vector_store", {}).get("search_type", "similarity")

    def get_search_k(self) -> int:
        """Get number of chunks to retrieve."""
        return self.config.get("vector_store", {}).get("search_k", 3)

    def get_retry_config(self) -> Dict[str, int]:
        """Get retry configuration."""
        return self.config.get("retry", {
            "max_attempts": 3,
            "min_wait": 2,
            "max_wait": 10,
            "multiplier": 1
        })

    def get_logging_config(self) -> Dict[str, str]:
        """Get logging configuration."""
        return self.config.get("logging", {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "semantic_search.log"
        })

    def get_qa_system_prompt(self) -> str:
        """Get QA system prompt template."""
        return self.config.get("prompts", {}).get("qa_system",
            "You're a helpful assistant. Please answer the following question {question} only using the following information {document}. If you can't answer the question, just say you can't answer that question.")

    def get_follow_up_system_prompt(self) -> str:
        """Get follow-up question system prompt template."""
        return self.config.get("prompts", {}).get("follow_up_system",
            """You're a helpful assistant engaged in a conversation about a document.
            Use the previous conversation context to understand references and maintain continuity.
            Previous conversation: {conversation_context}
            Document context: {document}
            Please answer the following question: {question}
            If you can't answer the question, just say you can't answer that question.""")

    # Hybrid Retrieval Configuration
    def is_hybrid_retrieval_enabled(self) -> bool:
        """Check if hybrid retrieval is enabled."""
        return self.config.get("hybrid_retrieval", {}).get("enabled", True)

    def get_default_retrieval_method(self) -> str:
        """Get default retrieval method."""
        return self.config.get("hybrid_retrieval", {}).get("default_method", "hybrid")

    def get_hybrid_alpha(self) -> float:
        """Get alpha weight for semantic search in hybrid mode."""
        return self.config.get("hybrid_retrieval", {}).get("alpha", 0.5)

    def get_rrf_k(self) -> int:
        """Get RRF constant for rank fusion."""
        return self.config.get("hybrid_retrieval", {}).get("rrf_k", 60)

    def get_bm25_k1(self) -> float:
        """Get BM25 k1 parameter (term frequency saturation)."""
        return self.config.get("hybrid_retrieval", {}).get("bm25", {}).get("k1", 1.5)

    def get_bm25_b(self) -> float:
        """Get BM25 b parameter (length normalization)."""
        return self.config.get("hybrid_retrieval", {}).get("bm25", {}).get("b", 0.75)

    def is_reranking_enabled(self) -> bool:
        """Check if re-ranking is enabled."""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get("enabled", True)

    def get_reranker_provider(self) -> str:
        """Get re-ranker provider."""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get("provider", "auto")

    def get_cohere_rerank_model(self) -> str:
        """Get Cohere re-rank model name."""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "cohere_model", "rerank-english-v3.0")

    def get_jina_rerank_model(self) -> str:
        """Get Jina re-rank model name."""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "jina_model", "jinaai/jina-reranker-v1-tiny-en")

    def get_fetch_k_multiplier(self) -> int:
        """Get fetch_k multiplier for reranking."""
        return self.config.get("hybrid_retrieval", {}).get("reranking", {}).get(
            "fetch_k_multiplier", 3)

    # Conversation Configuration
    def is_conversation_enabled(self) -> bool:
        """Check if conversation history is enabled."""
        return self.config.get("conversation", {}).get("enabled", True)

    def get_conversation_storage_dir(self) -> str:
        """Get conversation storage directory."""
        return self.config.get("conversation", {}).get("storage_dir", "./conversation_history")

    def get_max_conversation_history(self) -> int:
        """Get maximum conversation history per session."""
        return self.config.get("conversation", {}).get("max_history", 50)

    def is_follow_up_optimization_enabled(self) -> bool:
        """Check if follow-up question optimization is enabled."""
        return self.config.get("conversation", {}).get("follow_up_optimization", True)

    def get_conversation_context_window(self) -> int:
        """Get number of recent Q&A pairs for follow-up context."""
        return self.config.get("conversation", {}).get("context_window", 3)

    # A/B Testing Configuration
    def is_ab_testing_enabled(self) -> bool:
        """Check if A/B testing is enabled."""
        return self.config.get("ab_testing", {}).get("enabled", True)

    def get_ab_testing_storage_dir(self) -> str:
        """Get A/B testing storage directory."""
        return self.config.get("ab_testing", {}).get("storage_dir", "./ab_testing_results")

    def get_default_ab_variants(self) -> list:
        """Get default A/B testing variants."""
        return self.config.get("ab_testing", {}).get("default_variants",
            ["semantic", "bm25", "hybrid", "hybrid_rerank"])

    def get_hybrid_retrieval_config(self) -> Dict[str, Any]:
        """Get complete hybrid retrieval configuration."""
        return self.config.get("hybrid_retrieval", {
            "enabled": True,
            "default_method": "hybrid",
            "alpha": 0.5,
            "rrf_k": 60,
            "bm25": {"k1": 1.5, "b": 0.75},
            "reranking": {
                "enabled": True,
                "provider": "auto",
                "cohere_model": "rerank-english-v3.0",
                "jina_model": "jinaai/jina-reranker-v1-tiny-en",
                "fetch_k_multiplier": 3
            }
        })

    def get_conversation_config(self) -> Dict[str, Any]:
        """Get complete conversation configuration."""
        return self.config.get("conversation", {
            "enabled": True,
            "storage_dir": "./conversation_history",
            "max_history": 50,
            "follow_up_optimization": True,
            "context_window": 3
        })

    def get_ab_testing_config(self) -> Dict[str, Any]:
        """Get complete A/B testing configuration."""
        return self.config.get("ab_testing", {
            "enabled": True,
            "storage_dir": "./ab_testing_results",
            "default_variants": ["semantic", "bm25", "hybrid", "hybrid_rerank"]
        })

    # Retrieval Presets Configuration
    def get_retrieval_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all retrieval presets.

        Returns:
            Dictionary of preset configurations keyed by preset name.

        Example:
            >>> presets = config.get_retrieval_presets()
            >>> print(presets["high_precision"]["display_name"])
            'High Precision'
        """
        presets_config = self.config.get("retrieval_presets", {})

        # Default presets if none configured
        default_presets = {
            "high_precision": {
                "display_name": "High Precision",
                "description": "Fewer results, higher relevance. Best for specific questions.",
                "icon": "🎯",
                "k": 3,
                "alpha": 0.7,
                "rerank": True,
                "method": "hybrid"
            },
            "balanced": {
                "display_name": "Balanced",
                "description": "Good balance of precision and coverage. Recommended default.",
                "icon": "⚖️",
                "k": 5,
                "alpha": 0.5,
                "rerank": True,
                "method": "hybrid"
            },
            "high_recall": {
                "display_name": "High Recall",
                "description": "More results, broader coverage. Best for exploration.",
                "icon": "🔍",
                "k": 10,
                "alpha": 0.3,
                "rerank": False,
                "method": "hybrid"
            }
        }

        # Filter out non-preset keys like 'default_preset'
        presets = {}
        for key, value in presets_config.items():
            if isinstance(value, dict) and "display_name" in value:
                presets[key] = value

        return presets if presets else default_presets

    def get_preset_by_name(self, preset_name: str) -> Dict[str, Any]:
        """
        Get a specific preset by name.

        Args:
            preset_name: Name of the preset (e.g., "high_precision", "balanced")

        Returns:
            Preset configuration dictionary

        Raises:
            KeyError: If preset name doesn't exist

        Example:
            >>> preset = config.get_preset_by_name("high_precision")
            >>> print(preset["k"])
            3
        """
        presets = self.get_retrieval_presets()
        if preset_name not in presets:
            raise KeyError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        return presets[preset_name]

    def get_default_preset(self) -> str:
        """
        Get the default preset name.

        Returns:
            Name of the default preset
        """
        return self.config.get("retrieval_presets", {}).get("default_preset", "balanced")

    def get_preset_names(self) -> list:
        """
        Get list of available preset names.

        Returns:
            List of preset names
        """
        return list(self.get_retrieval_presets().keys())


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Load configuration from YAML file.

    Convenience function that creates and returns a ConfigLoader instance.

    Args:
        config_path: Path to configuration file (default: config.yaml)

    Returns:
        Configured ConfigLoader instance

    Example:
        >>> config = load_config()
        >>> chunk_size = config.get_chunk_size()
    """
    return ConfigLoader(config_path)
