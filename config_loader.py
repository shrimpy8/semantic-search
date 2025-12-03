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
        return self.config.get("vector_store", {}).get("collection_name", "semantic_search_docs")

    def get_persist_directory(self) -> str:
        """Get ChromaDB persistence directory."""
        return self.config.get("vector_store", {}).get("persist_directory", "./chroma/db")

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
