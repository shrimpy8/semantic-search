"""
Question Answering Chain Module

Handles retrieval-augmented generation (RAG) for answering questions based on documents.
"""

import logging
from typing import List, Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class QAChain:
    """
    Question-answering chain using retrieval-augmented generation.

    This class coordinates the RAG process:
    1. Retrieves relevant document chunks
    2. Formats context and question into a prompt
    3. Generates answer using LLM
    4. Streams the response token-by-token

    Attributes:
        llm_model: ChatOpenAI model for generation
        retriever: Vector store retriever for finding relevant chunks
        system_prompt: Template for QA system prompt

    Example:
        >>> qa = QAChain(
        ...     model_name="gpt-4o-mini",
        ...     retriever=retriever,
        ...     system_prompt="Answer based on: {document}"
        ... )
        >>> for chunk in qa.stream_answer("What is AI?"):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retriever: VectorStoreRetriever = None,
        system_prompt: str = None
    ):
        """
        Initialize the QA chain.

        Args:
            model_name: Name of ChatOpenAI model
            temperature: Model temperature (0.0 = deterministic)
            retriever: Vector store retriever instance
            system_prompt: System prompt template with {question} and {document} placeholders
        """
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = retriever

        # Initialize LLM
        self.llm_model = ChatOpenAI(model=model_name, temperature=temperature)
        logger.info(f"Initialized ChatOpenAI: model={model_name}, temp={temperature}")

        # Set default system prompt if not provided
        self.system_prompt = system_prompt or """
        You're a helpful assistant.
        Please answer the following question {question} only using the following information {document}.
        If you can't answer the question, just say you can't answer that question.
        """

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt)
        ])

    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User's question

        Returns:
            List of relevant Document objects

        Raises:
            ValueError: If retriever is not set

        Example:
            >>> docs = qa.retrieve_context("What is machine learning?")
            >>> print(f"Retrieved {len(docs)} chunks")
        """
        if not self.retriever:
            raise ValueError("Retriever not configured")

        logger.info(f"Retrieving context for query: {query[:50]}...")
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} relevant chunks")

        return docs

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.

        Args:
            documents: List of Document objects

        Returns:
            Concatenated document content

        Example:
            >>> context = qa.format_context(docs)
            >>> print(f"Context length: {len(context)} characters")
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Formatted context: {len(context)} characters")
        return context

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM (non-streaming).

        Args:
            question: User's question
            context: Retrieved document context

        Returns:
            Generated answer text

        Example:
            >>> answer = qa.generate_answer("What is AI?", context)
            >>> print(answer)
        """
        # Create prompt
        final_prompt = self.prompt_template.invoke({
            "question": question,
            "document": context
        })

        logger.info("Generating answer (non-streaming)...")
        response = self.llm_model.invoke(final_prompt)
        answer = response.content

        logger.info(f"Answer generated: {len(answer)} characters")
        return answer

    def stream_answer(self, question: str, context: str) -> Generator[str, None, None]:
        """
        Stream answer generation token-by-token.

        Args:
            question: User's question
            context: Retrieved document context

        Yields:
            Answer text chunks as they're generated

        Example:
            >>> for chunk in qa.stream_answer("What is AI?", context):
            ...     print(chunk, end="", flush=True)
        """
        # Create prompt
        final_prompt = self.prompt_template.invoke({
            "question": question,
            "document": context
        })

        logger.debug(f"Final prompt generated with {len(context)} characters of context")
        logger.info("Streaming LLM response...")

        # Stream response
        for chunk in self.llm_model.stream(final_prompt):
            yield chunk.content

    def answer_question(
        self,
        question: str,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """
        Complete RAG pipeline: retrieve, format, and answer.

        This is the main method that combines retrieval and generation.

        Args:
            question: User's question
            stream: Whether to stream the response

        Returns:
            Generator for streamed response, or string for non-streamed

        Raises:
            ValueError: If no relevant documents found

        Example:
            >>> # Streaming mode
            >>> for chunk in qa.answer_question("What is AI?"):
            ...     print(chunk, end="")
            >>>
            >>> # Non-streaming mode
            >>> answer = qa.answer_question("What is AI?", stream=False)
        """
        # Retrieve context
        docs = self.retrieve_context(question)

        if not docs:
            logger.warning("No relevant documents found")
            raise ValueError("No relevant information found in the document")

        # Format context
        context = self.format_context(docs)

        # Generate answer
        if stream:
            return self.stream_answer(question, context)
        else:
            return self.generate_answer(question, context)

    def update_retriever(self, retriever: VectorStoreRetriever) -> None:
        """
        Update the retriever instance.

        Args:
            retriever: New retriever instance

        Example:
            >>> new_retriever = vector_store.get_retriever(search_k=5)
            >>> qa.update_retriever(new_retriever)
        """
        self.retriever = retriever
        logger.info("Retriever updated")
