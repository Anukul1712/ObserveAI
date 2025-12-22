import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Config
from src.rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorStore:
    """
    LangChain Chroma-based vector store for semantic search.

    This wrapper handles initializing the vector store from a persistent
    directory and provides a custom method to add documents.

    The `create_and_store_embeddings` method implements its own
    batching, delay, and retry logic to avoid API rate limits
    and high memory usage, calling the embedding model directly.
    """

    def __init__(
        self,
        collection_name: str = "rag_collection"
    ):
        """
        Initialize or load the Chroma vector store.

        Args:
            collection_name: Name of the collection within Chroma.
        """
        self.embedding_model = EmbeddingModel()

        db_path_str = str(Config.VECTOR_DB_PATH)
        self.db_path = Config.VECTOR_DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing Chroma vector store at: {db_path_str}")

        self.client = chromadb.PersistentClient(path=db_path_str)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model.model_name}
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model.model,
        )

        logger.info(
            f"Vector store initialized. Collection '{self.collection_name}'\n"
            f"Loaded with {self.collection.count()} items."
        )

    def create_and_store_embeddings(
        self,
        documents: List[Document],
        processing_batch_size: int = 1024,
        embedding_batch_size: int = 64,
    ):
        """
        Creates embeddings and stores them in Chroma.
        Optimized for local GPU inference (approx 4GB VRAM).

        Args:
            documents: List of LangChain Document objects to add.
            processing_batch_size: Number of documents to write to DB at once.
            embedding_batch_size: Number of documents to embed at once (GPU batch).
        """
        if not documents:
            logger.warning("No documents provided to create embeddings.")
            return

        logger.info(
            f"Starting to create and store embeddings for {len(documents)} documents. "
            f"GPU Batch Size: {embedding_batch_size}"
        )

        total_added = 0

        for i in range(0, len(documents), processing_batch_size):
            batch_documents = documents[i: i + processing_batch_size]

            batch_texts = [doc.page_content for doc in batch_documents]
            batch_metadatas = [doc.metadata for doc in batch_documents]
            batch_ids = [str(uuid.uuid4()) for _ in batch_texts]

            all_batch_embeddings = []

            # Embed in sub-batches
            for j in range(0, len(batch_texts), embedding_batch_size):
                sub_batch_texts = batch_texts[j: j + embedding_batch_size]
                embeddings = self.embedding_model.model.embed_documents(
                    sub_batch_texts)
                all_batch_embeddings.extend(embeddings)

            # Store in Chroma
            if len(all_batch_embeddings) == len(batch_texts):
                self.collection.add(
                    embeddings=all_batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,  # type: ignore
                    ids=batch_ids
                )
                total_added += len(batch_documents)
                logger.info(
                    f"Stored {total_added}/{len(documents)} documents.")
            else:
                logger.error(
                    f"Embedding mismatch: {len(all_batch_embeddings)} embeddings for {len(batch_texts)} texts."
                )

        logger.info("Finished embedding generation and storage.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        """
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query, k=k)
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        return results

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> VectorStoreRetriever:
        """
        Get a LangChain retriever from the vector store as a Runnable

        Args:
            search_type: Type of search to perform (e.g., "similarity", "mmr").
            search_kwargs: Keyword arguments for the search (e.g., {"k": 4}).

        Returns:
            A LangChain VectorStoreRetriever.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        logger.debug(
            f"Creating retriever with search_type='{search_type}' and search_kwargs={search_kwargs}")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
