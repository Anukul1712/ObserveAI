import logging
from pathlib import Path
from typing import Optional
from src.config import Config
from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_vectors(transcripts_file: Optional[Path] = None) -> int:
    """
    Initialize vector store from transcripts

    Args:
        transcripts_file: Path to JSON transcripts file

    Returns:
        Number of documents added
    """
    logger.info("Initializing vector store from transcripts...")

    Config.validate()

    transcripts_file = transcripts_file or Config.LABELLED_TRANSCRIPT_FILE

    document_processor = DocumentProcessor(chunk_strategy="semantic")
    # VectorStore initializes EmbeddingModel internally
    vector_store = VectorStore()

    # Load transcripts
    transcripts = document_processor.load_transcripts(transcripts_file)
    if not transcripts:
        logger.warning(f"No transcripts found in {transcripts_file}")
        return 0
    logger.info(f"Loaded {len(transcripts)} transcripts")

    # Process documents
    documents = document_processor.process_transcripts(transcripts)
    if not documents:
        logger.warning("No documents generated from transcripts")
        return 0
    logger.info(f"Processed into {len(documents)} document chunks")

    # Add to vector store with batch processing to avoid rate limits
    vector_store.create_and_store_embeddings(documents)
    logger.info(f"Vector store initialized with {len(documents)} documents")
    return len(documents)


if __name__ == "__main__":
    initialize_vectors()
