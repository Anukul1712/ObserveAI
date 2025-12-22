# kaggle_vector_db_creator.py
# This script is designed to be run on Kaggle to create a ChromaDB vector store from transcripts.
# It uses GPU acceleration for embedding generation.

import os
import json
import logging
import shutil
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

# --- Install Dependencies (Uncomment if running in a notebook cell) ---
# !pip install -q langchain langchain-community langchain-huggingface chromadb sentence-transformers

import torch
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration ---
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    INPUT_FILE = "labeled_transcripts_for_graph_allminiLML6.json" # Upload this file to Kaggle
    OUTPUT_DIR = "vector_db"
    COLLECTION_NAME = "rag_collection"
    
    # Batch sizes optimized for T4 GPU (common on Kaggle)
    PROCESSING_BATCH_SIZE = 2000
    EMBEDDING_BATCH_SIZE = 128

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Document Processor ---
class DocumentProcessor:
    """Enhanced processor for conversation transcripts with semantic awareness"""

    def __init__(self):
        pass

    def load_transcripts(self, file_path: str) -> List[Dict]:
        """Load transcripts from a specific file"""
        transcripts = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    transcripts.extend(data)
                else:
                    transcripts.append(data)
            logger.info(f"Loaded {len(transcripts)} transcripts from {file_path}")
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
        return transcripts

    def process_transcripts(self, transcripts: List[Dict]) -> List[Document]:
        """Process multiple transcripts into searchable documents."""
        all_documents = []
        for transcript in transcripts:
            conversation = transcript.get("conversation", [])
            if not conversation:
                continue
            turn_docs = self._create_turn_documents_with_context(transcript)
            all_documents.extend(turn_docs)
        
        logger.info(f"Created {len(all_documents)} documents from {len(transcripts)} transcripts")
        return all_documents

    def _create_turn_documents_with_context(self, transcript: Dict) -> List[Document]:
        """Create documents for each turn with surrounding context (window of 3)."""
        conversation = transcript.get("conversation", [])
        documents = []
        
        tx_meta = {
            "transcript_id": transcript.get("transcript_id"),
            "domain": transcript.get("domain"),
            "global_intent": transcript.get("intent"),
            "reason_for_call": transcript.get("reason_for_call"),
            "time_of_interaction": transcript.get("time_of_interaction")
        }

        for i, turn in enumerate(conversation):
            start_idx = max(0, i - 1)
            end_idx = min(len(conversation), i + 2)
            window_turns = conversation[start_idx:end_idx]
            
            content_parts = []
            for t in window_turns:
                role = t.get("speaker", "Unknown")
                text = t.get("text", "")
                content_parts.append(f"{role}: {text}")
            
            page_content = "\n".join(content_parts)
            
            metadata = tx_meta.copy()
            metadata.update({
                "turn_id": turn.get("turn_id"),
                "turn_index": turn.get("turn_index"),
                "speaker": turn.get("speaker"),
                "primary_intent": turn.get("primary_intent"),
                "secondary_intent": turn.get("secondary_intent"),
                "type": "turn_with_context"
            })
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

# --- Vector Store Creator ---
class VectorStoreCreator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': self.device}
        )
        
        # Initialize Chroma Client
        if os.path.exists(Config.OUTPUT_DIR):
            shutil.rmtree(Config.OUTPUT_DIR)
        os.makedirs(Config.OUTPUT_DIR)
        
        self.client = chromadb.PersistentClient(path=Config.OUTPUT_DIR)
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"embedding_model": Config.EMBEDDING_MODEL}
        )

    def create_and_store_embeddings(self, documents: List[Document]):
        if not documents:
            return

        logger.info(f"Starting embedding generation for {len(documents)} documents...")
        
        total_added = 0
        processing_batch_size = Config.PROCESSING_BATCH_SIZE
        embedding_batch_size = Config.EMBEDDING_BATCH_SIZE

        for i in range(0, len(documents), processing_batch_size):
            batch_documents = documents[i : i + processing_batch_size]
            
            batch_texts = [doc.page_content for doc in batch_documents]
            batch_metadatas = [doc.metadata for doc in batch_documents]
            batch_ids = [str(uuid.uuid4()) for _ in batch_texts]
            
            all_batch_embeddings = []

            # Embed in sub-batches
            for j in range(0, len(batch_texts), embedding_batch_size):
                sub_batch_texts = batch_texts[j : j + embedding_batch_size]
                try:
                    embeddings = self.embedding_model.embed_documents(sub_batch_texts)
                    all_batch_embeddings.extend(embeddings)
                except Exception as e:
                    logger.error(f"Error embedding sub-batch: {e}")
                    raise e

            # Store in Chroma
            if len(all_batch_embeddings) == len(batch_texts):
                self.collection.add(
                    embeddings=all_batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                total_added += len(batch_documents)
                logger.info(f"Stored {total_added}/{len(documents)} documents.")
            else:
                logger.error("Embedding mismatch!")

        logger.info("Finished embedding generation and storage.")

# --- Main Execution ---
def main():
    # 1. Check for input file
    if not os.path.exists(Config.INPUT_FILE):
        print(f"Error: Input file '{Config.INPUT_FILE}' not found.")
        print("Please upload the transcript JSON file to the Kaggle environment.")
        return

    # 2. Process Documents
    processor = DocumentProcessor()
    transcripts = processor.load_transcripts(Config.INPUT_FILE)
    documents = processor.process_transcripts(transcripts)
    
    if not documents:
        print("No documents to process.")
        return

    # 3. Create Vector Store
    creator = VectorStoreCreator()
    creator.create_and_store_embeddings(documents)
    
    # 4. Zip the output
    print("Zipping vector database...")
    shutil.make_archive("vector_db", 'zip', Config.OUTPUT_DIR)
    print(f"Done! Download 'vector_db.zip' from the output section.")

if __name__ == "__main__":
    main()
