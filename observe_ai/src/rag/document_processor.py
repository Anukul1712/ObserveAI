import logging
import json
# import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from src.config import Config
from langchain_core.documents import Document
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Enhanced processor for conversation transcripts with semantic awareness"""

    def __init__(self, chunk_strategy: str = "semantic"):
        """
        Initialize document processor

        Args:
            chunk_strategy: "semantic" (turn-based, recommended) or "word" (original)
        """
        self.chunk_strategy = chunk_strategy

    def load_transcripts(self, file_path: Path) -> List[Dict]:
        """
        Load transcripts from a specific file

        Args:
            file_path: Path to the JSON transcript file

        Returns:
            List of loaded transcripts
        """
        transcripts = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both single transcript and list of transcripts
            if isinstance(data, list):
                transcripts.extend(data)
            else:
                transcripts.append(data)
        logger.info(
            f"Loaded {len(transcripts)} transcripts from {file_path}")

        return transcripts

    def process_transcripts(self, transcripts: List[Dict]) -> List[Document]:
        """
        Process multiple transcripts into searchable documents.
        Generates documents for each turn with surrounding context (window of 3).
        """
        all_documents = []

        for transcript in transcripts:
            conversation = transcript.get("conversation", [])
            if not conversation:
                logger.warning(
                    f"Empty conversation for {transcript.get('transcript_id')}")
                continue

            # Generate turn-level documents with context
            turn_docs = self._create_turn_documents_with_context(transcript)
            all_documents.extend(turn_docs)

        logger.info(
            f"Created {len(all_documents)} documents from {len(transcripts)} transcripts"
        )
        return all_documents

    def _create_turn_documents_with_context(self, transcript: Dict) -> List[Document]:
        """
        Create documents for each turn, including surrounding context (previous and next turn).
        This captures the flow: Prev -> Current -> Next.
        """
        conversation = transcript.get("conversation", [])
        documents = []

        # Transcript-level metadata
        tx_meta = {
            "transcript_id": transcript.get("transcript_id"),
            "domain": transcript.get("domain"),
            # Renamed to avoid confusion with turn intent
            "global_intent": transcript.get("intent"),
            "reason_for_call": transcript.get("reason_for_call"),
            "time_of_interaction": transcript.get("time_of_interaction")
        }

        for i, turn in enumerate(conversation):
            # Define window: [i-1, i, i+1]
            # Handle boundaries
            start_idx = max(0, i - 1)
            end_idx = min(len(conversation), i + 2)  # Exclusive

            window_turns = conversation[start_idx:end_idx]

            # Format content
            # e.g.
            # Agent: Hello...
            # Customer: Hi...
            # Agent: How can I help?
            content_parts = []
            for t in window_turns:
                role = t.get("speaker", "Unknown")
                text = t.get("text", "")
                content_parts.append(f"{role}: {text}")

            page_content = "\n".join(content_parts)

            # Turn-level metadata
            metadata = tx_meta.copy()
            metadata.update({
                "turn_id": turn.get("turn_id"),
                "turn_index": turn.get("turn_index"),
                "speaker": turn.get("speaker"),
                "primary_intent": turn.get("primary_intent"),
                "secondary_intent": turn.get("secondary_intent"),
                "type": "turn_with_context"
            })

            # Clean metadata (remove None values)
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))

        return documents

    # def _create_transcript_document(self, transcript: Dict) -> Document:
    #     """Create a document for the entire transcript"""
    #     conversation = transcript.get("conversation", [])
    #     full_text = self._conversation_to_text(conversation)
    #     metadata = self._extract_metadata(transcript, conversation)
    #     metadata = metadata.update({
    #         "type": "full_transcript",
    #     })

    #     return Document(
    #         page_content=full_text, metadata=metadata
    #     )

    # def _chunk_by_turn_groups(self, transcript: Dict) -> List[Document]:
    #     """
    #     Chunk conversation into semantic groups (turn-aware)
    #     Groups typically contain 2-3 related turns (exchange pattern)
    #     """
    #     conversation = transcript.get("conversation", [])
    #     documents = []
    #     group_idx = 0

    #     i = 0
    #     while i < len(conversation):
    #         group = []
    #         start_idx = i

    #         # Collect turns for a semantic unit
    #         # Strategy: Customer question → Agent answer → Customer acknowledgement
    #         group.append(conversation[i])
    #         i += 1

    #         # If customer initiated, get agent response
    #         if (i < len(conversation) and
    #             conversation[start_idx].get("speaker") == "Customer" and
    #                 conversation[i].get("speaker") != "Customer"):
    #             group.append(conversation[i])
    #             i += 1

    #         # Get one more turn for additional context
    #         if i < len(conversation) and len(group) < 3:
    #             group.append(conversation[i])
    #             i += 1

    #         # Create group document
    #         group_text = self._conversation_to_text(group)
    #         metadata = self._extract_metadata(transcript, conversation)
    #         metadata.update({
    #             "type": "turn_group",
    #         })

    #         documents.append(Document(
    #             page_content=group_text, metadata=metadata
    #         ))

    #         group_idx += 1

    #     return documents

    # def _chunk_by_words(self, transcript: Dict) -> List[Document]:
    #     """
    #     Original word-based chunking (fallback)
    #     Preserves metadata for each chunk
    #     """
    #     conversation = transcript.get("conversation", [])
    #     full_text = self._conversation_to_text(conversation)

    #     words = full_text.split()
    #     documents = []
    #     chunk_size = Config.CHUNK_SIZE
    #     chunk_overlap = Config.CHUNK_OVERLAP
    #     start_idx = 0
    #     chunk_idx = 0

    #     while start_idx < len(words):
    #         end_idx = min(start_idx + chunk_size, len(words))
    #         chunk_text = " ".join(words[start_idx:end_idx])

    #         metadata = self._extract_metadata(transcript, conversation)
    #         metadata.update({
    #             "type": "word_chunk",
    #         })

    #         documents.append(Document(
    #             page_content=chunk_text, metadata=metadata
    #         ))

    #         start_idx = end_idx - chunk_overlap
    #         chunk_idx += 1

    #     return documents

    # def _create_turn_documents(self, transcript: Dict) -> List[Document]:
    #     """Create individual documents for each turn (fine-grained)"""
    #     conversation = transcript.get("conversation", [])
    #     documents = []

    #     for idx, turn in enumerate(conversation):
    #         speaker = turn.get("speaker", "Unknown")
    #         text = turn.get("text", "")

    #         metadata = self._extract_metadata(transcript, conversation)
    #         metadata.update({
    #             "type": "turn",
    #         })

    #         documents.append(Document(
    #             page_content=f"{speaker}: {text}", metadata=metadata
    #         ))

    #     return documents

    # def _conversation_to_text(self, turns: List[Dict]) -> str:
    #     """Convert conversation turns to readable text format"""
    #     text_parts = []
    #     for turn in turns:
    #         speaker = turn.get("speaker", "Unknown")
    #         text = turn.get("text", "")
    #         text_parts.append(f"{speaker}: {text}")
    #     return " ".join(text_parts)

    # def _extract_metadata(self, transcript: Dict, conversation: List[Dict]) -> Dict:
    #     """Extract comprehensive metadata for searchability and filtering"""
    #     return {
    #         "transcript_id": transcript.get("transcript_id", "unknown"),
    #         "domain": transcript.get("domain", "general"),
    #         "intent": transcript.get("intent", "unknown"),
    #         "outcome": transcript.get("outcome", "unknown"),
    #     }
