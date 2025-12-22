import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from src.config import Config

logger = logging.getLogger(__name__)


class TranscriptStore:
    """
    In-memory store for raw transcripts to enable fast context retrieval.
    Loads the full transcripts.json into memory (~100MB).
    """

    def __init__(self):
        self.transcripts: Dict[str, List[Dict[str, Any]]] = {}
        self._load_data()

    def _load_data(self):
        """Load transcripts from JSON file."""
        file_path = Config.LABELLED_TRANSCRIPT_FILE

        logger.info(f"Loading transcripts from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0

        for item in data:
            t_id = item.get("transcript_id")
            if t_id:
                self.transcripts[t_id] = item.get("conversation", [])
                count += 1

        logger.info(f"Loaded {count} transcripts into memory.")

    def get_turn(self, transcript_id: str, turn_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific turn by index."""
        conversation = self.transcripts.get(transcript_id)
        if not conversation:
            return None

        if 0 <= turn_index < len(conversation):
            return conversation[turn_index]
        return None

    def get_window_content(self, transcript_id: str, center_index: int, window_radius: int) -> List[str]:
        """
        Reconstructs the text content for a window centered at `center_index`.
        Matches the logic in DocumentProcessor: [i-window_radius , i+window_radius].
        """
        if window_radius < 2:
            return [""]

        conversation = self.transcripts.get(transcript_id)
        if not conversation:
            return [""]

        start_idx = max(0, center_index - window_radius)
        end_idx = min(len(conversation), center_index +
                      window_radius + 1)  # Exclusive

        window_turns = conversation[start_idx:end_idx]

        content_parts = []
        for t in window_turns:
            role = t.get("speaker", "Unknown")
            text = t.get("text", "")
            content_parts.append(f"{role}: {text}")

        return content_parts
