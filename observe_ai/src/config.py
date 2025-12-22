"""Configuration management for Observe AI"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env in observe_ai directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration"""

    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPEN_ROUTER_KEY: str = os.getenv("OPEN_ROUTER_KEY", "")
    GEMINI_MODEL: str = "google/gemini-2.5-flash-lite"
    DEEPSEEK_MODEL: str = "deepseek/deepseek-v3.2"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "12345678")

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TRANSCRIPTS_DIR: Path = DATA_DIR / "transcripts"
    LABELLED_TRANSCRIPT_FILE: Path = DATA_DIR / \
        "labelled_transcript_bert.json"
    VECTOR_DB_PATH: Path = DATA_DIR / "vector_db"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "app.log"

    @classmethod
    def validate(cls) -> None:
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in .env file")

        # Create required directories
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls) -> dict:
        """Convert configuration to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, Path))
        }
