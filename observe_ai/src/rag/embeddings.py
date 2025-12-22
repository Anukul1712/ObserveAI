import logging
# import torch
import numpy as np
from src.config import Config
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        """
        Initialize embedding model
        """
        self.api_key = Config.GEMINI_API_KEY
        self.model_name = Config.EMBEDDING_MODEL

        logger.info(f"Loading embedding model: {self.model_name}")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # FORCE CPU USE
        device = "cpu"
        logger.info(f"Using device: {device}")

        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': device}
        )
        # self.model = GoogleGenerativeAIEmbeddings(
        #     model=self.model_name,
        #     google_api_key=self.api_key  # type: ignore
        # )
        logger.info(f"Initialized embedding model: {self.model_name}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string
        """
        return np.array(self.model.embed_query(query))
