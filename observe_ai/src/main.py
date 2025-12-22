import logging
from pathlib import Path
from typing import Optional, List, Dict
from src.config import Config
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.rag.document_processor import DocumentProcessor
from src.rag.agentic_graph import ObserveAIAgent
from src.rag.graph_store import GraphStore
from src.rag.transcript_store import TranscriptStore
from src.rag.causal_pipeline import CausalPipeline
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_deepseek import ChatDeepSeek
from src.logger import setup_logging, log_execution_time

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class ObserveAI:
    @log_execution_time(logger)
    def __init__(self):
        """Initialize Observe AI system"""
        logger.info("Initializing Observe AI system...")

        Config.validate()

        # Initialize LLM (Single Instance)
        # self.llm = ChatGoogleGenerativeAI(
        #     model=Config.GEMINI_MODEL,
        #     api_key=Config.GEMINI_API_KEY,
        #     base_url="https://openrouter.ai/api/v1/chat/completions"
        #     temperature=0.0,
        #     max_tokens=10000
        # )

        self.llm = ChatOpenAI(
            model=Config.GEMINI_MODEL,  # example OpenRouter model name
            api_key=Config.OPEN_ROUTER_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_completion_tokens=10000,
        )
        # self.llm = ChatDeepSeek(
        #     model="deepseek/deepseek-v3.2",
        #     api_key=Config.OPEN_ROUTER_KEY,
        #     # note: use `api_base`, not `base_url`
        #     api_base="https://openrouter.ai/api/v1",
        #     extra_body={"reasoning": {"enabled": False},
        #                 "usage": {"include": True}},  # optional — e.g. enable reasoning tokens
        #     temperature=0.0,
        #     max_tokens=10000,
        # )
        # Initialize RAG components
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.transcript_store = TranscriptStore()
        self.reranker = CrossEncoder(Config.RERANKER_MODEL)
        # Use the custom Retriever class
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            transcript_store=self.transcript_store,
            reranker=self.reranker,
            llm=self.llm
        )
        self.document_processor = DocumentProcessor(
            chunk_strategy="semantic"
        )

        self.causal_pipeline = CausalPipeline(llm=self.llm)

        # Initialize agent
        self.agent = ObserveAIAgent(
            retriever=self.retriever, causal_pipeline=self.causal_pipeline, llm=self.llm, embedder=self.embedding_model)

        logger.info("Observe AI system initialized successfully")

    def clear_memory(self):
        """Clear the agent's memory."""
        if self.agent:
            self.agent.clear_memory()
            logger.info("ObserveAI memory cleared.")

    def query(self, query_text: str) -> Dict:
        """
        Process a user query using the LangGraph agent

        Args:
            query_text: User query

        Returns:
            Agent response with retrieved documents and analysis
        """
        logger.info(f"Processing query: {query_text[:50]}...")

        try:
            result = self.agent.invoke(query_text)
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "success": False
            }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Observe AI - Agentic RAG System")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--query", type=str, help="Process a single query"
    )

    args = parser.parse_args()

    # Initialize system
    app = ObserveAI()

    if args.query:
        logger.info(f"Processing query: {args.query}")
        response = app.query(args.query)
        print("\n=== RESPONSE ===")
        print(f"Response: {response['response']}")
        if response.get("retrieved_docs"):
            print(f"\nRetrieved {len(response['retrieved_docs'])} documents")

    elif args.interactive:
        logger.info("Starting interactive mode...")
        interactive_mode(app)

    else:
        parser.print_help()


def interactive_mode(app: ObserveAI):
    """Interactive query mode"""
    print("\n=== Observe AI - Interactive Mode ===")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("\n> Enter query: ").strip()

            if query.lower() == "exit":
                break

            response = app.query(query)

            print(f"\nResponse: {response['response']}")
            if response.get("retrieved_docs"):
                print(f"Retrieved {len(response['retrieved_docs'])} documents")

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
