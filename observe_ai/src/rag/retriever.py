import logging
from typing import List, Dict, Optional, Any
from sentence_transformers import CrossEncoder
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.graph_store import GraphStore
from src.rag.query_classifier import QueryClassifier
from src.rag.transcript_store import TranscriptStore
from src.models import RetrievalResult
from src.config import Config
from src.logger import log_execution_time

logger = logging.getLogger(__name__)


class Retriever:
    """
    SOTA Dual-Retrieval Engine with Cross-Encoder Reranking.
    """

    def __init__(
        self,
        top_k: int = 50,  # High recall (Step 1)
        rerank_top_n: int = 10,  # High precision (Step 2)
        alpha: float = 0.1,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        transcript_store: Optional[TranscriptStore] = None,
        reranker: Optional[CrossEncoder] = None,
        llm=None
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.graph_store = graph_store or GraphStore()
        self.transcript_store = transcript_store or TranscriptStore()
        self.reranker = reranker
        self.query_classifier = QueryClassifier(llm=llm)

        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.alpha = alpha

    def _get_initial_candidates(self, query: str, target_intent: str | None = None) -> List[Dict]:
        """Helper to get initial candidates before reranking."""
        # Step 1: Query Analysis
        if target_intent is None:
            target_intent = self.query_classifier.classify(query)

        # Step 2: High Recall Semantic Search
        semantic_results = self.vector_store.search(query, k=self.top_k)

        # Step 3: Intent Graph Fusion
        intent_weights = self.graph_store.get_related_intents(target_intent)

        fused_candidates = []
        for res in semantic_results:
            turn_intent = res["metadata"].get(
                "secondary_intent") or res["metadata"].get("intent") or "Unknown"
            graph_score = intent_weights.get(turn_intent, 0.0)

            # Convert distance to similarity
            raw_score = res["score"]
            sim_score = 1 / (1 + raw_score)

            # Alpha fusion
            fusion_score = (self.alpha * graph_score) + \
                ((1 - self.alpha) * sim_score)

            fused_candidates.append({
                "doc": res,
                "fusion_score": fusion_score,
                "text": res["content"],
                "intent": turn_intent,
                "transcript_id": res["metadata"].get("transcript_id"),
                "turn_index": res["metadata"].get("turn_index")
            })
        return fused_candidates

    @log_execution_time(logger)
    def retrieve_batch(self, queries: List[str]) -> List[RetrievalResult]:
        """
        Batch retrieval for multiple queries with optimized cross-encoder usage.
        """
        # 1. Gather candidates for all queries
        batch_tasks = []
        for q in queries:
            cands = self._get_initial_candidates(q)
            batch_tasks.append((q, cands))

        return self._process_batch_results(batch_tasks, queries)

    async def aretrieve_batch(self, queries: List[str]) -> List[RetrievalResult]:
        """
        Async Batch retrieval for multiple queries with parallel intent classification.
        """
        # 1. Parallel Intent Classification
        intents = await self.query_classifier.aclassify_batch(queries)

        # 2. Gather candidates for all queries (Synchronous part)
        batch_tasks = []
        for q, intent in zip(queries, intents):
            cands = self._get_initial_candidates(q, target_intent=intent)
            batch_tasks.append((q, cands))

        return self._process_batch_results(batch_tasks, queries)

    def _process_batch_results(self, batch_tasks, queries):
        """
        Shared logic for reranking and processing candidates.
        """
        # 2. Prepare batch for reranker
        all_pairs = []
        # Map: global_index -> (query_index, candidate_index)
        map_back = []

        for q_idx, (q, cands) in enumerate(batch_tasks):
            for c_idx, cand in enumerate(cands):
                all_pairs.append([q, cand["text"]])
                map_back.append((q_idx, c_idx))

        # 3. Batch Rerank
        if self.reranker and all_pairs:
            logger.info(
                f"Batch reranking {len(all_pairs)} pairs across {len(queries)} queries...")
            scores = self.reranker.predict(all_pairs)

            # Assign scores back
            for global_idx, score in enumerate(scores):
                q_idx, c_idx = map_back[global_idx]
                batch_tasks[q_idx][1][c_idx]["final_score"] = float(score)
        else:
            # Fallback
            for q_idx, (q, cands) in enumerate(batch_tasks):
                for cand in cands:
                    cand["final_score"] = cand["fusion_score"]

        # 4. Process per query (Sort, Slice, Expand) & Deduplicate globally
        final_results_map = {}  # Key by (transcript_id, turn_index)

        for q, cands in batch_tasks:
            # Sort and Slice per query
            cands.sort(key=lambda x: x["final_score"], reverse=True)
            top_results = cands[:self.rerank_top_n]

            for item in top_results:
                key = (item["transcript_id"], item["turn_index"])
                if key not in final_results_map:
                    # Expand context (only once per unique doc)
                    expanded_content = self.transcript_store.get_window_content(
                        item["transcript_id"], item["turn_index"], window_radius=3
                    )

                    result = RetrievalResult(
                        transcript_id=item["transcript_id"],
                        turn_index=item["turn_index"],
                        content=expanded_content,
                        intent=item["intent"],
                        score=item["final_score"],
                        metadata=item["doc"]["metadata"]
                    )
                    final_results_map[key] = result
                else:
                    # Update score if higher
                    if item["final_score"] > final_results_map[key].score:
                        final_results_map[key].score = item["final_score"]

        return list(final_results_map.values())

    @log_execution_time(logger)
    def retrieve_dual(self, query: str) -> List[RetrievalResult]:
        """
        Executes the SOTA retrieval pipeline: 
        1. Classify Intent
        2. Semantic Search (High Recall)
        3. Graph Fusion
        4. Cross-Encoder Reranking (High Precision)
        5. Context Expansion
        """
        return self.retrieve_batch([query])
