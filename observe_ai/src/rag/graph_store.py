import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from src.config import Config
from src.models import IntentScore

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Neo4j-based Graph Store for Intent Graph Retrieval (Pathway B).
    """

    def __init__(self, uri: Optional[str] = None, auth: Optional[tuple] = None):
        uri = uri or Config.NEO4J_URI
        user = Config.NEO4J_USER
        password = Config.NEO4J_PASSWORD

        if auth is None:
            auth = (user, password)

        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connection()

    def verify_connection(self):
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j GraphStore.")

    def close(self):
        self.driver.close()

    def get_related_intents(self, target_intent: str, limit: int = 5) -> Dict[str, float]:
        """
        Finds frequent preceding (causes) and succeeding (effects) intents for a target intent.
        Returns a dictionary mapping intent_name -> normalized_frequency_score.
        """
        # We want to find:
        # 1. Preceding: (prev)-[r:NEXT_INTENT]->(target)
        # 2. Succeeding: (target)-[r:NEXT_INTENT]->(next)

        # Note: In the graph construction:
        # (pair:IntentPair)-[:NEXT_INTENT]->(next:Intent)
        # (prev:Intent)-[:IN_PAIR_WITH]->(pair)
        # So the path is (prev)-[:IN_PAIR_WITH]->(pair)-[:NEXT_INTENT]->(target)

        # Let's look at preceding intents (Causes)
        # We want 'prev' such that 'prev' leads to 'target'.
        # Pattern: (prev)-[:IN_PAIR_WITH]->(pair)-[:NEXT_INTENT]->(target)

        query_preceding = """
        MATCH (prev:Intent)-[:IN_PAIR_WITH]->(pair:IntentPair)-[r:NEXT_INTENT]->(target:Intent {name: $intent})
        RETURN prev.name as intent, sum(r.count) as weight
        ORDER BY weight DESC
        LIMIT $limit
        """

        # Succeeding intents (Effects)
        # Pattern: (target)-[:IN_PAIR_WITH]->(pair)-[:NEXT_INTENT]->(next)
        query_succeeding = """
        MATCH (target:Intent {name: $intent})-[:IN_PAIR_WITH]->(pair:IntentPair)-[r:NEXT_INTENT]->(next:Intent)
        RETURN next.name as intent, sum(r.count) as weight
        ORDER BY weight DESC
        LIMIT $limit
        """

        intent_scores: Dict[str, float] = {}

        with self.driver.session() as session:
            # Fetch Preceding
            result_prev = session.run(
                query_preceding, intent=target_intent, limit=limit)
            for record in result_prev:
                intent_scores[record["intent"]] = intent_scores.get(
                    record["intent"], 0.0) + float(record["weight"])

            # Fetch Succeeding
            result_next = session.run(
                query_succeeding, intent=target_intent, limit=limit)
            for record in result_next:
                intent_scores[record["intent"]] = intent_scores.get(
                    record["intent"], 0.0) + float(record["weight"])

        # Also include the target intent itself with a high score?
        # The paper says "Retrieve turns linked to these high-probability intent paths."
        # The target intent is definitely relevant.
        # Let's normalize the scores.

        if not intent_scores:
            return {target_intent: 1.0}

        max_weight = max(intent_scores.values()) if intent_scores else 1.0
        normalized_scores = {k: v / max_weight for k,
                             v in intent_scores.items()}

        # Ensure target intent is included with max score (1.0) as it's the direct match
        normalized_scores[target_intent] = 1.0

        return normalized_scores

    def get_conversations_for_intent(self, intent_name: str, limit: int = 10) -> List[str]:
        """
        Get conversation IDs associated with an intent.
        """
        query = """
        MATCH (i:Intent {name: $intent})-[:HAS_CONVERSATION]->(c:Conversation)
        RETURN c.id as id
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, intent=intent_name, limit=limit)
            return [record["id"] for record in result]
