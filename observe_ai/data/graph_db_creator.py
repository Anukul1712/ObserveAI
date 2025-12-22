import json
import os
from neo4j import GraphDatabase
from tqdm import tqdm

# --- Configuration ---
# Use environment variables for Docker compatibility, fallback to localhost for local dev
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
AUTH = (USER, PASSWORD)

# Adjust path to be relative to where the script is run (usually project root)
# If running from observe_ai/ folder, data is in data/
INPUT_FILE = os.path.join(os.path.dirname(__file__), "labelled_transcript_bert.json")


class CIDGraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.verify_connection()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Make sure Neo4j Desktop/Docker is running!")
            exit()

    def close(self):
        self.driver.close()

    def clean_db(self):
        """Wipes the DB clean. useful for testing."""
        print("Cleaning Database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_indexes(self):
        """Creates indexes for performance."""
        print("Creating Indexes...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Intent) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:IntentPair) REQUIRE p.uid IS UNIQUE"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)  # type: ignore

    def build_graph(self, transcripts):
        print(f"Ingesting {len(transcripts)} transcripts...")
        with self.driver.session() as session:
            for transcript in tqdm(transcripts):
                self._process_transcript(session, transcript)

if __name__ == "__main__":
    builder = CIDGraphBuilder(URI, AUTH)
    try:
        # Load data
        print(f"Loading data from {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        
        # Build
        builder.clean_db()
        builder.create_indexes()
        builder.build_graph(data)
        print("Graph construction complete!")
    finally:
        builder.close()

    def _process_transcript(self, session, transcript):
        tx_id = transcript.get('transcript_id')
        if not tx_id:
            return

        # 1. Create Conversation Node (The Anchor)
        session.run("""
            MERGE (c:Conversation {id: $id})
            SET c.domain = $domain, c.reason = $reason
        """, {
            "id": tx_id,
            "domain": transcript.get('domain', 'Unknown'),
            "reason": transcript.get('reason_for_call', 'Unknown')
        })

        # 2. Sliding Window to find Patterns
        # We look for the pattern: Agent -> User -> Agent
        # The first two form the "Context Pair"
        # The third is the "Next Intent"

        prev_agent_intent = None
        curr_user_intent = None

        for turn in transcript['conversation']:
            role = turn['speaker'].lower()
            intent_name = turn.get('secondary_intent', 'Unknown')

            # Skip if embedding failed significantly
            if not intent_name or intent_name == "Error":
                print(f"Skipping turn due to bad intent: {turn['turn_id']}")
                continue

            if "agent" in role:
                # --- AGENT TURN ---

                # If we have a full context (PrevAgent + CurrUser), we can draw a transition edge
                if prev_agent_intent and curr_user_intent:
                    self._create_transition(
                        session, prev_agent_intent, curr_user_intent, intent_name, tx_id)

                # Reset for next window: This agent turn becomes the "Previous" for the future
                prev_agent_intent = intent_name
                curr_user_intent = None

            else:
                # --- USER TURN ---
                if prev_agent_intent:
                    curr_user_intent = intent_name
                # If no prev_agent_intent, it's likely the start of the call (User speaks first),
                # we wait for an Agent response to establish flow.

    def _create_transition(self, session, ag_intent, usr_intent, next_ag_intent, tx_id):
        """
        Creates the specific structure from the Paper (Figure 2):
        (AgentIntent) -> [IN_PAIR] -> (IntentPair) <- [IN_PAIR] <- (UserIntent)
                                           |
                                      [NEXT_INTENT] (Weighted)
                                           |
                                           v
                                     (NextAgentIntent)
        """
        query = """
        // 1. Ensure Indivudal Intent Nodes Exist
        MERGE (ag:Intent {name: $ag_name})
        MERGE (usr:Intent {name: $usr_name})
        MERGE (next:Intent {name: $next_name})

        // 2. Create the Intent Pair (Context)
        // UID ensures unique node for this specific combination
        WITH ag, usr, next
        MERGE (pair:IntentPair {uid: $ag_name + '||' + $usr_name})
        SET pair.agent_intent = $ag_name, pair.user_intent = $usr_name
        
        // Link individual intents to the pair
        MERGE (ag)-[:IN_PAIR_WITH]->(pair)
        MERGE (usr)-[:IN_PAIR_WITH]->(pair)

        // 3. Create the Weighted Transition (The Core Logic)
        MERGE (pair)-[r:NEXT_INTENT]->(next)
        ON CREATE SET r.count = 1
        ON MATCH SET r.count = r.count + 1

        // 4. Anchor to History (For Retrieval Phase)
        // We link the 'Next Intent' node to the actual Transcript ID
        // This allows us to fetch the raw text later.
        WITH next
        MATCH (c:Conversation {id: $tx_id})
        MERGE (next)-[:HAS_CONVERSATION]->(c)
        """

        session.run(query, {
            "ag_name": ag_intent,
            "usr_name": usr_intent,
            "next_name": next_ag_intent,
            "tx_id": tx_id
        })


if __name__ == "__main__":
    # Load Data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
            transcripts = data if isinstance(
                data, list) else data.get("transcripts", [])
    except FileNotFoundError:
        print("File not found! Run Phase 2 first.")
        exit()

    builder = CIDGraphBuilder(URI, AUTH)

    # Optional: Clear DB to start fresh
    builder.clean_db()
    builder.create_indexes()

    # Run
    builder.build_graph(transcripts)
    builder.close()
    print("Done! Graph built.")
