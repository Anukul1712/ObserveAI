import logging
import json
import os
import re
import hashlib
import numpy as np
from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import START, END, StateGraph

from src.config import Config
from src.rag.retriever import Retriever
from src.rag.causal_pipeline import CausalPipeline
from src.models import RetrievalResult
from src.logger import log_execution_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------
# Memory store (in-memory)
# ---------------------------


@dataclass
class MemoryEntry:
    id: str
    summary: str             # 1-3 lines
    # placeholder, not used — kept for extension
    embedding: Optional[np.ndarray]
    documents: List[RetrievalResult]
    query_ref: str
    outcome_flag: Optional[bool] = None


class MemoryStore:
    """Lightweight in-memory memory store.
    No timestamps. No tags. Minimal API: add, query (similarity), link_documents.
    Similarity is computed by embedding cosine similarity.
    """

    def __init__(self, embedder):
        self._store: Dict[str, MemoryEntry] = {}
        self.embedder = embedder

    def clear(self):
        """Clear all memory entries."""
        self._store.clear()

    def _make_id(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def add(self, summary: str, documents: List[RetrievalResult], query_ref: str,
            outcome_flag: Optional[bool] = None) -> MemoryEntry:
        mid = self._make_id(summary + query_ref)
        # Compute embedding
        embedding = self.embedder.embed_query(summary)
        entry = MemoryEntry(id=mid, summary=summary, embedding=embedding,
                            documents=documents, query_ref=query_ref,
                            outcome_flag=outcome_flag)
        self._store[mid] = entry
        return entry

    def link_documents(self, mem_id: str, documents: List[RetrievalResult]):
        if mem_id in self._store:
            existing = self._store[mem_id].documents
            combined = existing + documents
            self._store[mem_id].documents = combined

    def all(self) -> List[MemoryEntry]:
        return list(self._store.values())

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        if vec1 is None or vec2 is None:
            return 0.0

        v1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
        v2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k memory entries with embedding cosine similarity."""
        query_embedding = self.embedder.embed_query(query_text)
        scores = []
        for m in self.all():
            if m.embedding is None:
                score = 0.0
            else:
                score = self._cosine_similarity(query_embedding, m.embedding)

            scores.append({"id": m.id, "score": score, "summary": m.summary,
                        "documents": m.documents, "outcome_flag": m.outcome_flag})
        # sort by score desc
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]




# ---------------------------
# ObserveAIAgent (patched)
# ---------------------------

class AgentState(TypedDict):
    query: str
    documents: List[RetrievalResult]
    generation: str
    critique_pass: bool
    critique_reason: str
    retry_count: int
    rewrites: Optional[List[str]]
    mem_context: Optional[List[Dict[str, Any]]]
    routing_policy: Optional[str]
    critique_failed_checks: Optional[List[str]]
    response_type: Optional[str]


class ObserveAIAgent:
    def __init__(self, retriever: Retriever, causal_pipeline: CausalPipeline, llm, embedder):
        self.retriever = retriever
        self.causal_pipeline = causal_pipeline

        # Main generator - Low temp for analytical precision
        self.llm = llm
        self.embedder = embedder
        self.memory = MemoryStore(embedder=self.embedder)
        # token budget defaults (context size per request)
        self.model_context_budget = 8192  # replace with model limit if known
        self.prompt_budget_fraction_for_memory = 0.15
        self.prompt_budget_fraction_for_docs = 0.50

        self.graph = self._build_graph()

    def clear_memory(self):
        """Clear the agent's memory."""
        self.memory.clear()
        logger.info("Agent memory cleared.")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("query_rewrite", self.query_rewrite_node)
        workflow.add_node("memory_router", self.memory_router_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_causal_node)
        workflow.add_node("critique", self.critique_node)
        workflow.add_node("summary_gen", self.summary_gen_node)

        workflow.add_edge(START, "query_rewrite")
        workflow.add_edge("query_rewrite", "memory_router")
        workflow.add_edge("memory_router", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "critique")
        # workflow.add_edge("generate", "summary_gen")

        # if critique fails, go to generate (retry)
        workflow.add_conditional_edges(
            "critique",
            self.decide_next_step,
            {
                "retry": "generate",
                "continue": "summary_gen"
            })
        workflow.add_edge("summary_gen", END)

        return workflow.compile()

    # ---------------------------
    # Node: query_rewrite
    # ---------------------------
    def query_rewrite_node(self, state: AgentState) -> Dict:
        """Generate multiple rewrites. 4-7 for initial, <=2 for follow-ups.
        Single LLM call returns newline-separated rewrites. Minimize API calls.
        """
        query = state["query"]

        # Check memory to determine if this is the first turn
        all_memories = self.memory.all()
        is_initial = len(all_memories) == 0

        # Template: ask LLM to produce rewrites in a JSON list for robust parsing
        max_initial = 5
        min_initial = 2
        max_followup = 2
        min_followup = 1

        if is_initial:
            target = max_initial
            min_count = min_initial
        else:
            target = max_followup
            min_count = min_followup

        # Check for recent memory to help with follow-ups
        recent_context = ""
        if all_memories:
            # Get most recent memory (assuming insertion order)
            last_mem = all_memories[-1]
            recent_context = f"\nContext from previous turn: {last_mem.summary}\n"

        prompt = PromptTemplate(
            template="""You are a query rewriter for an evidence-based causal analysis system.
Input query: {query}
{recent_context}

Return a JSON array called "rewrites" containing between {min_count} and {target} candidate rewrites of the input. 
Each rewrite should be a concise paraphrase or focused variant aimed at retrieving evidence:
- include causal phrasing ("why", "causes", "what leads to")
- include example-seeking phrasing ("examples when X did not happen")
- include counterfactual phrasing ("how could Y have been avoided")
- include narrow entity-driven phrasing if applicable

Return ONLY valid JSON, e.g.:
{{ "rewrites": ["...", "..."] }}
""",
            input_variables=["query", "min_count", "target", "recent_context"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        # try:
        parsed = chain.invoke(
            {"query": query, "min_count": min_count, "target": target, "recent_context": recent_context})
        rewrites = parsed.get("rewrites", [])
        if not isinstance(rewrites, list) or len(rewrites) == 0:
            raise ValueError("Empty rewrites")
        # ensure cap
        rewrites = rewrites[:target]
        # except Exception as e:
        #     logger.warning(
        #         f"Query rewrite failed: {e}. Falling back to original query as single rewrite.")
        #     rewrites = [query]

        logger.info("Rewritten queries:")
        for r in rewrites:
            logger.info(f"- {r}")

        logger.info(f"Rewrites generated: {len(rewrites)}")
        return {"rewrites": rewrites}

    # ---------------------------
    # Node: memory_router
    # ---------------------------
    def memory_router_node(self, state: AgentState) -> Dict:
        """Select memory summaries and decide routing policy.
        - No recency in scoring.
        - Use simple relative ranking (percentiles) based on overlap similarity.
        """
        query = state["query"]
        rewrites: List[str] = state.get("rewrites") or [query]
        # choose the canonical rewrite to compute similarity: use first rewrite
        canonical = rewrites[0]

        # First conversation: skip memory logic entirely, always retrieve
        if len(self.memory.all()) == 0:
            logger.info(
                "First conversation detected (no memories). Routing: retrieve")
            return {"mem_context": [], "routing_policy": "retrieve"}

        mem_hits = self.memory.query(canonical, top_k=10)  # fetch up to 10
        logger.info(f"Memory candidates: {len(mem_hits)}")

        # Build score distribution
        scores = [m["score"] for m in mem_hits]

        # Define meaningful similarity threshold (0.2 = 20% token overlap minimum)
        MEANINGFUL_THRESHOLD = 0.2

        # pick top memories with meaningful scores (above threshold)
        selected = []
        for m in mem_hits:
            if m["score"] >= MEANINGFUL_THRESHOLD and len(selected) < 5:
                selected.append(m)

        # Decide routing policy:
        # If there's at least one strong memory (score >= 0.5) and the query looks like follow-up, prefer memory_only
        # Otherwise hybrid if some memories exist, else retrieve
        routing = "retrieve"
        if selected:
            top_score = selected[0]["score"]
            # Strong memory: score >= 0.5 (50% token overlap)
            if top_score >= 0.5:
                # further heuristics: if the query looks like follow-up (short) then memory_only
                if len(state["query"].split()) <= 8:
                    routing = "memory_only"
                else:
                    routing = "hybrid"
            else:
                # Weak memories (0.2-0.5): always hybrid to supplement with fresh retrieval
                routing = "hybrid"
        else:
            # No meaningful memories found
            routing = "retrieve"

        logger.info(
            f"Memory routing decision: {routing} (selected {len(selected)} memories)")
        return {"mem_context": selected, "routing_policy": routing}

    # ---------------------------
    # Node: retrieve
    # ---------------------------

    def retrieve_node(self, state: AgentState) -> Dict:
        """Fetch documents according to routing policy.
        - memory_only: use documents from memory
        - retrieve: run dual retriever for ALL rewrites to maximize recall
        - hybrid: combine both
        Also cap per-doc spans to token limits.
        """
        routing = state.get("routing_policy", "retrieve")
        rewrites: List[str] = state.get("rewrites") or [state["query"]]

        mem_context = state.get("mem_context") or []

        retrieved_docs = []

        if routing == "memory_only":
            # Use documents directly from memory
            seen_ids = set()
            for m in mem_context:
                for d in m.get("documents", []):
                    # Deduplicate by transcript_id + turn_index
                    key = (d.transcript_id, getattr(d, 'turn_index', -1))
                    if key not in seen_ids:
                        retrieved_docs.append(d)
                        seen_ids.add(key)
            logger.info(
                f"Retrieved {len(retrieved_docs)} docs from memory (memory_only).")
        elif routing == "retrieve":
            logger.info(
                f"Running full dual retrieval (retrieve) on {len(rewrites)} queries.")
            try:
                retrieved_docs = self.retriever.retrieve_batch(rewrites)
            except Exception as e:
                logger.warning(f"Batch retrieval failed: {e}")
        else:  # hybrid
            logger.info(
                "Hybrid: combine memory docs and narrow retrieval.")
            seen_ids = set()
            # Memory docs first
            for m in mem_context:
                for d in m.get("documents", []):
                    key = (d.transcript_id, getattr(d, 'turn_index', -1))
                    if key not in seen_ids:
                        retrieved_docs.append(d)
                        seen_ids.add(key)

            # New retrieval using all rewrites
            try:
                more = self.retriever.retrieve_batch(rewrites)
                for d in more:
                    key = (d.transcript_id, getattr(d, 'turn_index', -1))
                    if key not in seen_ids:
                        retrieved_docs.append(d)
                        seen_ids.add(key)
            except Exception as e:
                logger.warning(f"Hybrid retrieval extra step failed: {e}")

        logger.info(
            f"Retrieved {len(retrieved_docs)} docs")
        return {"documents": retrieved_docs}

    # ---------------------------
    # Node: generate_causal_node
    # ---------------------------
    def generate_causal_node(self, state: AgentState) -> Dict:
        """Prepend memory summary block (3-5 lines) and include retrieved documents labeled CASE 1..n.
        Minimal extra LLM calls; keep prompt template stable.
        """
        logger.info("---GENERATE CAUSAL ANALYSIS---")
        query = state["query"]
        docs = state.get("documents", [])
        mem_context = state.get("mem_context", []) or []

        # 1. Classify Intent (Lightweight)
        classification_prompt = PromptTemplate(
            template="""You are an intent classifier. Your job is to determine whether the user's query requires:
• "analysis" → multi-step reasoning, interpretation, inference, synthesis, diagnosis, or causal explanation.
• "conversational" → direct retrieval, listing, quoting, clarifying, or lightweight responses with no reasoning.

You must classify based on the user's intended depth of thinking, not on specific keywords.

Definitions:

1. "analysis":
The user expects reasoning that goes beyond direct facts. This includes:
- Interpreting meaning, motives, sentiment, or behavior
- Explaining causes, drivers, or mechanisms
- Identifying trends, anomalies, transitions, or hidden patterns
- Synthesizing information across multiple parts of a conversation
- Making inferences not explicitly stated
- Providing diagnostic or insight-generating explanations

Examples:
    • “Identify where the first misunderstanding occurs and explain why.”
    • “Analyze how customer sentiment shifts throughout the call.”
    • “What patterns reveal the root cause of long hold times?”

2. "conversational":
The user expects a direct answer, retrieval, or simple response with no reasoning. This includes:
- Listing items
- Fetching examples or cases
- Quoting or pointing to specific conversation segments
- Asking clarifying questions
- Greetings or acknowledgments

Examples:
    • “Show an example of an angry customer.”
    • “List the case IDs.”
    • “What did the agent say in Case 5?”
    • “Hello”, “Thanks”

Follow-Up Rule (Important):
- If the current query is a follow-up to a previous query, you must classify it as **"conversational"**, regardless of phrasing.
- A query counts as a follow-up when it:
• Refers back to something previously asked  
• Seeks continuation, refinement, or narrowing of the prior output  
• Requests “more”, “continue”, “clarify”, “give an example of that”, etc.

Disambiguation Rules:
- If the user asks for explanation, interpretation, or reasoning → "analysis".
- If the user asks for retrieval, example, listing, or clarification → "conversational".
- If the query mixes both, choose based on the *expected output*, not the keywords.
- If the query is a follow-up, always choose "conversational".

Query: {query}

Return ONLY: "analysis" or "conversational".
""",
            input_variables=["query"]
        )
        try:
            classifier_chain = classification_prompt | self.llm | StrOutputParser()
            response_type = classifier_chain.invoke(
                {"query": query}).strip().lower()
        except Exception as e:
            logger.warning(
                f"Classification failed: {e}. Defaulting to analysis.")
            response_type = "analysis"

        if "conversational" in response_type:
            response_type = "conversational"
        else:
            response_type = "analysis"

        logger.info(f"Generation Mode: {response_type}")

        # Build memory block: use up to top 3 summaries, each 1-3 lines
        mem_block = ""
        if mem_context:
            chosen = mem_context[:3]
            mem_block = "### PRIOR ANALYSIS SUMMARY (condensed):\n"
            for m in chosen:
                summary = m["summary"].strip()
                mem_block += f"- {summary}\n"

        # Build context_str from distinct cases
        context_str = ""
        for i, d in enumerate(docs):
            context_str += f"\n--- CASE {i+1} (ID: {d.transcript_id}) ---\n"
            context_str += f"Intent Label: {getattr(d, 'intent', 'N/A')}\n"
            content = "\n".join(d.content) if isinstance(
                d.content, list) else str(d.content)
            context_str += "Segment:\n" + content + "\n"

        critique_reason = state.get("critique_reason", "")
        critique_context = ""
        if critique_reason:
            critique_context = f"\n### PREVIOUS ATTEMPT CRITIQUE:\nThe previous answer failed because: {critique_reason}. Please address this in your new response.\n"

        # Run causal pipeline first (we still want quantitative causal signals)
        transcript_ids = [d.transcript_id for d in docs]
        rewrites: List[str] = state.get("rewrites") or []
        causal_results = self.causal_pipeline.run(
            rewrites[0], transcript_ids) if transcript_ids else ""

        if response_type == "conversational":
            prompt = PromptTemplate(
                template="""You are a helpful AI assistant.
You have retrieved {num_docs} conversation segments.
The user has asked a follow-up question or requested examples.
Answer the user's query directly and helpfully using the provided context.
You do NOT need to follow a strict report format.

PREVIOUSLY:
{mem_block}
{critique_context}

QUERY: "{query}"

### CONTEXT (Distinct Cases):
{context_str}

### CAUSAL ANALYSIS (Optional Context):
{causal_results}

Instructions:
1. Answer the query directly.
2. Use specific examples from the cases (cite Case IDs like "Case 1").
3. Be concise but comprehensive.
4. If the user asks for a list, provide a list.
""",
                input_variables=["num_docs", "mem_block", "critique_context",
                                "query", "context_str", "causal_results"]
            )
        else:
            # STRICT ANALYSIS PROMPT
            prompt = PromptTemplate(
                template="""
You are a Lead Data Analyst specializing in causal inference and customer interaction analysis.
You have retrieved {num_docs} distinct conversation segments from a database.
Your task is to SYNTHESIZE a generalized answer to the query by finding common patterns across these cases.
PREVIOUSLY:
{mem_block}
{critique_context}

QUERY: "{query}"

### INPUT DATA (Distinct Cases):
{context_str}

### CAUSAL ANALYSIS RESULTS:
{causal_results}

The causal analysis above provides quantitative evidence showing:
- **Interventions**: Variables that were hypothetically changed based on the query
- **Targets**: Outcome variables measured before and after the intervention
- **Delta values**: The numerical impact of the intervention (positive delta = increase, negative = decrease)

### ANALYSIS METHODOLOGY:
1. **Pattern Matching**: Do not treat these as one story. Look for recurring behaviors across Case 1, Case 2, etc.
2. **Synthesize the Root Cause**: What is the common thread? (e.g., "In 80% of cases, agents failed to authenticate").
4. **Counterfactuals**: Apply the "What If" logic to the *pattern*. (e.g., "").
3. **Quantitative Reasoning**: Use the delta values from the causal analysis to support your conclusions. For example:
    - If empathy increased by 0.15 and customer_satisfaction increased by 0.23, explain this relationship
    - If resolution_time decreased by -0.45, highlight this improvement
    - Compare magnitudes across different cases to identify strongest effects
    - DO NOT fabricate numbers; only use what is provided
    - DO NOT OUTPUT NUMBERS. ONLY INTERPRET THEM.
4. **Cite Examples**: Use the Cases to prove your pattern (e.g., "seen in Case 1 and Case 3"). ONLY specify the IDS in *Citations* section.
5. **Counterfactuals with Evidence**: Apply the "What If" logic supported by numerical evidence from the causal model. 
    (e.g., "If agents had followed protocol X, these escalations might have been avoided.").
### OUTPUT FORMAT:

**1. Executive Summary**
*A high-level synthesis of why this event occurs based on the dataset.*

**2. Key Causal Patterns**
* **Pattern A:** [Description of behavior]
    * *Evidence:* "In Case 1, the customer said... In Case 3, the agent..."
    * *Effect:* [Result of this behavior]
* **Pattern B:** [Secondary behavior if any]
    * *Evidence:* ...

**3. Counterfactual Recommendation**
* *Observation:* "We observed that when agents [Action X], outcomes were negative."
* *Hypothesis:* "If agents were trained to [Alternative Action Y], we expect a reduction in these events."

**4. Citations**
* List the Case IDs referenced.
Case 1: ID xxx
Case 2: ID yyy
""",
                input_variables=[
                    "num_docs", "mem_block", "critique_context", "query", "context_str", "causal_results"]
            )

        # Prepare chain and invoke once
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "mem_block": mem_block,
            "critique_context": critique_context,
            "num_docs": len(docs),
            "query": query,
            "context_str": context_str,
            "causal_results": causal_results
        })

        logger.info("Generation complete.")
        return {"generation": response, "causal_results": causal_results, "response_type": response_type}

    # ---------------------------
    # Node: critique_node (binary)
    # ---------------------------
    def critique_node(self, state: AgentState) -> Dict:
        """Binary pass/fail with checklist. If fail, return reason and failed_checks list.
        Uses a single LLM call to evaluate.
        """
        logger.info("---CRITIQUE ANSWER---")
        question = state["query"]
        generation = state["generation"]
        causal_results = state.get("causal_results", "")
        response_type = state.get("response_type", "analysis")

        if response_type == "conversational":
            # Relaxed critique for conversational queries
            prompt = PromptTemplate(
                template="""You are a helpful assistant evaluator.
Evaluate the following answer for the QUERY and mark PASS or FAIL.
Return JSON with keys: pass ("yes" or "no"), failed_checks (array), reason (short string).

Query: {question}
Generated Answer: {generation}

CHECKLIST (all required to PASS):
1) Relevance: Does the answer directly address the user's question or request?
2) No Fabrication: Does the answer avoid inventing facts?
3) Politeness: Is the tone helpful and polite?

Return only valid JSON.
""",
                input_variables=["question", "generation"]
            )
            chain = prompt | self.llm | JsonOutputParser()
            try:
                result = chain.invoke(
                    {"question": question, "generation": generation})
                p = result.get("pass", "no")
                passed = True if str(p).lower() == "yes" else False
                failed_checks = result.get("failed_checks", [])
                reason = result.get("reason", "") if not passed else ""
            except Exception as e:
                logger.warning(
                    f"Critique parsing failed: {e}. Defaulting to PASS for conversational.")
                passed = True
                failed_checks = []
                reason = ""
        else:
            # Strict critique for analysis
            prompt = PromptTemplate(
                template="""You are a strict Quality Control Editor. Evaluate the following answer for the QUERY and mark PASS or FAIL.
Return JSON with keys: pass ("yes" or "no"), failed_checks (array), reason (short string - REQUIRED if pass == "no").

Query: {question}
Causal Results: {causal_results}
Generated Analysis: {generation}

CHECKLIST (all required to PASS):
1) Synthesis: Does the analysis identify a pattern across at least two distinct cases, not just summarizing one call?
2) Citations: Does the analysis cite the Case IDs for evidence?
3) Causality Mechanism: Does it explain how the pattern leads to the outcome (a mechanism)?
4) No Fabrication: Does the answer avoid inventing numbers or facts not present in Causal Results?
5) Actionability: If the query asks for recommendations or counterfactuals, does it provide at least one?

Return only valid JSON, e.g.:
{{ "pass": "yes", "failed_checks": [], "reason": "" }}
or
{{ "pass": "no", "failed_checks": ["Causality Mechanism","Citations"], "reason": "Missing citations and causal explanation." }}
""",
                input_variables=["question", "causal_results", "generation"]
            )

            chain = prompt | self.llm | JsonOutputParser()
            try:
                result = chain.invoke(
                    {"question": question, "generation": generation, "causal_results": causal_results})
                p = result.get("pass", "no")
                passed = True if str(p).lower() == "yes" else False
                failed_checks = result.get("failed_checks", [])
                reason = result.get("reason", "") if not passed else ""
            except Exception as e:
                logger.warning(
                    f"Critique parsing failed: {e}. Defaulting to FAIL with reason.")
                passed = False
                failed_checks = ["Critique parsing error"]
                reason = "Critique parsing failed."

        logger.info(
            f"Critique pass={passed}; failed_checks={failed_checks}; reason={reason}")

        # Increment retry count
        new_retry_count = state.get("retry_count", 0) + 1

        return {"critique_pass": passed, "critique_reason": reason, "critique_failed_checks": failed_checks, "retry_count": new_retry_count}

    # ---------------------------
    # Node: summary_gen_node
    # ---------------------------
    def summary_gen_node(self, state: AgentState) -> Dict:
        """Condense final generation into 1-3 sentence summary and store memory if critique passed.
        Also implement doc_id linking when reused evidence is detected.
        """
        logger.info("---SUMMARY GENERATION & MEM STORE---")
        generation = state["generation"]
        passed = state.get("critique_pass", False)
        docs = state.get("documents", [])

        if not passed:
            logger.info("Not storing to memory because critique FAILED.")
            return {"memory_stored": False}

        # Create summary via LLM (single short call). Keep concise.
        prompt = PromptTemplate(
            template="""Analyze the following response and create a structured memory summary.
The summary must be concise but semantically rich to support future retrieval.

Input Analysis:
{generation}

Instructions:
1. **Summary**: Write a 2-3 sentence abstract of the core insight or causal finding.
2. **Keywords**: List 5-7 high-value keywords (topics, entities, metrics).
3. **Cases**: List the Case IDs referenced in the analysis (e.g., "Case 1", "Case 10").

Output Format (Plain Text):
Summary: Your summary here
Keywords: Keyword1, Keyword2, ...
Cases: Case ID1, Case ID2, ...
""",
            input_variables=["generation"]
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            summary = chain.invoke({"generation": generation})
        except Exception as e:
            logger.warning(
                f"Summary generation failed: {e}. Falling back to first 200 chars.")
            summary = (generation[:200] + "...") if generation else ""

        # store memory entry
        mem_entry = self.memory.add(
            summary=summary, documents=docs, query_ref=state["query"])
        logger.info(
            f"Stored MemoryEntry id={mem_entry.id} with {len(docs)} docs.")
        return {"memory_stored": True, "memory_id": mem_entry.id}

    # ---------------------------
    # Conditional logic: decide_next_step
    # ---------------------------
    def decide_next_step(self, state: AgentState) -> str:
        """If critique fails, retry generation. Else continue to summary_gen."""
        passed = state.get("critique_pass", False)
        retry_count = state.get("retry_count", 0)
        MAX_RETRIES = 3

        if not passed:
            if retry_count >= MAX_RETRIES:
                logger.warning(
                    f"Max retries ({MAX_RETRIES}) reached. Proceeding despite failure.")
                return "continue"

            logger.info("Critique failed -> retry")
            return "retry"
        return "continue"

    # ---------------------------
    # Invocation
    # ---------------------------
    @log_execution_time(logger)
    def invoke(self, query: str) -> dict:
        initial_state = {
            "query": query,
            "documents": [],
            "generation": "",
            "critique_pass": False,
            "critique_reason": "",
            "retry_count": 0,
            "rewrites": None,
            "mem_context": None,
            "routing_policy": None,
            "critique_failed_checks": None
        }

        result = self.graph.invoke(initial_state)

        # Collect outputs and print simple flow logs
        logger.info("----- REQUEST FLOW SUMMARY -----")
        logger.info(
            f"Final generation length: {len(result.get('generation',''))} chars")
        logger.info(f"Critique pass: {result.get('critique_pass')}")
        logger.info(f"Critique reason: {result.get('critique_reason')}")
        logger.info(f"Memory stored: {len(self.memory.all())}")
        # print decisions (if present)
        if "routing_policy" in result:
            logger.info(f"Routing: {result['routing_policy']}")
        if "rewrites" in result:
            logger.info(f"Used rewrites count: {len(result['rewrites'])}")

        # return structured summary
        return {
            "response": result.get("generation", ""),
            "retrieved_results": result.get("documents", []),
            "final_pass": result.get("critique_pass"),
            "critique_reason": result.get("critique_reason"),
            "memory_stored": result.get("memory_stored"),
            "iterations": result.get("retry_count")
        }
