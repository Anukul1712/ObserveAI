from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Turn(BaseModel):
    turn_id: str
    turn_index: int
    speaker: str
    text: str
    primary_intent: Optional[str] = None
    secondary_intent: Optional[str] = None


class Transcript(BaseModel):
    transcript_id: str
    domain: Optional[str] = None
    global_intent: Optional[str] = None
    reason_for_call: Optional[str] = None
    time_of_interaction: Optional[str] = None
    conversation: List[Turn]


class RetrievalResult(BaseModel):
    transcript_id: str
    turn_index: int
    content: List[str]
    metadata: Dict[str, Any]
    score: float
    graph_score: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0
    intent: Optional[str] = None


class IntentScore(BaseModel):
    intent: str
    score: float
