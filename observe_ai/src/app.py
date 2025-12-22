# isort: skip_file
# fmt: off
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import logging
import sys
from pathlib import Path

# Add the parent directory (observe_ai) to sys.path so we can import 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import ObserveAI
from logger import setup_logging, log_execution_time
# fmt: on


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instance to hold the initialized system
observe_system: Optional[ObserveAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global observe_system
    try:
        logger.info("Initializing Observe AI system...")
        observe_system = ObserveAI()
        logger.info("Observe AI system initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Observe AI system: {e}")
        # We might want to raise an error here to prevent the app from starting if initialization is critical
        raise e
    yield


app = FastAPI(title="Observe AI API", lifespan=lifespan)


class InitResponse(BaseModel):
    status: str
    message: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


class MultiQueryRequest(BaseModel):
    queries: List[str]


class MultiQueryResponse(BaseModel):
    results: List[str]


class InitRequest(BaseModel):
    reset_memory: bool = False


@app.post("/reset_memory", response_model=InitResponse)
async def reset_memory():
    """
    Clear the agent's memory without re-initializing the whole system.
    """
    global observe_system
    if observe_system is None:
        raise HTTPException(
            status_code=500, detail="System not initialized.")

    try:
        observe_system.clear_memory()
        return InitResponse(status="success", message="Memory cleared successfully.")
    except Exception as e:
        logger.error(f"Memory reset failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Memory reset failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def single_query(request: QueryRequest):
    """
    Process a single query.
    """
    global observe_system
    if observe_system is None:
        raise HTTPException(
            status_code=500, detail="System not initialized.")

    try:
        observe_system.clear_memory()
        result = observe_system.query(request.query)
        # The result from ObserveAI.query is a dict. We map it to our response model.
        return QueryResponse(
            response=result.get("response", "")
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation", response_model=MultiQueryResponse)
async def multi_query_conversation(request: MultiQueryRequest):
    """
    Process a list of queries sequentially as a conversation/follow-up.
    """
    global observe_system
    if observe_system is None:
        raise HTTPException(
            status_code=500, detail="System not initialized.")

    observe_system.clear_memory()
    conversation_results = []
    try:
        for q in request.queries:
            logger.info(f"Processing conversation step: {q}")
            result = observe_system.query(q)

            # Append result with the query for context in response
            conversation_results.append(
                result.get("response", "")
            )

        return MultiQueryResponse(results=conversation_results)
    except Exception as e:
        logger.error(f"Conversation processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactive_query", response_model=QueryResponse)
async def interactive_query(request: QueryRequest):
    """
    Process a query for an interactive session. 
    Unlike /query, this does NOT clear the memory, allowing for follow-up questions.
    """
    global observe_system
    if observe_system is None:
        raise HTTPException(
            status_code=500, detail="System not initialized.")

    try:
        # We intentionally do NOT clear memory here
        result = observe_system.query(request.query)
        return QueryResponse(
            response=result.get("response", "")
        )
    except Exception as e:
        logger.error(f"Interactive query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
