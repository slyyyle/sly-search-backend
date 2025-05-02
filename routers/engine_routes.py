from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
import json
import os

# Import the engine selection logic
from engine_round_robin import select_engines_round_robin, MemoryBuffer
from langchain_community.chat_models import ChatOllama
from main_config import LLM_MODEL, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT, LLM_CTX

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/engines",
    tags=["Engine Selection"]
)

# Pydantic models for request/response
class EngineSelectionRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class EngineSelectionResponse(BaseModel):
    selected_engines: List[str]
    error: Optional[str] = None
    session_id: Optional[str] = None

# Dependency to get LLM instance
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_REQUEST_TIMEOUT,
        num_ctx=LLM_CTX
    )

# Dependency to get memory buffer
def get_memory_buffer(session_id: Optional[str] = None):
    return MemoryBuffer()

@router.post("/select", response_model=EngineSelectionResponse)
async def select_engines(
    request: EngineSelectionRequest,
    llm: ChatOllama = Depends(get_llm),
    memory: MemoryBuffer = Depends(get_memory_buffer)
):
    """
    Select appropriate search engines for a given query.
    
    This endpoint uses the engine selection process to determine which search engines
    are most appropriate for the given query. It returns a list of selected engine IDs
    that can be used for searching.
    
    Args:
        query: The user's search query
        session_id: Optional session ID for maintaining context across requests
    
    Returns:
        EngineSelectionResponse containing:
        - selected_engines: List of engine IDs to use
        - error: Any error message if something went wrong
        - session_id: The session ID used (if provided)
    """
    try:
        # Load engine catalog
        with open("engine_catalog.json", 'r', encoding='utf-8') as f:
            all_engines = json.load(f)
        
        # Run engine selection
        selected_engines, error = select_engines_round_robin(
            request.query,
            llm,
            all_engines,
            memory
        )
        
        return EngineSelectionResponse(
            selected_engines=selected_engines,
            error=error,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error selecting engines: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error selecting engines: {str(e)}"
        )

@router.get("/status")
async def get_engine_status():
    """
    Get the status of the engine selection system.
    
    Returns basic information about the engine selection system,
    including the LLM model being used and whether required files are present.
    """
    try:
        # Check for required files
        files_exist = {
            "engine_catalog.json": os.path.exists("engine_catalog.json"),
            "engine_compare.json": os.path.exists("engine_compare.json")
        }
        
        return {
            "status": "ok",
            "llm_model": LLM_MODEL,
            "files_exist": files_exist
        }
        
    except Exception as e:
        logger.error(f"Error getting engine status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting engine status: {str(e)}"
        ) 