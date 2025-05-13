from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
import time
import traceback
import httpx

# Import components from engine_round_robin
from engine_round_robin import (
    MemoryBuffer,
    create_all_chains,
    parse_llm_json_output,
    refine_engines_with_comparison,
    load_json_data,
    CATEGORY_GROUPINGS,
    CATEGORY_FILTER_MAP,
    ENGINE_CATEGORIES
)
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig
from main_config import LLM_MODEL, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT, LLM_CTX, SEARXNG_URL

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/engines",
    tags=["Engine Selection"]
)

# Add Pydantic model for request body validation
class EngineSelectionPostBody(BaseModel):
    query: str
    session_id: Optional[str] = None # Allow session_id in body too, though header is preferred

# Keep Response model definition for reference
class EngineSelectionResponse(BaseModel):
    selected_engines: List[str]
    error: Optional[str] = None
    session_id: Optional[str] = None

# --- Dependencies ---
def get_llm():
    # Consider making this a singleton or using a dependency management framework
    # for potentially better resource management if called frequently.
    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_REQUEST_TIMEOUT,
        num_ctx=LLM_CTX
    )

# Simple session management example (replace with more robust solution if needed)
sessions: Dict[str, MemoryBuffer] = {}
def get_memory_buffer(request: Request) -> MemoryBuffer:
    session_id = request.headers.get("X-Session-ID")
    if session_id and session_id in sessions:
        logger.info(f"Reusing memory buffer for session: {session_id}")
        return sessions[session_id]
    # If no session_id or not found, create a new one
    new_memory = MemoryBuffer()
    if session_id:
        sessions[session_id] = new_memory # Store new buffer for session
        logger.info(f"Created new memory buffer for session: {session_id}")
    else:
        logger.info("Created new memory buffer (no session id)")
    return new_memory


# --- SSE Event Generator ---
async def engine_selection_stream_generator(
    query: str,
    session_id: Optional[str], # Pass session_id if needed for memory persistence
    llm: ChatOllama,
    memory: MemoryBuffer,
    all_engines: List[Dict]
):
    """Async generator for streaming engine selection progress via SSE."""

    start_time = time.time()
    selected_engine_ids: List[str] = []
    llm_config = RunnableConfig(configurable={"session_id": f"engine_select_{session_id or time.time()}"})
    engine_compare_data = []

    def yield_event(event_type: str, data: Dict):
        """Helper to format SSE events."""
        # Add timestamp automatically
        data_with_ts = {**data, "timestamp": time.time()}
        return f"event: {event_type}\ndata: {json.dumps(data_with_ts)}\n\n"

    try:
        logger.debug("Generator started. Yielding initial message.")
        yield yield_event("log_message", {"message": f"Processing query: '{query}'..."})
        logger.debug("Yielded initial message. Sleeping.")
        await asyncio.sleep(0.05)
        logger.debug("Slept after initial message.")

        # --- Load comparison data ---
        logger.debug("Loading comparison data...")
        try:
            engine_compare_data = await asyncio.to_thread(load_json_data, "engine_compare.json")
            logger.debug("Comparison data loaded. Yielding message.")
            yield yield_event("log_message", {"message": "Engine comparison data loaded."})
            logger.debug("Yielded comparison data loaded message.")
        except Exception as e:
            logger.warning(f"Failed to load comparison data: {e}")
            yield yield_event("log_message", {"message": f"WARN: Failed to load engine comparison data: {e}", "type": "warning"})
            logger.debug("Yielded comparison data load warning.")
        await asyncio.sleep(0.05)

        # --- Create chains ---
        logger.debug("Creating LLM chains...")
        try:
            chains = await asyncio.to_thread(create_all_chains, llm)
            logger.debug("LLM chains created. Yielding message.")
            yield yield_event("log_message", {"message": "LLM chains initialized."})
            logger.debug("Yielded chains initialized message.")
        except Exception as chain_e:
            logger.error(f"Failed to create LLM chains: {chain_e}", exc_info=True)
            detailed_error = f"Failed to create LLM chains: {chain_e}\n{traceback.format_exc()}"
            raise RuntimeError(detailed_error) # Let outer handler catch and yield error
        await asyncio.sleep(0.05)

        # --- Step 1: Query Expansion ---
        logger.debug("Step 1: Starting Query Expansion. Yielding message.")
        yield yield_event("log_message", {"step": 1, "status": "start", "message": "Expanding query..."})
        logger.debug("Step 1: Invoking chain...")
        step1_input = {"query": query}
        try:
            raw_output_1 = await chains["query_expansion_chain"].ainvoke(step1_input, config=llm_config)
            # Emit raw LLM output for front-end display
            yield yield_event("llm_output", {
                "step": 1,
                "chain": "query_expansion",
                "raw_output": getattr(raw_output_1, "content", str(raw_output_1))
            })
            parsed_output_1 = parse_llm_json_output(raw_output_1, chains["query_expansion_parser"])
            # Emit parsed LLM output for front-end display
            yield yield_event("parsed_output", {
                "step": 1,
                "chain": "query_expansion",
                "parsed_output": parsed_output_1
            })
            if not parsed_output_1:
                raw_str_error = getattr(raw_output_1, 'content', str(raw_output_1))
                raise ValueError(f"Failed to parse Step 1 JSON. Raw: {raw_str_error[:100]}...")

            plausible_groupings = parsed_output_1.get("plausible_groupings", [])
            if not isinstance(plausible_groupings, list): plausible_groupings = []
            valid_plausible_groupings = [g for g in plausible_groupings if g in CATEGORY_GROUPINGS]

            memory.update({
                "original_query": query,
                "expanded_query": parsed_output_1.get("expanded_query", query),
                "plausible_groupings": valid_plausible_groupings,
                "analysis": parsed_output_1.get("analysis", "")
            })
            logger.debug("Step 1: Logic complete. Yielding result message.")
            yield yield_event("log_message", {
                "step": 1, "status": "complete",
                "message": f"Query Expansion Results",
                "expanded_query": memory.get("expanded_query"),
                "plausible_groupings": memory.get("plausible_groupings"),
                "analysis": memory.get("analysis")[:100] + "..." if memory.get("analysis") else ""
            })
            logger.debug("Step 1: Yielded result message.")
        except Exception as e:
            logger.error(f"Step 1 Error: {e}", exc_info=True)
            yield yield_event("log_message", {"step": 1, "status": "error", "message": f"Error: {e}", "type": "error"})
            logger.debug("Step 1: Yielded error message.")
            raise
        await asyncio.sleep(0.05)

        # --- Step 2: Grouping Selection ---
        logger.debug("Step 2: Starting Grouping Selection. Yielding message.")
        yield yield_event("log_message", {"step": 2, "status": "start", "message": "Selecting category grouping..."})
        if not memory.get("plausible_groupings"):
             raise ValueError("No plausible groupings identified in Step 1")
        step2_input = {k: memory.get(k) for k in ["original_query", "expanded_query", "plausible_groupings", "analysis"]}
        logger.debug("Step 2: Invoking chain...")
        try:
            raw_output_2 = await chains["grouping_selection_chain"].ainvoke(step2_input, config=llm_config)
            # Emit raw LLM output for front-end visualization
            yield yield_event("llm_output", {
                "step": 2,
                "chain": "grouping_selection",
                "raw_output": getattr(raw_output_2, "content", str(raw_output_2))
            })
            parsed_output_2 = parse_llm_json_output(raw_output_2, chains["grouping_selection_parser"])
            # Emit parsed LLM output for front-end visualization
            yield yield_event("parsed_output", {
                "step": 2,
                "chain": "grouping_selection",
                "parsed_output": parsed_output_2
            })
            selected_grouping = parsed_output_2.get("selected_grouping")
            # Basic validation
            if selected_grouping not in CATEGORY_GROUPINGS:
                 yield yield_event("log_message", {"message": f"WARN: Selected grouping '{selected_grouping}' not canonical. Using it anyway.", "type": "warning"})


            initial_categories = CATEGORY_FILTER_MAP.get(selected_grouping, [])
            if not initial_categories:
                raise ValueError(f"No engine categories mapped from grouping: {selected_grouping}")

            memory.update({
                "selected_grouping": selected_grouping,
                "initial_categories": initial_categories
            })
            logger.debug("Step 2: Logic complete. Yielding result message.")
            yield yield_event("log_message", {
                "step": 2, "status": "complete",
                "message": "Grouping Selection Results",
                "selected_grouping": selected_grouping,
                "initial_categories": initial_categories
            })
            logger.debug("Step 2: Yielded result message.")
        except Exception as e:
            logger.error(f"Step 2 Error: {e}", exc_info=True)
            yield yield_event("log_message", {"step": 2, "status": "error", "message": f"Error: {e}", "type": "error"})
            logger.debug("Step 2: Yielded error message.")
            raise
        await asyncio.sleep(0.05)


        # --- Step 3: Category Refinement ---
        logger.debug("Step 3: Starting Category Refinement. Yielding message.")
        yield yield_event("log_message", {"step": 3, "status": "start", "message": "Refining engine categories..."})
        step3_input = {k: memory.get(k) for k in ["original_query", "expanded_query", "analysis", "selected_grouping", "initial_categories"]}
        logger.debug("Step 3: Invoking chain...")
        try:
            raw_output_3 = await chains["category_refinement_chain"].ainvoke(step3_input, config=llm_config)
            # Emit raw LLM output for front-end visualization
            yield yield_event("llm_output", {
                "step": 3,
                "chain": "category_refinement",
                "raw_output": getattr(raw_output_3, "content", str(raw_output_3))
            })
            parsed_output_3 = parse_llm_json_output(raw_output_3, chains["category_refinement_parser"])
            # Emit parsed LLM output for front-end visualization
            yield yield_event("parsed_output", {
                "step": 3,
                "chain": "category_refinement",
                "parsed_output": parsed_output_3
            })
            refined_categories = parsed_output_3.get("refined_categories", [])
             # Basic validation: ensure it's a list and categories are valid
            if not isinstance(refined_categories, list): refined_categories = []
            refined_categories = [cat for cat in refined_categories if cat in ENGINE_CATEGORIES]

            if not refined_categories:
                 yield yield_event("log_message", {"message": "WARN: Category refinement resulted in empty list. Falling back to initial categories.", "type": "warning"})
                 refined_categories = memory.get("initial_categories", [])

            memory.update({"refined_categories": refined_categories})
            logger.debug("Step 3: Logic complete. Yielding result message.")
            yield yield_event("log_message", {
                "step": 3, "status": "complete",
                "message": "Category Refinement Results",
                "refined_categories": refined_categories,
                "rationale": parsed_output_3.get("rationale", "")[:100] + "..." if parsed_output_3.get("rationale") else ""
            })
            logger.debug("Step 3: Yielded result message.")
        except Exception as e:
            logger.error(f"Step 3 Error: {e}", exc_info=True)
            yield yield_event("log_message", {"step": 3, "status": "error", "message": f"Error: {e}", "type": "error"})
            logger.debug("Step 3: Yielded error message.")
            raise
        await asyncio.sleep(0.05)


        # --- Step 4: Map Categories to Engines ---
        logger.debug("Step 4: Starting Mapping. Yielding message.")
        yield yield_event("log_message", {"step": 4, "status": "start", "message": "Mapping categories to candidate engines..."})
        candidate_engines = []
        seen_ids = set()
        refined_categories_set = set(memory.get("refined_categories", []))
        if not refined_categories_set:
             raise ValueError("No refined categories available for mapping")

        for engine in all_engines:
            engine_id = engine.get("id")
            if not engine_id: continue
            engine_categories_set = set(engine.get("categories", []))
            if engine_categories_set.intersection(refined_categories_set):
                if engine_id not in seen_ids:
                    candidate_engines.append(engine)
                    seen_ids.add(engine_id)

        if not candidate_engines:
            raise ValueError("No engines matched the refined categories")

        memory.set("candidate_engine_ids", [engine.get("id") for engine in candidate_engines])
        logger.debug("Step 4: Mapping complete. Yielding result.")
        yield yield_event("log_message", {
            "step": 4, "status": "complete",
            "message": f"Found {len(candidate_engines)} candidate engines.",
            "candidate_ids": memory.get("candidate_engine_ids")
        })
        logger.debug("Step 4: Yielded result message.")
        await asyncio.sleep(0.05)

        # --- Engine Health Check for 404s ---
        for engine in candidate_engines:
            eid = engine.get("id")
            try:
                # Ping SearXNG with a dummy query to test engine availability
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        str(SEARXNG_URL).rstrip('/') + '/',
                        params={"format": "json", "q": "__health_check__", "engines": eid}
                    )
                if resp.status_code == 404:
                    yield yield_event("engine_error", {"engine_id": eid, "error": "404 Not Found"})
            except Exception as e:
                # Other HTTP errors
                err = getattr(e, 'message', str(e))
                yield yield_event("engine_error", {"engine_id": eid, "error": err})
        logger.debug("Engine health check complete.")

        # --- Step 5: Evaluate Individual Engines ---
        logger.debug("Step 5: Starting Engine Evaluation. Yielding message.")
        yield yield_event("log_message", {
            "step": 5, 
            "status": "start", 
            "message": f"Evaluating {len(candidate_engines)} engines individually...",
            "candidate_engines": [engine.get("id") for engine in candidate_engines]
        })
        common_context_keys = ["original_query", "expanded_query", "analysis", "selected_grouping"] # Removed refined_categories from context for simplified prompt
        common_context = {k: memory.get(k) for k in common_context_keys}
        temp_selected_ids = [] # Store decisions from this step

        for i, engine in enumerate(candidate_engines):
            engine_id = engine.get("id", f"unknown_{i}")
            engine_name = engine.get("name", "Unknown")
            logger.debug(f"Step 5: Evaluating engine {i+1}/{len(candidate_engines)}: {engine_id}. Yielding progress.")
            yield yield_event("log_message", {"step": 5, "status": "progress", "message": f"Evaluating ({i+1}/{len(candidate_engines)}): {engine_name} ({engine_id})"})
            step5_input = {
                **common_context, "engine_id": engine_id, "engine_name": engine_name,
                "primary_use_case": engine.get("primary_use_case", "N/A"),
                "returns": engine.get("returns", "N/A"),
                "suitability": engine.get("suitability", "N/A")
            }
            decision = "EXCLUDE" # Default
            try:
                logger.debug(f"Step 5 ({engine_id}): Invoking chain...")
                # Invoke chain and capture raw LLM output
                raw_output_5 = await chains["engine_decision_chain"].ainvoke(step5_input, config=llm_config)
                # Emit raw LLM output for front-end visualization
                yield yield_event("llm_output", {
                    "step": 5,
                    "chain": "engine_decision",
                    "engine_id": engine_id,
                    "raw_output": getattr(raw_output_5, "content", str(raw_output_5))
                })
                # Parse LLM output JSON
                parsed_output_5 = parse_llm_json_output(raw_output_5, chains["engine_decision_parser"])
                # Emit parsed LLM output for front-end visualization
                yield yield_event("parsed_output", {
                    "step": 5,
                    "chain": "engine_decision",
                    "engine_id": engine_id,
                    "parsed_output": parsed_output_5
                })
                if not parsed_output_5 or "decision" not in parsed_output_5:
                    yield yield_event("log_message", {"step": 5, "message": f"WARN: Failed parse/missing decision for {engine_id}.", "type": "warning"})
                    decision = "EXCLUDE"
                else:
                    # Decision should already be validated by wrap_string_in_json in the chain
                    decision = parsed_output_5.get("decision", "EXCLUDE").strip().upper()

                # Emit explicit decision event for front-end
                yield yield_event("decision", {
                    "step": 5,
                    "chain": "engine_decision",
                    "engine_id": engine_id,
                    "decision": decision
                })
                if decision == "INCLUDE":
                    temp_selected_ids.append(engine_id)
                    logger.debug(f"Step 5 ({engine_id}): Included. Yielding status.")
                    yield yield_event("engine_status", {"engine_id": engine_id, "engine_name": engine_name, "status": "selected"})
                elif decision == "EXCLUDE":
                     logger.debug(f"Step 5 ({engine_id}): Excluded. Yielding status.")
                     yield yield_event("engine_status", {"engine_id": engine_id, "engine_name": engine_name, "status": "rejected"})
                # No need for else, wrap_string_in_json defaults invalid output to EXCLUDE

            except Exception as eval_e:
                 logger.error(f"Step 5 ({engine_id}): Error evaluating: {eval_e}", exc_info=True)
                 yield yield_event("log_message", {"step": 5, "status": "error", "message": f"Error evaluating {engine_id}: {eval_e}", "type": "error"})
                 yield yield_event("engine_status", {"engine_id": engine_id, "engine_name": engine_name, "status": "error"})
                 logger.debug(f"Step 5 ({engine_id}): Yielded error status.")
            await asyncio.sleep(0.05) # Small delay between engine evals

        selected_engine_ids = temp_selected_ids # Assign results from this step
        logger.debug("Step 5: Evaluation loop finished. Yielding completion message.")
        yield yield_event("log_message", {"step": 5, "status": "complete", "message": f"Engine evaluation finished. Initially selected: {len(selected_engine_ids)}"})
        logger.debug("Step 5: Yielded completion message.")
        await asyncio.sleep(0.05)


        # --- Step 6: Refine Selection with Comparison Data ---
        logger.debug("Step 6: Starting Comparison Refinement. Yielding message.")
        yield yield_event("log_message", {"step": 6, "status": "start", "message": "Refining selection with comparison data..."})
        if memory.get("refined_categories") and engine_compare_data:
            original_count = len(selected_engine_ids)
            selected_engine_ids = refine_engines_with_comparison(
                selected_engine_ids,
                memory.get("refined_categories"),
                engine_compare_data
            )
            logger.debug("Step 6: Refinement applied. Yielding message.")
            yield yield_event("log_message", {"step": 6, "status": "complete", "message": f"Comparison refinement applied. Count changed from {original_count} to {len(selected_engine_ids)}."})
        else:
             logger.debug("Step 6: Skipping refinement. Yielding message.")
             yield yield_event("log_message", {"step": 6, "status": "skipped", "message": "Skipping comparison refinement (no categories or data)."})
        logger.debug("Step 6: Yielded message.")
        await asyncio.sleep(0.05)


        # --- Step 7: Finalize ---
        logger.debug("Step 7: Starting Finalization. Yielding message.")
        yield yield_event("log_message", {"step": 7, "status": "start", "message": "Finalizing selection..."})
        if not selected_engine_ids:
            yield yield_event("log_message", {"message": "WARN: No engines selected after evaluation/refinement. Attempting fallback.", "type": "warning"})
            candidate_ids = memory.get("candidate_engine_ids", [])
            if candidate_ids:
                 selected_engine_ids = [candidate_ids[0]] # Fallback to first candidate
                 yield yield_event("log_message", {"message": f"Using first candidate as fallback: {selected_engine_ids[0]}", "type": "info"})
            else:
                 yield yield_event("log_message", {"message": "ERROR: No candidates identified, final list is empty.", "type": "error"})
                 # No fallback possible, final list will be empty

        memory.set("selected_engine_ids", selected_engine_ids)
        logger.debug("Step 7: Finalization complete. Yielding message.")
        yield yield_event("log_message", {"step": 7, "status": "complete", "message": "Finalization complete."})
        logger.debug("Step 7: Yielded message.")
        await asyncio.sleep(0.05)


        # --- Send Final Result ---
        logger.debug("Yielding final result.")
        yield yield_event("final_result", {
            "selected_engines": selected_engine_ids,
            "final_query_details": memory.get_dict(), # Send final memory state for context
            })
        logger.debug("Yielded final result.")

    except Exception as e:
        error_message = f"Error during engine selection stream: {e}"
        # Ensure traceback is included only if appropriate (e.g., debug mode)
        tb = traceback.format_exc()
        logger.error(f"{error_message}\n{tb}") # Log server-side too
        logger.debug("Yielded error event in main except block.")
        yield yield_event("error", {"message": error_message, "detail": tb if os.getenv("DEBUG") else "See server logs for details."})


    finally:
        end_time = time.time()
        logger.debug("Generator finished. Yielding final log message.")
        yield yield_event("log_message", {"message": f"Stream finished. Total time: {end_time - start_time:.2f} seconds.", "status": "end"})
        logger.debug("Yielded final log message.")

# --- API Route ---
@router.post("/select")
async def stream_engine_selection(
    payload: EngineSelectionPostBody,
    request: Request, # Keep request for other potential uses (or remove if only needed for header)
    llm: ChatOllama = Depends(get_llm),
    memory: MemoryBuffer = Depends(get_memory_buffer) # MODIFIED: Use Depends(get_memory_buffer)
):
    """Streams engine selection progress using Server-Sent Events."""
    logger.debug(f"Received request for /select. Payload: {payload.dict()}")
    try:
        query = payload.query
        # Get session_id from the already processed dependency or payload
        # Note: memory object doesn't store the session_id itself, so we still might need it from header/payload if generator needs it
        session_id = request.headers.get("X-Session-ID") or payload.session_id

        # Memory buffer is now handled by the dependency

        # Load engine catalog off the main thread to avoid blocking
        try:
            all_engines_data = await asyncio.to_thread(load_json_data, "engine_catalog.json")
        except Exception as e:
            logger.error(f"Failed to load engine_catalog.json: {e}", exc_info=True)
            async def error_gen(): yield f"event: error\ndata: {json.dumps({'message': 'Server configuration error: Cannot load engines.'})}\n\n"
            return StreamingResponse(
                error_gen(),
                media_type="text/event-stream",
                status_code=500,
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        # Start the stream
        return StreamingResponse(
            engine_selection_stream_generator(query, session_id, llm, memory, all_engines_data),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }  # Important for SSE, disable proxy buffering
        )

    except Exception as e:
        logger.error(f"Error processing engine selection request: {e}", exc_info=True)
        async def error_gen(): yield f"event: error\ndata: {json.dumps({'message': f'Internal server error: {e}'})}\n\n"
        return StreamingResponse(
            error_gen(),
            media_type="text/event-stream",
            status_code=500,
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )


# Keep the /status endpoint
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
