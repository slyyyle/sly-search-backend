# main_orchestrator.py
import time
import traceback
import json
from typing import List, Dict, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig

from common import (
    ENGINES_FILE, LLM_MODEL, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT, LLM_NUM_CTX,
    check_files_exist, load_json_data, create_all_chains, CATEGORY_GROUPINGS
)
from step1_intent_inference import run_step1_intent_inference, Step1Result
from step2_refine_intent import run_step2_refine_intent
from step3_query_analysis import run_step3_analyze_query
from step4_generate_candidates import run_step4_generate_candidates
from step5_evaluate_engines import run_step5_evaluate_engines

def run_full_selection_process(user_query: str, llm: ChatOllama, all_engines: List[Dict]):
    print(f"\n--- Running Full Selection Process for Query: '{user_query}' ---")
    start_time = time.time()
    selected_ids: Optional[List[str]] = None
    error_message: Optional[str] = None

    try:
        # 0. Setup
        llm_config = RunnableConfig(configurable={"session_id": f"engine_select_{time.time()}"})
        try: chains = create_all_chains(llm); print("Chains created successfully.")
        except Exception as chain_e: raise RuntimeError(f"Failed to create LLM chains: {chain_e}")

        # 1. Infer Initial Intent
        step1_output: Optional[Step1Result] = run_step1_intent_inference(
            user_query=user_query, llm=llm, chains=chains, llm_config=llm_config
        )
        if step1_output is None: raise ValueError("Step 1 failed.")

        # Unpack results - naming reflects Step 1's output
        best_groupings, clarification_needed, clarification_question, query_terms = step1_output
        print("\n--- Output from Step 1 ---")
        print(f"Best Groupings: {best_groupings}") # Changed label
        print(f"Clarification Needed: {clarification_needed}")
        print(f"Clarification Question: {clarification_question}")
        print(f"Query Terms: '{query_terms}'")
        print("---------------------------\n")

        # 2. Refine Groupings
        refined_groupings: List[str] = run_step2_refine_intent(
            best_categories=best_groupings, # Pass the list from step 1
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            user_query=user_query, llm=llm, chains=chains, llm_config=llm_config
        )
        print("\n--- Output from Step 2 ---")
        print(f"Refined Groupings: {refined_groupings}")
        print("---------------------------\n")
        if not refined_groupings:
            print("WARN: No refined groupings available after Step 2.")
            selected_ids = []

        if refined_groupings:
            # 3. Perform Query Analysis
            analysis: str = run_step3_analyze_query(
                query_terms=query_terms,
                refined_groupings=refined_groupings, # Pass the final list
                llm=llm, chains=chains, llm_config=llm_config
            )
            print("\n--- Output from Step 3 ---")
            print(f"Query Analysis:\n{analysis}")
            print("---------------------------\n")

            # 4. Generate Candidate Engines
            candidate_engines: List[Dict] = run_step4_generate_candidates(
                refined_groupings=refined_groupings, # Pass the final list
                query_terms=query_terms,
                analysis_or_considerations=analysis,
                all_engines=all_engines, llm=llm, chains=chains, llm_config=llm_config
            )
            print("\n--- Output from Step 4 ---")
            print(f"Candidate Engines ({len(candidate_engines)}):")
            print(json.dumps([e.get('id', 'NO_ID') for e in candidate_engines], indent=2))
            print("---------------------------\n")

            if not candidate_engines: selected_ids = []
            else:
                # 5. Evaluate Individual Engines
                selected_ids = run_step5_evaluate_engines(
                    candidate_engines=candidate_engines, query_terms=query_terms,
                    refined_groupings=refined_groupings, # Pass the final list
                    analysis_or_considerations=analysis,
                    llm=llm, chains=chains, llm_config=llm_config
                )

    except Exception as e:
        print(f"\n--- ERROR during orchestration: {e} ---")
        traceback.print_exc()
        error_message = f"Orchestration failed: {e}"
        selected_ids = None

    finally:
        end_time = time.time()
        print(f"\n--- Orchestration Complete (Took {end_time - start_time:.2f} seconds) ---")
        if error_message: print(f"PROCESSING ERROR: {error_message}")
        elif selected_ids is not None:
            print("\n--- Final Selected Engine ID List ---")
            if selected_ids: print("[\n  " + ",\n  ".join(f'"{id_}"' for id_ in selected_ids) + "\n]")
            else: print("[] (No engines selected)")

# --- Main Execution Block (Keep as is) ---
if __name__ == "__main__":
    print("--- Initializing Orchestrator ---")
    if not check_files_exist(): exit(1)
    engines_data = []
    try:
        engines_data = load_json_data(ENGINES_FILE)
        print(f"Loaded {len(engines_data)} engine entries.")
        if not engines_data: print("FATAL ERROR: Engine catalog is empty."); exit(1)
        valid_engines = [e for e in engines_data if isinstance(e, dict) and e.get("id") and e.get("name")]
        if len(valid_engines) != len(engines_data): print(f"WARN: Filtered {len(engines_data) - len(valid_engines)} invalid/incomplete engine entries.")
        engines_data = valid_engines
        if not engines_data: print(f"FATAL ERROR: No valid engines loaded from '{ENGINES_FILE}'."); exit(1)
    except Exception as e: print(f"Fatal Error loading engine data: {e}"); traceback.print_exc(); exit(1)

    print(f"Initializing Ollama LLM ({LLM_MODEL})...")
    llm = None
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, request_timeout=LLM_REQUEST_TIMEOUT, num_ctx=LLM_NUM_CTX)
        print("Testing LLM connection...")
        test_output = llm.invoke("Respond with only OK")
        print(f"LLM Connection Test Response: {getattr(test_output, 'content', str(test_output))}")
    except Exception as e: print(f"FATAL ERROR initializing LLM: {e}"); traceback.print_exc(); exit(1)

    print("\n--- Starting Interactive Session ---")
    print("Type 'exit' or 'quit' to end.")
    while True:
        try:
            user_query_main = input("\nEnter Query: ")
            if user_query_main.lower() in ["exit", "quit"]: break
            if not user_query_main.strip(): continue
            run_full_selection_process(user_query_main, llm, engines_data)
        except KeyboardInterrupt: print("\nExiting..."); break
        except EOFError: print("\nInput stream closed. Exiting..."); break
        except Exception as e: print(f"\nUNEXPECTED ERROR in main loop: {e}"); traceback.print_exc()

    print("\n--- Session Ended ---")
