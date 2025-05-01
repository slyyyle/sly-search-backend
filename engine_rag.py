# engine_round_robin.py

# --- Core Imports ---
import json
import os
import warnings
import time
import traceback
from operator import itemgetter
from typing import List, Dict, Any, Optional, Literal

# *** No Pydantic models needed for this version's core logic ***
# from pydantic.v1 import BaseModel, Field

from langchain_community.chat_models import ChatOllama
# Only StrOutputParser needed for parsing decision
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# --- Configuration ---
ENGINES_FILE = "engine_catalog.json"
LLM_MODEL = "llama3:8b-instruct-q8_0" # Or your preferred model
LLM_TEMPERATURE = 0.1 # Keep low for deterministic decision
LLM_REQUEST_TIMEOUT = 120.0

# --- Suppress Warnings ---
# warnings.filterwarnings("ignore", message="Importing BaseSettings from pydantic v1")

# --- File Existence Check ---
def check_files_exist():
    """Checks if the required engine catalog file exists."""
    if not os.path.exists(ENGINES_FILE):
        print(f"\nCRITICAL ERROR: Engine catalog file not found!")
        print(f"Expected '{ENGINES_FILE}' in the directory: {os.getcwd()}")
        return False
    return True

# --- Data Loading ---
def load_json_data(filepath):
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {filepath}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred loading {filepath}: {e}")

# --- LCEL Chain Creation (Simplified Decision Prompt) ---
def create_round_robin_chains(llm):
    """Creates the LCEL chains for query expansion and direct engine decision based only on RAG context."""

    # 1. Query Expansion Chain
    query_expansion_prompt_template = """
You are an AI assistant analyzing a user's search query to understand its nuances before evaluating potential search engines.
Expand on the provided USER QUERY, considering potential underlying goals, desired content types (general web, images, files, academic, etc.), implied language, and geographical scope.
Output a concise verbal consideration of these factors. This consideration will guide the evaluation of individual search engines.

USER QUERY: {query}

EXPANDED CONSIDERATIONS:
"""
    query_expansion_prompt = ChatPromptTemplate.from_template(query_expansion_prompt_template)
    query_expander_chain = (query_expansion_prompt | llm | StrOutputParser())

    # 2. Direct Engine Inclusion Decision Chain (MODIFIED: Uses only rag_context)
    engine_decision_prompt_template = """
Analyze the specific SEARCH ENGINE described below based on the ORIGINAL USER QUERY and the EXPANDED QUERY CONSIDERATIONS. Focus primarily on the RAG CONTEXT provided for the engine.

ORIGINAL USER QUERY: {original_query}

EXPANDED QUERY CONSIDERATIONS:
{expanded_considerations}

SEARCH ENGINE DETAILS:
Name: {engine_name}
RAG Context: {rag_context}

Decision Task: Decide *solely* based on the information above (especially the RAG Context) whether this *specific* engine is potentially relevant enough to the user's query and considerations to be included in an initial list for further refinement later. Consider the engine's focus and described functionality against the query goal and content type. Ignore how many other engines might be included.

Output *only* the word "INCLUDE" if the engine seems potentially relevant, or "EXCLUDE" if it clearly does not seem relevant. Do not add any other words or explanation.

Decision (INCLUDE or EXCLUDE):
"""
    engine_decision_prompt = ChatPromptTemplate.from_template(engine_decision_prompt_template)
    # Output is just the single word decision
    engine_decider_chain = (engine_decision_prompt | llm | StrOutputParser())

    # Return only the two chains
    return query_expander_chain, engine_decider_chain


# --- Main Orchestration Logic (Simplified Input) ---
def select_engines_round_robin(query: str, llm, engine_catalog: List[Dict]) -> List[str]:
    """
    Expands query and iterates through engines, deciding inclusion for each based
    on query, considerations, engine name, and engine RAG context.

    Args:
        query: The user's original search query.
        llm: The initialized ChatOllama model.
        engine_catalog: The loaded list of engine dictionaries.

    Returns:
        A list of names (strings) of the engines selected for inclusion.
    """
    print(f"\n--- Starting Engine Selection for Query: '{query}' ---")

    query_expander, engine_decider = create_round_robin_chains(llm)

    # 1. Expand Query
    print("Step 1: Expanding query...")
    try:
        expanded_considerations = query_expander.invoke({"query": query})
        print(f"  Expanded Considerations:\n{expanded_considerations}")
    except Exception as e:
        print(f"  ERROR during query expansion: {e}. Proceeding without expansion.")
        expanded_considerations = "Query expansion failed."

    # 2. Iterate and Decide for Each Engine
    print(f"\nStep 2: Evaluating {len(engine_catalog)} engines individually...")
    master_list_names = [] # Will store names of included engines
    evaluated_count = 0
    include_count = 0

    for engine in engine_catalog:
        evaluated_count += 1
        engine_name = engine.get("name", f"Unknown Engine #{evaluated_count}")
        engine_id = engine.get("id", engine_name) # Keep ID for logging if needed
        # Fetch ONLY the RAG context for the LLM decision
        engine_rag = engine.get("rag_context", "No RAG context available.") # Provide default

        if not engine_name or engine_name.startswith("Unknown Engine"):
            print(f"  Skipping engine entry {evaluated_count} due to missing/invalid name.")
            continue
        # Optional: Skip if rag_context is missing or empty
        # if not engine_rag or engine_rag == "No RAG context available.":
        #     print(f"  Skipping engine {engine_name} due to missing RAG context.")
        #     continue

        print(f"  Evaluating ({evaluated_count}/{len(engine_catalog)}): {engine_name} ({engine_id})")

        # Prepare input for the decider chain (Simplified)
        decision_input = {
            "original_query": query,
            "expanded_considerations": expanded_considerations,
            "engine_name": engine_name,
            "rag_context": engine_rag # Only pass RAG context
        }

        decision_output = "EXCLUDE" # Default

        try:
            # Make the decision for this engine
            decision_output = engine_decider.invoke(decision_input)
            decision_text = decision_output.strip().upper()
            print(f"    LLM Decision Raw: '{decision_output.strip()}' -> Processed: '{decision_text}'")

            # Check if the decision is "INCLUDE"
            # More robust check: must contain INCLUDE and NOT contain EXCLUDE
            if "INCLUDE" in decision_text and "EXCLUDE" not in decision_text:
                 print(f"    Decision: INCLUDE")
                 master_list_names.append(engine_name) # Add only the name
                 include_count += 1
            else:
                 if "EXCLUDE" not in decision_text:
                      print(f"    WARN: Unexpected decision output '{decision_text}'. Treating as EXCLUDE.")
                 print(f"    Decision: EXCLUDE")

        except Exception as e:
            print(f"    ERROR evaluating engine {engine_name}: {e}")
            # traceback.print_exc()
            print(f"    Decision: EXCLUDE (due to error)")

        # time.sleep(0.05) # Optional delay

    print(f"\n--- Engine Selection Complete ---")
    print(f"  Included {include_count} out of {evaluated_count} evaluated engines.")

    return master_list_names

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Initializing Engine Round Robin Selector ---")

    if not check_files_exist():
        exit(1)

    # --- Load Engine Data ---
    try:
        print(f"Loading engine data from {ENGINES_FILE}...")
        engines_data = load_json_data(ENGINES_FILE)
        if isinstance(engines_data, dict) and len(engines_data) == 1:
            key = list(engines_data.keys())[0]
            if isinstance(engines_data[key], list):
                print(f"Assuming engine list is under key '{key}'")
                engines_data = engines_data[key]
        if not isinstance(engines_data, list):
             raise TypeError(f"Loaded engine data from '{ENGINES_FILE}' is not a list.")
        print(f"Loaded {len(engines_data)} engine entries.")

    except (ValueError, RuntimeError, TypeError) as e:
        print(f"Fatal Error: Failed to load or process engine data. {e}")
        exit(1)
    except FileNotFoundError:
         print(f"FATAL ERROR: File not found '{ENGINES_FILE}'")
         exit(1)


    # --- Setup LLM ---
    print(f"Initializing Ollama LLM ({LLM_MODEL})...")
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            request_timeout=LLM_REQUEST_TIMEOUT
        )
        print("LLM Initialized.")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize LLM '{LLM_MODEL}'. Check Ollama. Error: {e}")
        exit(1)

    # --- Sample Query ---
    sample_query = "Find academic papers about quantum computing algorithms published after 2020."
    print(f"\n--- Running Sample Query ---")
    print(f"Sample User Query: {sample_query}")

    start_time = time.time()
    try:
        selected_engine_names = select_engines_round_robin(sample_query, llm, engines_data)
        print("\n--- Final Selected Engine Name List ---")
        if selected_engine_names:
            print("[\n  " + ",\n  ".join(f'"{name}"' for name in selected_engine_names) + "\n]")
        else:
            print("[] (No engines selected)")

    except Exception as e:
        print(f"\nERROR during main execution: {e}")
        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"\n(Selection took {end_time - start_time:.2f} seconds)")


    # --- Interactive Session ---
    print("\n--- Starting Interactive Session ---")
    print("Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_query = input("\nEnter User Query: ")
            if user_query.lower() in ["exit", "quit"]: print("Exiting..."); break
            if not user_query.strip(): continue

            start_time = time.time()
            try:
                 selected_engine_names = select_engines_round_robin(user_query, llm, engines_data)
                 print("\n--- Final Selected Engine Name List ---")
                 if selected_engine_names:
                     print("[\n  " + ",\n  ".join(f'"{name}"' for name in selected_engine_names) + "\n]")
                 else:
                     print("[] (No engines selected)")

            except Exception as e:
                 print(f"\nERROR during execution: {e}")
                 traceback.print_exc()
            finally:
                 end_time = time.time()
                 print(f"\n(Selection took {end_time - start_time:.2f} seconds)")

        except KeyboardInterrupt: print("\nExiting..."); break
        except EOFError: print("\nExiting..."); break

    print("\n--- Session Ended ---")
