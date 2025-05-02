# engine_round_robin.py (Correctly Escaping Injected Content Braces)
# --- Core Imports ---
import json
import os
import warnings
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set

# Use Pydantic v1 namespace
from pydantic.v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.exceptions import OutputParserException

# --- Configuration ---
ENGINES_FILE = "engine_catalog.json"
LLM_MODEL = "llama3:8b-instruct-q8_0"
LLM_TEMPERATURE = 0.2
LLM_REQUEST_TIMEOUT = 120.0
LLM_CTX = 4096

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", message=".*The class `ChatOllama` was deprecated.*")

# --- Category Groupings & Mapping ---
CATEGORY_GROUPINGS = ["information", "images", "music/videos", "news", "torrents", "software/tools", "gardens/museums/archives"]
CATEGORY_GROUPINGS_IMPLICATIONS = {
    "information": "this category grouping is for user requests related to finding out more information. While it leads to good sources of videos, music, news, and all other specific media or file based searches, if inferred intent is to find this media, use the other groupings.",
    "images": "if a user is searching for pics, pictures, images, photos, art, they mean static images and this category is the only answer.",
    "music/videos": "if user intent is to find music or videos, music/videos is the only answer.",
    "news": "if a user is looking for news, they will explicitly say latest, news, current events, etc. If they don't - information is the answer.",
    "torrents": "a user must specify they want torrents or peer to peer P2P downloading for this to be the answer, in which case it's the only answer.",
    "software/tools": "if a user is searching for software or computer related programs, and they don't mention torrents, the answer is software/tools and gardens/museums/archives",
    "gardens/museums/archives": "if a user mentions niche, fun, old, retro, or odd lesser found content, gardens/museums/archives is the only correct answer."
}
ENGINE_CATEGORIES = ["general", "it", "social media", "science", "images", "videos", "files", "music", "news"]
CATEGORY_FILTER_MAP = {
    "information": ["general", "it", "social media", "science"], "images": ["images", "general"],
    "music/videos": ["music", "videos", "general"], "news": ["news", "general"], "torrents": ["files"],
    "software/tools": ["it", "general"], "gardens/museums/archives": ["general", "social media"],
}
ENGINE_CATEGORY_IMPLICATIONS = {
    "general": "Broad web search, text, some media links, retro/niche content.", "it": "Software, code documentation, package managers, tech help.",
    "social media": "Forums, discussions, human opinions/experiences.", "science": "Academic research, papers, math/science questions.",
    "images": "Dedicated image file search.", "videos": "Dedicated video file/stream search.",
    "files": "Torrent/P2P file search ONLY.", "music": "Dedicated music file/stream search.", "news": "Dedicated news article/feed search.",
}

# --- Memory Buffer ---
class MemoryBuffer:
    def __init__(self): self.data = {}
    def set(self, key: str, value: Any) -> None: self.data[key] = value
    def get(self, key: str, default: Any = None) -> Any: return self.data.get(key, default)
    def update(self, key_values: Dict[str, Any]) -> None: self.data.update(key_values)
    def clear(self) -> None: self.data.clear()
    def get_dict(self) -> Dict[str, Any]: return self.data.copy()

# --- Pydantic Models ---
class QueryExpansionOutput(BaseModel):
    expanded_query: str = Field(description="Expanded query with additional context")
    plausible_groupings: List[str] = Field(description=f"List of plausible CATEGORY GROUPINGS from {CATEGORY_GROUPINGS}", default_factory=list)
    analysis: str = Field(description="Brief analysis of query intent")
class GroupingSelectionOutput(BaseModel):
    selected_grouping: str = Field(description=f"The single most appropriate CATEGORY GROUPING from {CATEGORY_GROUPINGS}")
    reasoning: str = Field(description="Brief explanation for selection")
class CategoryRefinementOutput(BaseModel):
    initial_categories: List[str] = Field(description="List of ENGINE CATEGORIES mapped from grouping")
    refined_categories: List[str] = Field(description=f"List of refined ENGINE CATEGORIES. Subset of {ENGINE_CATEGORIES}", default_factory=list)
    rationale: str = Field(description="Explanation for refinement")
class EngineEvaluationOutput(BaseModel):
    decision: str = Field(description="'INCLUDE' or 'EXCLUDE'")
    reason: str = Field(description="One sentence explaining confidence level and key factor in decision")

# --- File/Data Utils ---
def check_files_exist():
    if not os.path.exists(ENGINES_FILE): print(f"CRITICAL ERROR: '{ENGINES_FILE}' not found."); return False
    if not os.path.exists("engine_compare.json"): print(f"CRITICAL ERROR: 'engine_compare.json' not found."); return False
    return True
def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e: raise RuntimeError(f"Error loading {filepath}: {e}")

# --- Robust JSON Parsing ---
def parse_llm_json_output(raw_output: Any, parser: JsonOutputParser) -> Optional[Dict]:
    # (Keep the previous version of this function)
    if isinstance(raw_output, dict): return raw_output
    if isinstance(raw_output, BaseModel): return raw_output.dict()
    if hasattr(raw_output, 'content') and isinstance(raw_output.content, str): raw_str = raw_output.content
    elif isinstance(raw_output, str): raw_str = raw_output
    else: print(f"  ERROR: Unexpected output type for parsing: {type(raw_output)}"); return None
    try:
        cleaned_str = raw_str.strip()
        if cleaned_str.startswith("```json"): cleaned_str = cleaned_str[7:]
        if cleaned_str.endswith("```"): cleaned_str = cleaned_str[:-3]
        cleaned_str = cleaned_str.strip()
        try: return json.loads(cleaned_str)
        except json.JSONDecodeError: pass
        try: return parser.parse(raw_str).dict()
        except Exception as p_err: print(f"  ERROR: Pydantic parsing failed: {p_err}. Raw string: {raw_str[:200]}..."); return None
    except Exception as e: print(f"  ERROR: Unexpected error during JSON parsing: {e}"); return None

def wrap_string_in_json(raw_str: str) -> Dict[str, str]:
    """Wrap a string output in a JSON object with 'decision' key."""
    cleaned_str = raw_str.strip().upper()
    if cleaned_str not in ["INCLUDE", "EXCLUDE"]:
        print(f"  WARN: Invalid decision '{cleaned_str}'. Defaulting to EXCLUDE.")
        cleaned_str = "EXCLUDE"
    return {"decision": cleaned_str}

# --- Create LCEL Chains (Escaping braces in INJECTED content) ---
def create_all_chains(llm: ChatOllama):
    """Creates all LCEL chains, escaping literal braces in formatted context."""

    # --- Helper function to escape braces in context strings ---
    def escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    # --- Prepare context strings AND escape them ---
    # Original content
    category_groupings_list_str_raw = str(CATEGORY_GROUPINGS)
    category_groupings_implications_json_raw = json.dumps(CATEGORY_GROUPINGS_IMPLICATIONS, indent=2)
    engine_category_implications_json_raw = json.dumps(ENGINE_CATEGORY_IMPLICATIONS, indent=2)
    engine_categories_list_str_raw = str(ENGINE_CATEGORIES)

    # Escaped versions for insertion into templates
    category_groupings_list_str = escape_braces(category_groupings_list_str_raw)
    category_groupings_implications_json = escape_braces(category_groupings_implications_json_raw)
    engine_category_implications_json = escape_braces(engine_category_implications_json_raw)
    engine_categories_list_str = escape_braces(engine_categories_list_str_raw)

    # --- Define Templates using ESCAPED context strings ---
    # IMPORTANT: The *input variables* like {query}, {original_query} still use SINGLE braces.

    # 1. Query Expansion Chain (Step 1)
    # Uses escaped context strings via f-string formatting. {query} remains single.
    query_expansion_prompt_template = f"""
Task: Expand the USER QUERY using an understanding of CATEGORY GROUPINGS IMPLICATIONS.

CATEGORY GROUPINGS LIST: {category_groupings_list_str}
CATEGORY GROUPINGS IMPLICATIONS:
{category_groupings_implications_json}

USER QUERY: {{query}}

Analyze the query to:
1. Expand it with additional relevant context or keywords.
2. Identify ALL plausible CATEGORY GROUPINGS from the list that might apply.
3. Provide a brief analysis of the user's primary intent.

Output Format Instructions:
Generate ONLY a valid JSON object containing three keys:
- "expanded_query": A string with the expanded query.
- "plausible_groupings": A JSON list of strings, where each string is a valid grouping from the CATEGORY GROUPINGS LIST provided above.
- "analysis": A string containing your brief analysis.
Do not include any text before or after the JSON object.
"""
    query_expansion_tmpl = ChatPromptTemplate.from_template(query_expansion_prompt_template)
    query_expansion_parser = JsonOutputParser(pydantic_object=QueryExpansionOutput)
    query_expansion_chain = query_expansion_tmpl | llm | query_expansion_parser

    # 2. Grouping Selection Chain (Step 2)
    # Uses escaped context string. {original_query}, {expanded_query}, etc. remain single.
    grouping_selection_prompt_template = f"""
Task: Select the SINGLE MOST APPROPRIATE category grouping from the plausible options.

CONTEXT:
Original Query: {{original_query}}
Expanded Query: {{expanded_query}}
Plausible Groupings: {{plausible_groupings}}
Analysis from Step 1: {{analysis}}

CATEGORY GROUPINGS IMPLICATIONS:
{category_groupings_implications_json}

Instructions:
1. Review the plausible groupings provided.
2. Choose EXACTLY ONE grouping from the `Plausible Groupings` list that best matches the user's core intent described in the Analysis.
3. If multiple groupings seem valid, select the one with the most specific match according to the IMPLICATIONS.
4. Provide brief reasoning for your selection, explaining why it's better than the alternatives considered.

Output Format Instructions:
Generate ONLY a valid JSON object containing two keys:
- "selected_grouping": A string containing the single chosen grouping.
- "reasoning": A string containing your brief reasoning.
Do not include any text before or after the JSON object.
"""
    grouping_selection_tmpl = ChatPromptTemplate.from_template(grouping_selection_prompt_template)
    grouping_selection_parser = JsonOutputParser(pydantic_object=GroupingSelectionOutput)
    grouping_selection_chain = grouping_selection_tmpl | llm | grouping_selection_parser

    # 3. Category Refinement Chain (Step 3)
    # Uses escaped context strings. Input variables remain single.
    category_refinement_prompt_template = f"""
Task: Refine the list of ENGINE CATEGORIES based on the selected grouping and query context.

AVAILABLE ENGINE CATEGORIES: {engine_categories_list_str}

CONTEXT:
Original Query: {{original_query}}
Expanded Query: {{expanded_query}}
Analysis from Step 1: {{analysis}}
Selected Grouping: {{selected_grouping}}
Initial Mapped Categories (based on grouping): {{initial_categories}}

ENGINE CATEGORY IMPLICATIONS:
{engine_category_implications_json}

Instructions:
1. Start with the `Initial Mapped Categories`.
2. Evaluate each category against the specific `Expanded Query` and `Analysis`.
3. Keep only the categories from the initial list that directly serve the user's specific intent. Use the `ENGINE CATEGORY IMPLICATIONS` to understand what each category is for.
4. Remove any categories from the initial list that don't align well or are too broad/narrow for the specific query.
5. Your final `refined_categories` list must be a subset of the `Initial Mapped Categories`.
6. Explain your refinement decisions in the `rationale`, justifying inclusions and exclusions based on the query.

Output Format Instructions:
Generate ONLY a valid JSON object containing three keys:
- "initial_categories": A JSON list of strings, representing the initial categories provided in the context.
- "refined_categories": A JSON list of strings, containing the final refined list of engine categories (must be valid categories from AVAILABLE ENGINE CATEGORIES above).
- "rationale": A string explaining your refinement decisions.
Do not include any text before or after the JSON object.
"""
    category_refinement_tmpl = ChatPromptTemplate.from_template(category_refinement_prompt_template)
    category_refinement_parser = JsonOutputParser(pydantic_object=CategoryRefinementOutput)
    category_refinement_chain = category_refinement_tmpl | llm | category_refinement_parser

    # 4. Engine Decision Chain (Step 5)
    # Input variables remain single. No large pre-formatted context here.
    engine_decision_prompt_template = f"""
Task: Output EXACTLY ONE WORD: "INCLUDE" or "EXCLUDE".

CONTEXT:
Original Query: {{original_query}}
Expanded Query: {{expanded_query}}
Analysis from Step 1: {{analysis}}
Selected Grouping: {{selected_grouping}}

ENGINE DETAILS TO EVALUATE:
Name: {{engine_name}}
ID: {{engine_id}}
Primary Use Case: {{primary_use_case}}
Returns: {{returns}}
Suitability: {{suitability}}

Instructions:
1. Consider the engine's primary use case, returns, and suitability in light of the query.
2. Output EXACTLY ONE WORD: "INCLUDE" or "EXCLUDE".
3. Do not include any other text or explanation.
"""
    engine_decision_tmpl = ChatPromptTemplate.from_template(engine_decision_prompt_template)
    engine_decision_parser = StrOutputParser()  # Changed to StrOutputParser since we want raw string
    engine_decision_chain = engine_decision_tmpl | llm | engine_decision_parser | wrap_string_in_json

    # Return all chains
    return {
        "query_expansion_chain": query_expansion_chain, "query_expansion_parser": query_expansion_parser,
        "grouping_selection_chain": grouping_selection_chain, "grouping_selection_parser": grouping_selection_parser,
        "category_refinement_chain": category_refinement_chain, "category_refinement_parser": category_refinement_parser,
        "engine_decision_chain": engine_decision_chain, "engine_decision_parser": engine_decision_parser,
    }

def refine_engines_with_comparison(selected_engines: List[str], refined_categories: List[str], engine_compare_data: List[Dict]) -> List[str]:
    """Refine engine selection using engine_compare.json data."""
    if not refined_categories:
        print("  WARN: No refined categories provided. Using original selection.")
        return selected_engines

    # Get all valid engines from the refined categories
    valid_engines = set()
    for category in refined_categories:
        category_data = next((item for item in engine_compare_data if item["category"] == category), None)
        if category_data:
            category_engines = category_data.get("engines", [])
            valid_engines.update(category_engines)
        else:
            print(f"  WARN: No comparison data found for category '{category}'.")

    if not valid_engines:
        print("  WARN: No valid engines found in any refined category. Using original selection.")
        return selected_engines

    # Filter selected engines to only those in valid categories
    filtered_engines = [engine for engine in selected_engines if engine in valid_engines]
    if not filtered_engines:
        print("  WARN: No selected engines match any refined category. Using first valid engine.")
        return [list(valid_engines)[0]] if valid_engines else []

    return filtered_engines

# --- Main Orchestration Logic (Keep Previous Version) ---
def select_engines_round_robin(user_query: str, llm: ChatOllama, all_engines: List[Dict], memory: MemoryBuffer) -> Tuple[List[str], Optional[str]]:
    # (Keep the previous version of this function with try/except/finally blocks)
    print(f"\n--- Starting Engine Selection for Query: '{user_query}' ---")
    start_time = time.time()
    selected_engine_ids: List[str] = []
    error_message: Optional[str] = None
    # memory.clear() # Optional: Uncomment if each run needs full independence

    try:
        # 0. Setup
        llm_config = RunnableConfig(configurable={"session_id": f"engine_select_{time.time()}"})
        try:
            chains = create_all_chains(llm)
            print("Chains created successfully.")
        except Exception as chain_e:
            detailed_error = f"Failed to create LLM chains: {chain_e}\n{traceback.format_exc()}"
            print(detailed_error)
            raise RuntimeError(detailed_error) # Propagate chain creation error

        # Load engine comparison data
        try:
            engine_compare_data = load_json_data("engine_compare.json")
            print("Engine comparison data loaded successfully.")
        except Exception as e:
            print(f"  WARN: Failed to load engine comparison data: {e}")
            engine_compare_data = []

        # --- Step 1: Query Expansion ---
        print("\n--- Step 1: Query Expansion ---")
        step1_input = {"query": user_query}
        try:
            raw_output_1 = chains["query_expansion_chain"].invoke(step1_input, config=llm_config)
            parsed_output_1 = parse_llm_json_output(raw_output_1, chains["query_expansion_parser"])
            if not parsed_output_1:
                raw_str_error = getattr(raw_output_1, 'content', str(raw_output_1))
                raise ValueError(f"Failed to parse Step 1 JSON. Raw output: {raw_str_error[:200]}...")

            plausible_groupings = parsed_output_1.get("plausible_groupings", [])
            if not isinstance(plausible_groupings, list):
                 print(f"  WARN: 'plausible_groupings' not a list: {plausible_groupings}. Setting to empty.")
                 plausible_groupings = []
            valid_plausible_groupings = [g for g in plausible_groupings if g in CATEGORY_GROUPINGS]
            if len(valid_plausible_groupings) != len(plausible_groupings):
                 print(f"  WARN: Filtered invalid groupings: {set(plausible_groupings) - set(valid_plausible_groupings)}")

            memory.update({
                "original_query": user_query,
                "expanded_query": parsed_output_1.get("expanded_query", user_query),
                "plausible_groupings": valid_plausible_groupings,
                "analysis": parsed_output_1.get("analysis", "")
            })
            print(f"  Expanded: {memory.get('expanded_query')}, Plausible: {memory.get('plausible_groupings')}, Analysis: {memory.get('analysis')[:50]}...")
            if not memory.get("plausible_groupings"): print("  WARN: No plausible groupings identified.")

        except Exception as e:
            print(f"  ERROR during Step 1: {e}")
            traceback.print_exc()
            error_message = f"Step 1 (Query Expansion) failed: {e}"
            # Use 'raise' to stop execution here if Step 1 is critical
            raise ValueError(error_message) # Or return [], error_message

        # --- Step 2: Select Single Grouping ---
        print("\n--- Step 2: Grouping Selection ---")
        if not memory.get("plausible_groupings"):
            error_message = "No plausible groupings identified in Step 1"
            print(f"  Skipping Step 2: {error_message}")
            raise ValueError(error_message) # Stop if no groupings

        step2_input = {k: memory.get(k) for k in ["original_query", "expanded_query", "plausible_groupings", "analysis"]}
        try:
            raw_output_2 = chains["grouping_selection_chain"].invoke(step2_input, config=llm_config)
            parsed_output_2 = parse_llm_json_output(raw_output_2, chains["grouping_selection_parser"])
            if not parsed_output_2:
                raw_str_error = getattr(raw_output_2, 'content', str(raw_output_2))
                raise ValueError(f"Failed to parse Step 2 JSON. Raw output: {raw_str_error[:200]}...")

            selected_grouping = parsed_output_2.get("selected_grouping")
            plausible_set = set(memory.get("plausible_groupings", [])) # Use set for faster lookup
            if not selected_grouping or selected_grouping not in plausible_set:
                print(f"  WARN: Invalid/non-plausible grouping '{selected_grouping}'. Falling back.")
                selected_grouping = memory.get("plausible_groupings")[0] # Fallback
            elif selected_grouping not in CATEGORY_GROUPINGS:
                 print(f"  WARN: Selected grouping '{selected_grouping}' not in defined CATEGORY_GROUPINGS. Using anyway.")

            initial_categories = CATEGORY_FILTER_MAP.get(selected_grouping, [])
            if not initial_categories:
                error_message = f"No engine categories mapped from grouping: {selected_grouping}"
                print(f"  WARN: {error_message}. Cannot proceed.")
                raise ValueError(error_message)

            memory.update({
                "selected_grouping": selected_grouping,
                "grouping_reasoning": parsed_output_2.get("reasoning", ""),
                "initial_categories": initial_categories
            })
            print(f"  Selected Grouping: {selected_grouping}, Initial Cats: {initial_categories}, Reason: {memory.get('grouping_reasoning')[:50]}...")

        except Exception as e:
            print(f"  ERROR during Step 2: {e}")
            traceback.print_exc()
            error_message = f"Step 2 (Grouping Selection) failed: {e}"
            raise ValueError(error_message) # Stop if step 2 fails

        # --- Step 3: Refine Engine Categories ---
        print("\n--- Step 3: Engine Category Refinement ---")
        step3_input = {k: memory.get(k) for k in ["original_query", "expanded_query", "analysis", "selected_grouping", "initial_categories"]}
        try:
            raw_output_3 = chains["category_refinement_chain"].invoke(step3_input, config=llm_config)
            parsed_output_3 = parse_llm_json_output(raw_output_3, chains["category_refinement_parser"])
            if not parsed_output_3:
                raw_str_error = getattr(raw_output_3, 'content', str(raw_output_3))
                raise ValueError(f"Failed to parse Step 3 JSON. Raw output: {raw_str_error[:200]}...")

            refined_categories = parsed_output_3.get("refined_categories", [])
            if not isinstance(refined_categories, list):
                 print(f"  WARN: 'refined_categories' not a list: {refined_categories}. Using initial.")
                 refined_categories = memory.get("initial_categories", [])
            else:
                valid_refined_categories = []
                initial_set = set(memory.get("initial_categories", []))
                engine_cat_set = set(ENGINE_CATEGORIES)
                for cat in refined_categories:
                    if cat in initial_set and cat in engine_cat_set: valid_refined_categories.append(cat)
                    else: print(f"  WARN: Ignoring invalid refined category '{cat}'.")
                if not valid_refined_categories and memory.get("initial_categories"):
                    print("  WARN: Refinement empty/invalid. Falling back to initial categories.")
                    refined_categories = memory.get("initial_categories")
                elif not valid_refined_categories:
                    print("  WARN: Refinement resulted in empty list (initial was also empty).")
                    refined_categories = []
                else: refined_categories = valid_refined_categories

            memory.update({
                "refined_categories": refined_categories,
                "category_rationale": parsed_output_3.get("rationale", "")
            })
            print(f"  Refined Categories: {refined_categories}, Rationale: {memory.get('category_rationale')[:50]}...")

        except Exception as e:
            print(f"  ERROR during Step 3: {e}")
            traceback.print_exc()
            error_message = f"Step 3 (Category Refinement) failed: {e}"
            raise ValueError(error_message) # Stop if step 3 fails

        # --- Step 4: Map Categories to Engines ---
        print("\n--- Step 4: Mapping Categories to Engines ---")
        candidate_engines = []
        seen_ids = set()
        refined_categories_set = set(memory.get("refined_categories", []))
        if not refined_categories_set:
            error_message = "No refined categories available for mapping"
            print(f"  Skipping Step 4: {error_message}")
            raise ValueError(error_message)

        print(f"  Finding engines matching any of: {refined_categories_set}")
        for engine in all_engines:
            engine_id = engine.get("id")
            if not engine_id: print(f"  WARN: Skipping engine with missing ID: {engine.get('name', 'N/A')}"); continue
            engine_categories_set = set(engine.get("categories", []))
            if engine_categories_set.intersection(refined_categories_set):
                if engine_id not in seen_ids: candidate_engines.append(engine); seen_ids.add(engine_id)

        print(f"  Found {len(candidate_engines)} candidate engines: {[e.get('id') for e in candidate_engines]}")
        if not candidate_engines:
            error_message = "No engines matched the refined categories"
            print(f"  WARN: {error_message}")
            raise ValueError(error_message) # Stop if no candidates found

        memory.set("candidate_engine_ids", [engine.get("id") for engine in candidate_engines])

        # --- Step 5: Evaluate Individual Engines ---
        print(f"\n--- Step 5: Evaluating {len(candidate_engines)} Candidate Engines ---")
        common_context_keys = ["original_query", "expanded_query", "analysis", "selected_grouping", "refined_categories"]
        common_context = {k: memory.get(k) for k in common_context_keys}

        for i, engine in enumerate(candidate_engines):
            engine_id = engine.get("id", f"unknown_id_{i}")
            engine_name = engine.get("name", "Unknown Engine")
            print(f"\n  Evaluating ({i+1}/{len(candidate_engines)}): {engine_name} (ID: {engine_id})")
            step5_input = {
                **common_context, "engine_id": engine_id, "engine_name": engine_name,
                "engine_categories": engine.get("categories", []), "engine_type": engine.get("engine_type", "N/A"),
                "primary_use_case": engine.get("primary_use_case", "N/A"), "returns": engine.get("returns", "N/A"),
                "value_proposition": engine.get("value_proposition", "N/A"), "overlap": engine.get("overlap", "N/A"),
                "differentiation": engine.get("differentiation", "N/A"), "user_choice": engine.get("user_choice", "N/A"),
                "suitability": engine.get("suitability", "N/A")
            }
            try:
                raw_output_5 = chains["engine_decision_chain"].invoke(step5_input, config=llm_config)
                parsed_output_5 = parse_llm_json_output(raw_output_5, chains["engine_decision_parser"])
                if not parsed_output_5:
                    print(f"  WARN: Failed to parse decision for {engine_id}. Defaulting to EXCLUDE.")
                    decision, reasoning = "EXCLUDE", "Parsing failure"
                else:
                    decision = parsed_output_5.get("decision", "").strip().upper()
                    reasoning = parsed_output_5.get("reason", "No reason provided")

                print(f"  LLM Decision: {decision}, Reason: {reasoning[:50]}...")
                if decision == "INCLUDE": selected_engine_ids.append(engine_id); print(f"  --> Included: {engine_id}")
                elif decision == "EXCLUDE": print(f"  --> Excluded: {engine_id}")
                else: print(f"  WARN: Invalid decision '{decision}'. Excluded: {engine_id}")
            except Exception as e:
                print(f"  ERROR evaluating engine {engine_id}: {e}")
                traceback.print_exc()
                print(f"  --> Excluded (Evaluation Error): {engine_id}")

        # --- Step 6: Refine Selection with Comparison Data ---
        print("\n--- Step 6: Refining Selection with Comparison Data ---")
        if memory.get("refined_categories") and engine_compare_data:
            selected_engine_ids = refine_engines_with_comparison(
                selected_engine_ids,
                memory.get("refined_categories"),
                engine_compare_data
            )
            print(f"  Refined engine selection: {selected_engine_ids}")

        # --- Step 7: Finalize ---
        print("\n--- Step 7: Finalizing Engine Selection ---")
        if not selected_engine_ids:
            print("  WARN: No engines explicitly included by LLM.")
            candidate_ids = memory.get("candidate_engine_ids", [])
            if candidate_ids:
                selected_engine_ids = [candidate_ids[0]]  # Fallback to first candidate
                print(f"  Using first candidate as fallback: {selected_engine_ids[0]}")
            else:
                print("  No candidates identified, final list is empty.")
                error_message = error_message or "No engines selected and no candidates for fallback."

        memory.set("selected_engine_ids", selected_engine_ids)

    except ValueError as ve: # Catch specific errors raised within steps
        print(f"\n--- Orchestration Stopped Due to Error: {ve} ---")
        error_message = str(ve)
        # Ensure selected_engine_ids is empty if we errored out
        selected_engine_ids = []
    except Exception as e: # Catch unexpected errors
        print(f"\n--- UNEXPECTED ERROR during orchestration: {e} ---")
        traceback.print_exc()
        error_message = error_message or f"Orchestration failed unexpectedly: {e}"
        selected_engine_ids = []

    finally:
        end_time = time.time()
        print(f"\n--- Orchestration Complete (Took {end_time - start_time:.2f} seconds) ---")
        print("\n--- Final Results ---")
        print(f"Initial Query: {memory.get('original_query', 'N/A')}")
        # Add other memory items if useful for debugging
        print(f"Selected Grouping: {memory.get('selected_grouping', 'N/A')}")
        print(f"Refined Categories: {memory.get('refined_categories', 'N/A')}")
        print("-" * 20)
        if error_message: print(f"Error during processing: {error_message}")
        print(f"Selected Engine IDs ({len(selected_engine_ids)}):")
        print("[\n  " + ",\n  ".join(f'"{id_}"' for id_ in selected_engine_ids) + "\n]" if selected_engine_ids else "[]")
        print("-" * 20)

    # Ensure return types match annotation
    return selected_engine_ids, error_message


# --- Main Execution Block (Keep Previous Version) ---
if __name__ == "__main__":
    # (Keep the previous version of the main block)
    print("--- Initializing ---")
    if not check_files_exist(): exit(1)
    try:
        engines_data_raw = load_json_data(ENGINES_FILE)
        print(f"Loaded {len(engines_data_raw)} entries from '{ENGINES_FILE}'.")
        if not engines_data_raw: raise ValueError("Engine catalog is empty.")
        valid_engines = []
        for i, e in enumerate(engines_data_raw):
            if isinstance(e, dict) and e.get("id") and e.get("name") and isinstance(e.get("categories"), list):
                valid_engines.append(e)
            else: print(f"  WARN: Invalid engine entry index {i}: {str(e)[:100]}...")
        engines_data = valid_engines
        if not engines_data: raise ValueError(f"No valid engines found in '{ENGINES_FILE}'.")
        print(f"Using {len(engines_data)} valid engines.")

        print(f"Initializing Ollama LLM ({LLM_MODEL})...")
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, request_timeout=LLM_REQUEST_TIMEOUT, num_ctx=LLM_CTX)
        print("Testing LLM connection..."); test_output = llm.invoke("Respond OK"); print(f"LLM Test: {getattr(test_output, 'content', str(test_output))[:50]}...")

    except Exception as e:
        print(f"\nFATAL Initialization Error: {e}")
        traceback.print_exc()
        exit(1)

    print("\n--- Starting Interactive Session ---"); print("Type 'exit' or 'quit' to end.")
    memory_buffer = MemoryBuffer()
    while True:
        try:
            user_query_main = input("\nEnter Query: ")
            if user_query_main.lower() in ["exit", "quit"]: break
            if not user_query_main.strip(): continue
            select_engines_round_robin(user_query_main, llm, engines_data, memory_buffer)
        except KeyboardInterrupt: print("\nExiting..."); break
        except EOFError: print("\nInput stream closed. Exiting..."); break
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in main loop: {e}")
            traceback.print_exc()
            time.sleep(1)
    print("\n--- Session Ended ---")
