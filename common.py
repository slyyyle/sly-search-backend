# common.py
import json
import os
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set

# Use Pydantic v1 namespace
from pydantic.v1 import BaseModel, Field, ValidationError
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
LLM_NUM_CTX = 4096 # Kept 4096 as requested previously

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", message=".*The class `ChatOllama` was deprecated.*")

# --- Category Groupings & Mapping (Updated & Simplified) ---
CATEGORY_GROUPINGS = ["information", "images", "music/videos", "news", "torrents", "software/tools", "gardens/museums/archives"]

# More concise implications
CATEGORY_GROUPINGS_IMPLICATIONS = {
    "information": "General info, articles, text. Not specific media files.",
    "images": "Searching for image files (photos, art).",
    "music/videos": "Searching for playable music or video files/streams.",
    "news": "Recent news articles, current events (explicitly mentioned).",
    "torrents": "Searching for .torrent files / P2P links (explicitly mentioned).",
    "software/tools": "Searching for computer programs, tools (non-torrent).",
    "gardens/museums/archives": "Niche, fun, old, retro, obscure content."
}

# Engine Categories (Removed wiki, qa, code)
ENGINE_CATEGORIES = ["general", "it", "social media", "science", "images", "videos", "files", "music"]

CATEGORY_FILTER_MAP = {
    # Maps GROUPINGS to ENGINE_CATEGORIES
    "information": ["general", "it", "social media", "science"],
    "images": ["images", "general"],
    "music/videos": ["music", "videos", "general"],
    "news": ["news", "general"],
    "torrents": ["files"],
    "software/tools": ["it", "general"],
    "gardens/museums/archives": ["general", "social media"], # Simplified mapping
}

# Simplified Engine Category Descriptions (for Step 4)
ENGINE_CATEGORY_IMPLICATIONS = {
    "general": "Broad web search, text, some media links, retro/niche content.",
    "it": "Software, code documentation, package managers, tech help.",
    "social media": "Forums, discussions, human opinions/experiences.",
    "science": "Academic research, papers, math/science questions.",
    "images": "Dedicated image file search.",
    "videos": "Dedicated video file/stream search.",
    "files": "Torrent/P2P file search ONLY.",
    "music": "Dedicated music file/stream search.",
}

# --- Pydantic Models ---
# Renamed field in Step 1.1 output model
class Step1_1_BestCategoryOutput(BaseModel):
    best_categories: List[str] = Field(description=f"List of BEST fitting CATEGORY GROUPINGS. Subset of {CATEGORY_GROUPINGS}.", default_factory=list)

class Step1_2_ClarificationOutput(BaseModel):
    clarification_needed: bool = Field(default=False)
    clarification_question: Optional[str] = Field(default=None)
    query_terms_for_expansion: str

class RefinedCategoriesOutput(BaseModel):
    refined_categories: List[str] = Field(description=f"List of final relevant CATEGORY GROUPINGS. Subset of {CATEGORY_GROUPINGS}.", default_factory=list)

class FinalTargetCategoriesOutput(BaseModel):
    final_target_categories: List[str] = Field(description=f"List of refined ENGINE_CATEGORIES to use for filtering. Subset of {ENGINE_CATEGORIES}.", default_factory=list)

# --- Utility Functions (Keep as is) ---
def check_files_exist():
    if not os.path.exists(ENGINES_FILE): print(f"CRITICAL ERROR: '{ENGINES_FILE}' not found."); return False
    return True
def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e: raise RuntimeError(f"Error loading {filepath}: {e}")
def parse_llm_json_output(raw_output: Any, parser: JsonOutputParser) -> Optional[Dict]:
    # Simplified parser, relies more on LLM getting it right
    if isinstance(raw_output, dict): return raw_output
    if isinstance(raw_output, str):
        try:
            # Attempt markdown stripping first
            if raw_output.strip().startswith("```json"):
                json_block = raw_output.strip()[7:]
                if json_block.endswith("```"): json_block = json_block[:-3]
                try:
                    # print("  DEBUG: Parsing JSON from markdown block.")
                    return json.loads(json_block)
                except json.JSONDecodeError: pass # Fall through if markdown parse fails
            return json.loads(raw_output) # Try direct parse
        except json.JSONDecodeError:
            print(f"  WARN: Direct JSON parsing failed for: {raw_output[:100]}... Trying Pydantic.")
            try:
                return parser.parse(raw_output).dict()
            except Exception as p_err:
                print(f"  ERROR: Pydantic parsing failed: {p_err}")
                return None
    # Handle if LLM directly returns the pydantic object (less common)
    try: return raw_output.dict()
    except AttributeError: print(f"  ERROR: Unexpected output type for parsing: {type(raw_output)}"); return None


# --- LCEL Chain Creation (Simplified Prompts) ---
def create_all_chains(llm: ChatOllama):
    """Creates all LCEL chains."""

    # Static context preparation
    category_groupings_list_str = str(CATEGORY_GROUPINGS)
    category_groupings_implications_json = json.dumps(CATEGORY_GROUPINGS_IMPLICATIONS, indent=2)
    engine_category_implications_json = json.dumps(ENGINE_CATEGORY_IMPLICATIONS, indent=2)

    # --- Step 1 Chains ---
    step1_1_prompt_template = """
Task: Identify BEST fitting CATEGORY GROUPINGS for USER QUERY.
Options: {category_groupings_list_str}
USER QUERY: {{query}}
Output ONLY JSON: {{"best_categories": ["list", "of", "strings"]}} (No markdown, preamble, or explanation)
"""
    step1_1_prompt_tmpl = ChatPromptTemplate.from_template(step1_1_prompt_template)
    step1_1_prompt = step1_1_prompt_tmpl.partial(category_groupings_list_str=category_groupings_list_str)
    step1_1_parser = JsonOutputParser(pydantic_object=Step1_1_BestCategoryOutput)
    step1_1_category_chain = step1_1_prompt | llm | step1_1_parser

    step1_2_prompt_template = """
Task: Check for conflicting implications in BEST_CATEGORY_GROUPINGS. If conflict, ask clarifying question. Extract core query terms.
Implications: {category_groupings_implications_json}
Inputs:
USER QUERY: {{query}}
BEST_CATEGORY_GROUPINGS: {{best_categories_list}}
Output ONLY JSON: {{"clarification_needed": boolean, "clarification_question": string|null, "query_terms_for_expansion": string}} (No markdown)
"""
    step1_2_prompt_tmpl = ChatPromptTemplate.from_template(step1_2_prompt_template)
    step1_2_prompt = step1_2_prompt_tmpl.partial(category_groupings_implications_json=category_groupings_implications_json)
    step1_2_parser = JsonOutputParser(pydantic_object=Step1_2_ClarificationOutput)
    step1_2_clarification_chain = step1_2_prompt | llm | step1_2_parser
    # --- End Step 1 Chains ---

    # --- Step 2 Chain ---
    refinement_prompt_template = """
Task: Refine BEST_CATEGORY_GROUPINGS based on USER_ANSWER.
Inputs:
PLAUSIBLE_CATEGORIES: {{plausible_categories}} # Original groupings
CLARIFICATION_QUESTION: {{clarification_question}}
USER_ANSWER: {{user_answer}}
Output ONLY JSON: {{"refined_categories": ["final", "groupings", "list"]}} (No markdown)
"""
    refinement_prompt = ChatPromptTemplate.from_template(refinement_prompt_template)
    refinement_parser = JsonOutputParser(pydantic_object=RefinedCategoriesOutput)
    category_refinement_chain = refinement_prompt | llm | refinement_parser
    # --- End Step 2 Chain ---

    # --- Step 3 Chain ---
    expansion_analysis_prompt_template = """
Task: Analyze QUERY_TERMS based on REFINED_CATEGORIES (groupings).
Inputs:
QUERY_TERMS: {{query_terms}}
REFINED_CATEGORIES: {{refined_categories}}
Output ONLY concise analysis string.
Analysis Output:
"""
    expansion_analysis_prompt = ChatPromptTemplate.from_template(expansion_analysis_prompt_template)
    expansion_analyzer_chain = expansion_analysis_prompt | llm | StrOutputParser()
    # --- End Step 3 Chain ---

    # --- Step 4 Chain ---
    category_target_refiner_prompt_template = """
Task: Refine target ENGINE_CATEGORIES based on groupings, analysis, and query terms.
Context:
Query Terms: {{query_terms}}
Refined Groupings: {{refined_user_intents}}
Potential Engine Categories: {{potential_target_engine_categories}}
Analysis: {{analysis_or_considerations}}
Engine Implications: {engine_category_implications_json}

Goal: Select best subset of Potential Engine Categories using Analysis.
Output ONLY JSON: {{"final_target_categories": ["engine", "category", "list"]}} (No markdown)
"""
    category_target_refiner_tmpl = ChatPromptTemplate.from_template(category_target_refiner_prompt_template)
    category_target_refiner_prompt = category_target_refiner_tmpl.partial(
         engine_category_implications_json=engine_category_implications_json
    )
    category_target_refiner_parser = JsonOutputParser(pydantic_object=FinalTargetCategoriesOutput)
    category_target_refiner_chain = category_target_refiner_prompt | llm | category_target_refiner_parser
    # --- End Step 4 Chain ---

    # --- Step 5 Chain ---
    engine_decision_prompt_template = """
Task: Decide INCLUDE or EXCLUDE for the SEARCH ENGINE based on CONTEXT.
CONTEXT:
Query Terms: {{original_query_terms}}
Refined Groupings: {{refined_categories}} (Primary: {{primary_refined_category}})
Analysis: {{analysis_or_considerations}}
SEARCH ENGINE DETAILS:
Name: {{engine_name}}
Categories: {{engine_categories}}
Suitability: {{suitability}}

Output ONLY the word "INCLUDE" or "EXCLUDE". No other text.
Decision (INCLUDE or EXCLUDE):
"""
    engine_decision_prompt = ChatPromptTemplate.from_template(engine_decision_prompt_template)
    engine_decider_chain = engine_decision_prompt | llm | StrOutputParser()
    # --- End Step 5 Chain ---

    return {
        "step1_1_category_chain": step1_1_category_chain,
        "step1_1_parser": step1_1_parser, # JsonOutputParser
        "step1_2_clarification_chain": step1_2_clarification_chain,
        "step1_2_parser": step1_2_parser,
        "category_refinement_chain": category_refinement_chain,
        "refinement_parser": refinement_parser,
        "expansion_analyzer_chain": expansion_analyzer_chain,
        "category_target_refiner_chain": category_target_refiner_chain,
        "category_target_refiner_parser": category_target_refiner_parser,
        "engine_decider_chain": engine_decider_chain,
    }
