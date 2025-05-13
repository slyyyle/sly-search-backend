# text_analysis.py
# (File previously named text_summarize.py)
# --- Core Imports ---
import warnings
import os
import traceback
import time
import json
from typing import List, Optional, TypedDict, Annotated
import operator # For state updates

# Use Pydantic v1 namespace
from pydantic.v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, Runnable
# NEW Imports for Refine
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# NEW Imports for LangGraph
from langgraph.graph import StateGraph, END, START

# --- Configuration (Matching engine_round_robin.py) ---
LLM_MODEL = "llama3:8b-instruct-q8_0"
LLM_TEMPERATURE = 0.2
LLM_REQUEST_TIMEOUT = 180.0
LLM_CTX = 4096

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", message=".*The class `ChatOllama` was deprecated.*")

# --- Pydantic Model for Detailed Analysis Output ---
class TextAnalysisOutput(BaseModel):
    core_arguments: Optional[List[str]] = Field(default=None, description="List of strings, each representing a core argument or key point.")
    tone_sentiment: Optional[str] = Field(default=None, description="Description of the dominant tone and overall sentiment, with justification.")
    implications: Optional[List[str]] = Field(default=None, description="List of strings, each detailing a potential implication or consequence based purely on the text content.")
    bias_analysis: Optional[str] = Field(default=None, description="Analysis of potential biases (author, selection, confirmation, framing, language) or statement of none apparent.")
    underlying_assumptions: Optional[List[str]] = Field(default=None, description="List of strings, each identifying an unspoken assumption or condition.")
    alternative_perspectives: Optional[List[str]] = Field(default=None, description="List of strings, each outlining a plausible alternative viewpoint or counterargument.")
    factual_synthesis: str = Field(description="Synthesis connecting key facts/arguments and their relationships based *only* on the provided text.")
    source_implications: Optional[List[str]] = Field(default=None, description="List of strings, each detailing potential financial, political, or social implications (or lack thereof) considering the text came from the stated source.")

# --- LangGraph State Definition ---
class AnalysisState(TypedDict):
    # Holds the content of all document chunks
    doc_contents: List[str]
    # The index of the *next* document chunk to process for refinement
    current_doc_index: int
    # The analysis produced so far, passed between refine steps
    current_analysis: str
    # Configuration for LLM calls (optional but good practice)
    config: Optional[RunnableConfig]
    # Store the LLM instance itself (can be passed in initial state)
    llm: ChatOllama
    # Store prompts (can be passed in initial state)
    initial_prompt: ChatPromptTemplate
    refine_prompt: ChatPromptTemplate

# --- LangGraph Node Functions ---
def initial_analysis_step(state: AnalysisState) -> dict:
    """Processes the first document chunk."""
    print(f"  Processing chunk 1 of {len(state['doc_contents'])}... (LangGraph Node)")
    initial_chain = state["initial_prompt"] | state["llm"] | JsonOutputParser() # Assuming initial works well with direct JSON parsing now

    # Invoke directly on the first chunk's content
    # Use RunnableConfig if passed in state, otherwise None
    config = state.get("config")
    first_doc_content = state["doc_contents"][0]
    initial_result_dict = initial_chain.invoke({"text": first_doc_content}, config=config)

    # Need to pass the result as a JSON string for the next step's prompt
    initial_analysis_str = json.dumps(initial_result_dict)

    return {"current_analysis": initial_analysis_str, "current_doc_index": 1}

def refine_analysis_step(state: AnalysisState) -> dict:
    """Processes subsequent document chunks, refining the analysis."""
    idx = state["current_doc_index"]
    print(f"  Refining with chunk {idx + 1} of {len(state['doc_contents'])}... (LangGraph Node)")
    refine_chain = state["refine_prompt"] | state["llm"] | StrOutputParser() # Refine still outputs string

    config = state.get("config")
    doc_content = state["doc_contents"][idx]
    existing_answer = state["current_analysis"]

    refined_analysis_str = refine_chain.invoke(
        {"existing_answer": existing_answer, "text": doc_content},
        config=config
    )

    return {"current_analysis": refined_analysis_str, "current_doc_index": idx + 1}

# --- LangGraph Conditional Edge Logic ---
def should_continue_refining(state: AnalysisState) -> str:
    """Determines whether to continue refining or end."""
    if state["current_doc_index"] >= len(state["doc_contents"]):
        print("  All chunks processed. Ending graph.")
        return END
    else:
        return "refine_analysis_node"

# --- Text Analysis Function (Refactored to use LangGraph) ---
def analyze_text_langgraph(text_to_analyze: str, source_name: str, llm: ChatOllama, chunk_size: int = 3000, chunk_overlap: int = 300) -> tuple[Optional[dict], Optional[List[dict]], Optional[str]]:
    """Analyzes the given text using LangGraph for the Refine method."""
    print(f"\n--- Performing Analysis (LangGraph Refine) for text from '{source_name}' starting with: '{text_to_analyze[:30]}...' ---")
    overall_start_time = time.time()
    graph_timings = [] # Store overall graph time for now
    analysis_result: Optional[dict] = None
    error_message: Optional[str] = None

    # Prepend source information
    full_text_with_source = f"Source: {source_name}\n\n{text_to_analyze}"

    # --- Define Prompts (same as before) ---
    analysis_instructions = """
Instructions for Analysis (of text originally from the stated source):
1.  **Identify Core Arguments/Key Points:** Clearly list the main arguments, claims, or key pieces of information presented.
2.  **Determine Overall Tone/Sentiment:** Describe the dominant tone and sentiment with justification.
3.  **Uncover Potential Implications (from text content):** Note potential consequences or outcomes suggested purely by the text content itself.
4.  **Examine Possible Biases:** Evaluate potential author, selection, confirmation, framing, or language biases. State if none are apparent.
5.  **Identify Underlying Assumptions/Conditions:** List unspoken assumptions or conditions required for claims to hold.
6.  **Consider Alternative Perspectives/Counterarguments:** Outline plausible alternative viewpoints.
7.  **Provide a Factual Synthesis:** Synthesize the key facts identified in the core arguments, explaining significant relationships or connections between them based *only* on the provided text. Explicitly state any significant figures, statistics, or quantifiable data mentioned rather than summarizing them. **The value for this field MUST be a single JSON string.** Avoid introducing outside information or broad interpretations not directly supported by the text itself.
8.  **Analyze Source Implications:** Specifically consider that the text came from the stated source. Analyze potential financial, political, or social implications (or lack thereof) suggested by the information *given its source*. Avoid speculation; base analysis on plausible connections between the text content and the source's typical context or influence.
"""
    json_output_instructions = """
Output Format Instructions:
Generate ONLY a valid JSON object containing the following keys, with appropriate string or list-of-string values:
- "core_arguments" (List[str] or null)
- "tone_sentiment" (str or null)
- "implications" (List[str] or null)
- "bias_analysis" (str or null)
- "underlying_assumptions" (List[str] or null)
- "alternative_perspectives" (List[str] or null)
- "factual_synthesis" (**Must be a single JSON string**)
- "source_implications" (List[str] or null)
Ensure the output is a single, valid JSON object and nothing else.
"""
    initial_prompt_template = f"""
You are an expert text analyst. Analyze the following text chunk (which includes its original source) according to the instructions below.

TEXT CHUNK (including source info):
```text
{{text}}
```

{analysis_instructions}

{json_output_instructions}
"""
    initial_prompt = ChatPromptTemplate.from_template(initial_prompt_template)
    refine_prompt_template = f"""
You are an expert text analyst. You have provided an existing analysis up to a certain point. Your task is to refine this existing analysis using the additional context (which includes its original source) below. Ensure the final output incorporates insights from BOTH the existing analysis and the new context, adhering to the required JSON format.

EXISTING ANALYSIS (JSON Object String):
```json
{{existing_answer}}
```

ADDITIONAL CONTEXT (New Text Chunk, including source info):
```text
{{text}}
```

Refinement Instructions:
- Carefully integrate the key information and nuances from the ADDITIONAL CONTEXT into the EXISTING ANALYSIS.
- Update each field of the JSON object (core_arguments, tone_sentiment, implications, source_implications, bias_analysis, underlying_assumptions, alternative_perspectives, factual_synthesis) to reflect the combined understanding from all text seen so far.
- Ensure the factual_synthesis accurately connects key facts and relationships from the *entire* text processed, explicitly stating significant figures/statistics rather than summarizing them. **The value for the 'factual_synthesis' key must be a single JSON string.**
- Pay specific attention to updating the source_implications based on the cumulative context.
- Ensure the final output is a SINGLE, valid JSON object string following the specified format, representing the complete analysis up to this point.

{json_output_instructions}
"""
    refine_prompt = ChatPromptTemplate.from_template(refine_prompt_template)

    # --- Build and Run Graph ---
    final_output_string = ""
    try:
        # 1. Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_text(full_text_with_source)
        if not texts:
            raise ValueError("Text splitting resulted in no chunks.")
        print(f"  Split text into {len(texts)} chunks.")

        # 2. Define Graph
        workflow = StateGraph(AnalysisState)
        workflow.add_node("initial_analysis_node", initial_analysis_step)
        workflow.add_node("refine_analysis_node", refine_analysis_step)

        # 3. Define Edges
        workflow.set_entry_point("initial_analysis_node")
        workflow.add_conditional_edges(
            "initial_analysis_node",
            should_continue_refining,
            # If only one chunk, initial output goes straight to END,
            # otherwise to refine node.
            # Note: END goes to final state implicitly.
            {
                END: END, # Map END signal directly to END state
                "refine_analysis_node": "refine_analysis_node" # Route to refine node otherwise
            }
        )
        workflow.add_conditional_edges(
            "refine_analysis_node",
            should_continue_refining,
            # After refining, check again if we need to refine more or end.
            {
                END: END, # Map END signal directly to END state
                "refine_analysis_node": "refine_analysis_node" # Route back to refine node if needed
            }
        )

        # 4. Compile Graph
        app = workflow.compile()

        # 5. Prepare Initial State and Invoke
        initial_state: AnalysisState = {
            "doc_contents": texts,
            "current_doc_index": 0, # Start before the first doc for initial step
            "current_analysis": "", # Start with empty analysis
            "config": RunnableConfig(configurable={"session_id": f"analyze-lg-{time.time()}"}),
            "llm": llm,
            "initial_prompt": initial_prompt,
            "refine_prompt": refine_prompt
        }
        print("  Invoking LangGraph...")
        final_state = app.invoke(initial_state)

        # The final analysis string is in the state after the graph runs
        final_output_string = final_state.get("current_analysis", "")

        # 6. Parse the final output string as JSON
        print("  Parsing final JSON output...")
        json_start_index = -1
        cleaned_output = final_output_string.strip()

        brace_index = cleaned_output.find('{')
        bracket_index = cleaned_output.find('[')

        if brace_index != -1 and bracket_index != -1:
            json_start_index = min(brace_index, bracket_index)
        elif brace_index != -1:
            json_start_index = brace_index
        elif bracket_index != -1:
            json_start_index = bracket_index

        if json_start_index == -1:
             raise json.JSONDecodeError("Could not find start of JSON object ({ or [) in the output.", cleaned_output, 0)

        json_string_part = cleaned_output[json_start_index:]

        if json_string_part.startswith('{') and json_string_part.rfind('}') != -1:
             json_string_part = json_string_part[:json_string_part.rfind('}') + 1]
        elif json_string_part.startswith('[') and json_string_part.rfind(']') != -1:
             json_string_part = json_string_part[:json_string_part.rfind(']') + 1]

        final_json = json.loads(json_string_part)
        parsed_analysis = TextAnalysisOutput.parse_obj(final_json)
        analysis_result = parsed_analysis.dict()

    except json.JSONDecodeError as json_e:
        print(f"  ERROR: Failed to parse the final JSON output: {json_e}")
        print("  Raw output string from graph:")
        print(final_output_string)
        error_message = f"Error during final JSON parsing: {json_e}. See console for raw output."
    except Exception as e:
        print(f"  ERROR during graph execution or parsing: {e}")
        traceback.print_exc()
        error_message = f"Error during analysis: {e}"
    finally:
        overall_end_time = time.time()
        total_duration = overall_end_time - overall_start_time
        # Simpler timing for now, just total graph execution
        graph_timings = [{"step": "Total LangGraph Analysis", "duration": total_duration}]
        print(f"--- Analysis Finished (Total time: {total_duration:.2f} seconds) ---")

    # Return tuple matching previous structure
    return analysis_result, graph_timings, error_message

# --- Main Execution Block for Testing (Updated to call new function) ---
if __name__ == "__main__":
    print("--- Initializing Text Analyzer (LangGraph Refine) ---")
    try:
        print(f"Initializing Ollama LLM ({LLM_MODEL})...")
        llm_analyze = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, request_timeout=LLM_REQUEST_TIMEOUT, num_ctx=LLM_CTX)
        print("Testing LLM connection...")
        test_output = llm_analyze.invoke("Respond OK")
        print(f"LLM Test: {getattr(test_output, 'content', str(test_output))[:50]}...")
        print("Initialization successful.")
    except Exception as e:
        print(f"\nFATAL Initialization Error: {e}")
        traceback.print_exc()
        exit(1)

    print("\n--- Starting Interactive Analysis Session ---")
    print("Enter long text to analyze (or paste). Type 'EOF' (or Ctrl+D) on a new line when done, then press Enter.")
    print("Type 'exit' or 'quit' to end the program.")

    while True:
        print("\nEnter text (end with EOF or Ctrl+D on a new line):")
        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() in ["exit", "quit"]:
                    lines = [line.strip().lower()]
                    break
                if line.strip().upper() == "EOF":
                    break
                lines.append(line)
        except EOFError:
            pass

        user_text = "\n".join(lines)

        if user_text.lower().strip() in ["exit", "quit"]:
             print("\nExiting...")
             break
        if not user_text.strip():
            print("No text entered.")
            continue

        # Call the new LangGraph function
        analysis_result, timing_data, error = analyze_text_langgraph(user_text, source_name="Wikipedia", llm=llm_analyze)

        print(f"\nAnalysis Result:\n-----------------")
        if error:
            print(f"ERROR: {error}")
        elif analysis_result:
            print(json.dumps(analysis_result, indent=2))
        else:
            print("No analysis result produced.")
        print("-----------------")

        if timing_data:
            print("\nTiming Breakdown (seconds):\n---------------------------")
            for step in timing_data:
                print(f"  - {step['step']:<25}: {step['duration']:>7.2f}") # Adjusted padding
            print("---------------------------")

    print("\n--- Session Ended ---") 