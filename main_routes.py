# main_routes.py
import logging
import urllib.parse
import os
from pathlib import Path as FilePath
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Path, Request
from fastapi.responses import FileResponse

# Import models, utils, and config needed for routes
# Use direct imports
import main_models as models
import main_utils as utils
import main_config as config 

logger = logging.getLogger(__name__)

# Create an API router for these core application routes
router = APIRouter()

# --- Root Endpoint ---
@router.get("/", tags=["Status"]) # Add a tag for documentation
async def read_root():
    """Root endpoint providing a basic status message."""
    return {"message": "SlySearch Backend API is running."}

# --- Search Endpoint ---
# Define the response model explicitly covering all possible search types + generic web search
SearchResponseUnion = Union[
    models.ObsidianSearchResponse,
    models.YouTubeSearchResponse,
    models.PhotosSearchResponse,
    models.MusicSearchResponse,
    Dict[str, Any] # For SearXNG web search results (which are typically Dict[str, Any])
]

@router.get("/search", response_model=SearchResponseUnion, tags=["Search"])
async def search(
    request: Request, # Inject request object if needed (e.g., for client headers, though not used here yet)
    q: str = Query(..., min_length=1, description="The search query term"),
    source: str = Query('web', description="Data source ID (e.g., web, obsidian, youtube, photos, music). Defaults to 'web'."),

    # --- SearXNG specific parameters ---
    category: Optional[str] = Query(None, description="SearXNG category filter (e.g., general, images, news)"),
    pageno: Optional[int] = Query(1, ge=1, description="Page number for results (applies to all sources)"),
    time_range: Optional[str] = Query(None, description="SearXNG: Time range filter (e.g., day, week, month, year)"),
    language: Optional[str] = Query(None, description="SearXNG: Language code (e.g., en, de, auto)"),
    safesearch: Optional[int] = Query(None, ge=0, le=2, description="SearXNG: Safe search level (0=off, 1=moderate, 2=strict)"),
    engines: Optional[str] = Query(None, description="Comma-separated list of specific SearXNG engine IDs to use (overrides active loadout/category filtering)"),

    # --- General parameters applicable to potentially multiple sources ---
    results: Optional[int] = Query(None, ge=1, le=100, description="Number of results per page (overrides source-specific or general setting)"),
    rag: Optional[str] = Query(None, description="Flag to enable RAG enhancement for web results (e.g., 'enabled')") # Simple flag for now
) -> SearchResponseUnion:
    """
    Performs a search based on the specified source ('web', 'obsidian', 'youtube', 'photos', 'music').
    Routes to SearXNG for 'web' source or specific local search functions otherwise.
    Applies pagination and relevant filters based on query parameters and application settings.
    """
    logger.info(f"Received search request: Query='{q}', Source='{source}', Category='{category}', Page='{pageno}'")

    current_settings = utils.load_settings() # Load current settings for source paths, defaults etc.
    actual_source = source.lower().strip() if source else 'web' # Normalize source ID, default to web

    try:
        # ===========================
        # --- Web Search (SearXNG) ---
        # ===========================
        if actual_source == 'web':
            # -- Determine SearXNG Engines to Use --
            effective_engines_str: Optional[str] = None # The final string to pass to SearXNG
            engine_param_received = engines is not None
            logger.debug(f"Received 'engines' parameter from request: {engines}")

            if engine_param_received:
                # If specific engines are requested via query param, use them directly.
                # No further filtering by category or loadout is done in this case.
                effective_engines_str = engines
                logger.info(f"Using engines specified directly in request parameter: '{effective_engines_str}'")
            else:
                # If no specific engines provided, determine from active loadout and category (if provided).
                try:
                    # 1. Get all available engines definitions from catalog file.
                    available_engines_list: List[models.EngineDetail] = await utils.get_searxng_engines()
                    if not available_engines_list:
                         logger.warning("No engines loaded from catalog. SearXNG instance might use its own internal defaults or fail.")
                         # effective_engines_str remains None
                    else:
                        # 2. Get the active loadout configuration (a list of dicts with merged catalog defaults + saved settings).
                        active_engines_config: List[Dict[str, Any]] = utils.get_active_engine_config(current_settings, available_engines_list)

                        # 3. Determine the active loadout name and configuration dictionary
                        active_loadout_name = current_settings.get('activeLoadout', 'default') # Default to 'default' if not set
                        all_loadouts = current_settings.get('engineLoadouts', {})
                        active_loadout_config = all_loadouts.get(active_loadout_name, {}) # Get the specific loadout dict, or empty if not found
                        # Make sure loadout_name is defined for later use
                        loadout_name = active_loadout_name
                        if not active_loadout_config:
                             logger.warning(f"Active loadout '{active_loadout_name}' specified in settings not found in 'engineLoadouts'. Engine selection might be incomplete.")
                             # Proceed with empty config, logic should handle defaults/fallbacks

                        # 4. Filter these active engines further by the requested category (if a category is provided in the query).
                        if category:
                            # Map 'web' category alias to 'general' if SearXNG uses that, or handle specific categories.
                            # Normalize category for comparison.
                            actual_filter_category = 'general' if category.lower() == 'web' else category.lower()
                            logger.info(f"Filtering active engines for category: '{actual_filter_category}'")

                            filtered_engine_ids = []
                            for engine in active_engines_config:
                                # Check if the engine is enabled in the *active loadout*
                                if engine.get('enabled', False):
                                    engine_categories = [cat.lower() for cat in engine.get('categories', [])]
                                    # Check if the requested category is in the engine's categories
                                    if actual_filter_category in engine_categories:
                                        filtered_engine_ids.append(engine['id'])
                                    # Special case: Allow 'general' category to also match engines explicitly categorized as 'web'
                                    elif actual_filter_category == 'general' and 'web' in engine_categories:
                                        filtered_engine_ids.append(engine['id'])

                            # If filtering by category yielded results, use them.
                            if filtered_engine_ids:
                                effective_engines_str = ','.join(filtered_engine_ids)
                                logger.info(f"Filtered active engines for category '{actual_filter_category}': {effective_engines_str}")
                            else:
                                # If no engines match the category, send no specific engines (SearXNG might default or fail category search).
                                logger.warning(f"No active and enabled engines found for category '{actual_filter_category}' in the current loadout. SearXNG might use defaults or return no results for this category.")
                                effective_engines_str = None # Explicitly send no engines to SearXNG
                        else:
                            # If no category was provided:
                            # Check if specific engines were requested via the 'engines' parameter.
                            if engines:
                                # If 'engines' param was given, use those specific engines (already handled above)
                                # This 'else' branch shouldn't strictly be reached if 'engines' param is present due to earlier check,
                                # but keeping logic clear. Assuming effective_engines_str was set earlier.
                                logger.info(f"Category not specified, but specific engines requested: {engines}. Using those.")
                                # effective_engines_str should already be set from the 'if engines:' block near the top
                            else:
                                # NO category AND NO specific engines requested. Default to 'general' category from the *active loadout*.
                                logger.info("No category or specific engines provided. Attempting to use 'general' category engines from active loadout.")
                                try:
                                    # Correctly access categories within the active loadout
                                    general_category_config = active_loadout_config.get('categories', {}).get('general', {})
                                    general_engines = general_category_config.get('engines', []) # List of engine IDs for 'general' category

                                    # Filter 'general' engines against the list of *globally enabled* engines in the active loadout
                                    globally_enabled_engine_ids = {eng['id'] for eng in active_engines_config if eng.get('enabled', False)}
                                    effective_general_engine_ids = [eid for eid in general_engines if eid in globally_enabled_engine_ids]

                                    if effective_general_engine_ids:
                                        effective_engines_str = ','.join(effective_general_engine_ids)
                                        logger.info(f"Using enabled engines from 'general' category within active loadout '{loadout_name}': {effective_engines_str}")
                                    else:
                                        logger.warning(f"The 'general' category in active loadout '{loadout_name}' has no engines defined or none are enabled globally. Falling back to all globally enabled engines.")
                                        # Fallback: Use all enabled engines from the active loadout.
                                        enabled_engine_ids = list(globally_enabled_engine_ids) # Reuse the set
                                        if enabled_engine_ids:
                                            effective_engines_str = ','.join(enabled_engine_ids)
                                            logger.info(f"Fallback: Using all globally enabled engines from active loadout '{loadout_name}': {effective_engines_str}")
                                        else:
                                            logger.warning(f"Fallback failed: No engines are enabled in the active loadout '{loadout_name}' at all. SearXNG will likely use its defaults.")
                                            effective_engines_str = None
                                except Exception as e:
                                     logger.error(f"Error accessing 'general' category engines in active loadout '{loadout_name}': {e}. Falling back to all enabled engines.", exc_info=True)
                                     # Fallback: Use all enabled engines from the active loadout.
                                     try: # Nested try in case active_engines_config is weird
                                         # Recalculate globally enabled engines just in case something went wrong before
                                         globally_enabled_engine_ids_fallback = {eng['id'] for eng in active_engines_config if eng.get('enabled', False)}
                                         enabled_engine_ids = list(globally_enabled_engine_ids_fallback)

                                         if enabled_engine_ids:
                                             effective_engines_str = ','.join(enabled_engine_ids)
                                             logger.info(f"Fallback due to error: Using all enabled engines from active loadout '{loadout_name}': {effective_engines_str}")
                                         else:
                                             logger.warning(f"Fallback failed after error: No engines enabled in active loadout '{loadout_name}'. SearXNG defaults.")
                                             effective_engines_str = None
                                     except Exception as inner_e:
                                         logger.error(f"Critical error during fallback engine selection for loadout '{loadout_name}': {inner_e}. Letting SearXNG use defaults.", exc_info=True)
                                         effective_engines_str = None

                except Exception as e:
                    # Catch errors during engine loading/filtering process.
                    logger.error(f"Error determining active/filtered engines for SearXNG: {e}. Letting SearXNG use defaults.", exc_info=True)
                    effective_engines_str = None # Ensure it's None if error occurs before assignment

            # Log final decision on engines parameter
            if effective_engines_str: logger.debug(f"Final 'engines' parameter value being sent to SearXNG: '{effective_engines_str}'")
            else: logger.debug("No specific 'engines' parameter being sent to SearXNG (will use SearXNG defaults/config).")

            # -- Determine Results Per Page for SearXNG --
            # Priority: Query param `results` > General Setting `resultsPerPage` > Fallback default (e.g., 10)
            results_per_page = results # Use request param `results` if provided
            if results_per_page is None:
                try:
                     # Use general setting, converting to int
                     results_per_page = int(current_settings.get('general', {}).get('resultsPerPage', 10))
                except (ValueError, TypeError):
                     results_per_page = 10 # Fallback if setting is invalid
            # Ensure results_per_page is within reasonable bounds if needed
            results_per_page = max(1, min(results_per_page, 100)) # Example bounds: 1-100
            logger.debug(f"Setting SearXNG results_per_page to: {results_per_page}")

            # -- Determine Language for SearXNG --
            # Priority: Query param `language` > General Setting `defaultLanguage` > Fallback ('all' or SearXNG default)
            lang_setting = language # Use query param `language` if provided
            if lang_setting is None:
                 lang_setting = current_settings.get('general', {}).get('defaultLanguage', 'all') # Use setting or default 'all'
            logger.debug(f"Setting SearXNG language to: {lang_setting}")

            # -- Determine Safe Search for SearXNG --
            # Priority: Query param `safesearch` > General Setting `safeSearch` > Fallback (0)
            safe_search_setting = safesearch # Use query param `safesearch` if provided
            if safe_search_setting is None:
                 try:
                     # Use general setting, converting to int
                     safe_search_setting = int(current_settings.get('general', {}).get('safeSearch', 0))
                 except (ValueError, TypeError):
                     safe_search_setting = 0 # Fallback if setting is invalid
            # Ensure safesearch value is within SearXNG's expected range (typically 0, 1, 2)
            if safe_search_setting not in [0, 1, 2]:
                 logger.warning(f"Invalid safeSearch value '{safe_search_setting}' received or configured. Defaulting to 0 (off).")
                 safe_search_setting = 0 # Default to off if value is invalid
            logger.debug(f"Setting SearXNG safeSearch to: {safe_search_setting}")

            # -- Build Final SearXNG Parameters Dictionary --
            searxng_params: Dict[str, Any] = {
                "pageno": pageno,
                "time_range": time_range, # Pass through directly if provided
                "language": lang_setting,
                "safesearch": safe_search_setting,
                "engines": effective_engines_str, # Pass the determined string (can be None)
                "results_per_page": str(results_per_page) # SearXNG often expects results count as string
                # Category is passed only if explicitly requested in the original query
            }
            if category:
                 logger.debug(f"Adding category '{category}' to SearXNG params.")
                 # Map 'web' back to 'general' if that's what SearXNG uses internally
                 searxng_params["category"] = 'general' if category.lower() == 'web' else category

            # Filter out any parameters that ended up being None before calling SearXNG
            searxng_params = {k: v for k, v in searxng_params.items() if v is not None}

            # --- Call SearXNG via utility function ---
            search_results = await utils.query_searxng(q, searxng_params)

            # Add source identifier to the response for the frontend
            search_results['source'] = 'web' # Mark the source as 'web'

            # --- Optional RAG Enhancement ---
            # Check both the request parameter `rag` and the general setting `ragEnabled`
            rag_enabled_setting = current_settings.get('general', {}).get('ragEnabled', False)
            trigger_rag = (rag == 'enabled') # Check if request explicitly asks for RAG

            if trigger_rag and rag_enabled_setting:
                try:
                    logger.info("RAG requested and enabled in settings. Attempting to enhance SearXNG results (placeholder)...")
                    # In the future, replace this with the actual RAG call using the LLM
                    search_results = await utils.enhance_results_with_ai(search_results, q)
                    search_results['rag_enhanced'] = True # Mark enhancement status
                except Exception as e:
                    logger.error(f"AI enhancement (RAG placeholder) failed: {e}", exc_info=True)
                    search_results['rag_enhanced'] = False # Mark enhancement failed
                    search_results['rag_error'] = str(e) # Add error message
            else:
                 # Mark RAG as not performed if not requested or not enabled
                 search_results['rag_enhanced'] = False
                 if trigger_rag and not rag_enabled_setting:
                     logger.info("RAG requested but not enabled in general settings.")
                     search_results['rag_info'] = "RAG requested but not enabled in settings."


            # Return the (potentially enhanced) dictionary from SearXNG
            return search_results

        # =======================================
        # --- Handle Other Configured Sources ---
        # =======================================
        elif actual_source in ['obsidian', 'youtube', 'photos', 'music']: # Add 'localFiles' etc. here when implemented
            source_config_key = actual_source # e.g., 'obsidian', 'youtube'
            # Get the specific configuration block for this source type from settings
            source_settings = current_settings.get('personalSources', {}).get(source_config_key)

            # Check if the source is configured and has a path set
            if not source_settings or not source_settings.get('path'):
                 logger.warning(f"Search requested for '{actual_source}', but its path is not configured.")
                 raise HTTPException(status_code=400, detail=f"Source '{actual_source}' is not configured with a valid path in settings.")

            try:
                # --- Determine Results Per Page for this Source ---
                # Priority: Query param `results` > Source-specific Setting `resultsPerPage` > Fallback default (10)
                results_per_page = results # Use query param `results` if provided
                if results_per_page is None:
                    try:
                         # Use source-specific setting, converting to int
                         results_per_page = int(source_settings.get('resultsPerPage', 10))
                    except (ValueError, TypeError):
                         results_per_page = 10 # Fallback if setting is invalid
                results_per_page = max(1, min(results_per_page, 100)) # Apply bounds

                # Get the base path from settings - DO NOT resolve/validate here, let helpers do it.
                base_path = FilePath(source_settings['path'])

                # --- Call the appropriate search helper function based on source ---
                all_results_list = [] # To hold results before pagination
                errors = []           # To hold non-fatal errors from search helpers
                total_results = 0     # Total number of hits found

                # Dispatch to the correct search utility function
                if actual_source == 'obsidian':
                    # Validate config against model first
                    config_model = models.ObsidianSourceConfig(**source_settings)
                    # Resolve path strictly *before* passing to helper (ensures base exists)
                    resolved_path = base_path.resolve(strict=True)
                    if not resolved_path.is_dir(): raise HTTPException(status_code=400, detail="Configured Obsidian path exists but is not a directory.")
                    # Call search helper
                    all_results_list, errors = await utils.search_obsidian_vault(resolved_path, q)
                    # Paginate results
                    total_results = len(all_results_list)
                    start_index = (pageno - 1) * results_per_page
                    paginated_results = all_results_list[start_index : start_index + results_per_page]
                    # Return structured response
                    return models.ObsidianSearchResponse(query=q, results=paginated_results, errors=errors, total_results=total_results)

                elif actual_source == 'youtube':
                    config_model = models.YouTubeSourceConfig(**source_settings)
                    resolved_path = base_path.resolve(strict=True)
                    if not resolved_path.is_file(): raise HTTPException(status_code=400, detail="Configured YouTube export path exists but is not a file.")
                    all_results_list, errors = await utils.search_tartube_json_export(resolved_path, q, config_model)
                    total_results = len(all_results_list)
                    start_index = (pageno - 1) * results_per_page
                    paginated_results = all_results_list[start_index : start_index + results_per_page]
                    return models.YouTubeSearchResponse(query=q, results=paginated_results, errors=errors, total_results=total_results)

                elif actual_source == 'photos':
                    config_model = models.PhotosSourceConfig(**source_settings)
                    resolved_path = base_path.resolve(strict=True)
                    if not resolved_path.is_dir(): raise HTTPException(status_code=400, detail="Configured Photos path exists but is not a directory.")
                    all_results_list, errors = await utils.search_photo_directory(resolved_path, q)
                    total_results = len(all_results_list)
                    start_index = (pageno - 1) * results_per_page
                    paginated_results = all_results_list[start_index : start_index + results_per_page]
                    return models.PhotosSearchResponse(query=q, results=paginated_results, errors=errors, total_results=total_results)

                elif actual_source == 'music':
                    # Check if Mutagen dependency is met before proceeding
                    if not utils.MUTAGEN_AVAILABLE:
                         logger.error("Music search requested, but Mutagen library is not installed.")
                         raise HTTPException(status_code=501, detail="Music search is unavailable on the server: Required library 'Mutagen' is missing.")
                    config_model = models.MusicSourceConfig(**source_settings)
                    resolved_path = base_path.resolve(strict=True)
                    if not resolved_path.is_dir(): raise HTTPException(status_code=400, detail="Configured Music path exists but is not a directory.")
                    all_results_list, errors = await utils.search_music_directory(resolved_path, q)
                    total_results = len(all_results_list)
                    start_index = (pageno - 1) * results_per_page
                    paginated_results = all_results_list[start_index : start_index + results_per_page]
                    return models.MusicSearchResponse(query=q, results=paginated_results, errors=errors, total_results=total_results)

                # Add elif blocks here for future sources like 'localFiles'

            # --- Error Handling for Local Source Search ---
            except FileNotFoundError:
                # Raised by resolve(strict=True) if the base path doesn't exist
                logger.error(f"Configured path for source '{actual_source}' not found: {source_settings.get('path')}")
                raise HTTPException(status_code=404, detail=f"Configured path for source '{actual_source}' not found.")
            except PermissionError:
                 # Raised by resolve(strict=True) or potentially by search helpers
                logger.error(f"Permission denied accessing path for source '{actual_source}': {source_settings.get('path')}")
                raise HTTPException(status_code=403, detail=f"Permission denied for source '{actual_source}' path.")
            except models.ValidationError as e:
                 # Catch validation errors when creating the specific config model (e.g., YouTubeSourceConfig(**source_settings))
                 logger.error(f"Invalid configuration settings for source '{actual_source}': {e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Invalid server configuration for '{actual_source}' source: {e.errors()}")
            except HTTPException as e:
                # Re-raise HTTP exceptions that might have been raised by helpers (e.g., invalid path type)
                raise e
            except Exception as e:
                # Catch unexpected errors during the search process for this source
                logger.exception(f"Unexpected error during '{actual_source}' search for query '{q}'")
                raise HTTPException(status_code=500, detail=f"Internal server error during {actual_source} search: {e}")

        # =========================
        # --- Invalid Source ID ---
        # =========================
        else:
            # Use the original source string from the request in the error message for clarity
            logger.warning(f"Invalid source ID requested: '{source}'")
            # Define valid sources dynamically or list them
            valid_sources = ['web', 'obsidian', 'youtube', 'photos', 'music'] # Keep updated
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source ID '{source}'. Valid sources are: {', '.join(valid_sources)}."
            )

    # --- General Exception Handling ---
    # Catch any exceptions not handled within the specific source blocks
    except HTTPException as e:
        # Re-raise known HTTP exceptions
        raise e
    except Exception as e:
        # Catch-all for any truly unexpected errors during the request handling setup
        logger.exception(f"Unexpected error handling search request (Query: '{q}', Source: '{source}')")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred processing the search request: {e}")


# --- Settings Endpoints ---

@router.get("/settings", response_model=models.SettingsData, tags=["Settings"])
async def get_settings():
    """Retrieves the current application settings."""
    logger.info("Processing GET request for /settings")
    try:
        # Load settings dictionary using the utility function
        settings_dict = utils.load_settings()
        # Validate the loaded dictionary against the Pydantic model before returning.
        # This ensures the response matches the defined schema, even if the file was slightly off
        # and load_settings returned defaults or corrected structure.
        validated_settings = models.SettingsData(**settings_dict)
        return validated_settings # FastAPI automatically converts Pydantic model to JSON
    except models.ValidationError as e:
         # This might happen if load_settings returns a dict that fails validation (e.g., defaults changed)
         logger.error(f"Loaded settings dictionary failed validation: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal server error: Failed to validate loaded settings.")
    except Exception as e:
        # Catch errors during loading itself
        logger.exception("Failed to load settings for GET /settings")
        raise HTTPException(status_code=500, detail=f"Internal error loading settings: {e}")

@router.post("/settings", status_code=200, tags=["Settings"]) # Use 200 OK for successful update by default
async def update_settings(settings_update: models.SettingsData):
    """
    Updates and saves the application settings.
    Expects the full, valid settings structure in the request body.
    Validates the input against the SettingsData model before attempting to save.
    """
    logger.info("Processing POST request for /settings")
    try:
        # --- Input Validation ---
        # FastAPI automatically validates the incoming request body against the 'settings_update: models.SettingsData' annotation.
        # If validation fails, FastAPI returns a 422 Unprocessable Entity error before this function is even called.
        logger.debug(f"Settings dictionary received and validated by FastAPI against SettingsData model.")

        # --- Prepare for Saving ---
        # Convert the validated Pydantic model back to a dictionary suitable for the save_settings utility.
        # mode='json' handles complex types like HttpUrl correctly.
        # exclude_unset=False ensures default values are included if they weren't provided in the request, maintaining a complete settings file.
        settings_dict_to_save = settings_update.model_dump(mode='json', exclude_unset=False)

        # --- Save Settings ---
        # Call the utility function which handles atomic write and potential re-validation/error handling.
        utils.save_settings(settings_dict_to_save)

        # --- Success Response ---
        # Optionally, load the settings back to confirm and return the *actual* saved state.
        # This accounts for any minor coercions or defaults applied during the save/load cycle.
        confirmed_settings_dict = utils.load_settings()
        # Validate again before sending back? Optional, but safe.
        validated_confirmed_settings = models.SettingsData(**confirmed_settings_dict)

        return {
            "message": "Settings updated successfully",
            "updated_settings": validated_confirmed_settings # Return the validated model
        }

    except HTTPException as e:
        # Re-raise exceptions that might have been raised by save_settings (e.g., 422 validation, 500 file IO)
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the process (e.g., after validation but before/during save)
        logger.exception("Unexpected error during settings update process")
        raise HTTPException(status_code=500, detail=f"Internal error processing settings update: {e}")


# --- Vault Browsing & Checking Endpoints ---

@router.get("/browse/obsidian", response_model=models.VaultBrowseResponse, tags=["Vault"])
async def browse_obsidian_vault(subpath: Optional[str] = Query("", description="Relative subpath within the vault (URL-encoded if necessary). Defaults to the vault root.")):
    """Browses the configured Obsidian vault directory structure."""
    # Note: FastAPI automatically URL-decodes query parameters like 'subpath'.
    logger.info(f"Browsing Obsidian vault at decoded subpath: '{subpath}'")
    settings = utils.load_settings()
    # Retrieve the configured base path for the Obsidian vault
    vault_base_path_str = settings.get('personalSources', {}).get('obsidian', {}).get('path')

    # Check if the vault path is configured in settings
    if not vault_base_path_str:
        logger.warning("Attempted to browse Obsidian vault, but path is not configured in settings.")
        # Return a valid VaultBrowseResponse indicating the error
        return models.VaultBrowseResponse(path=subpath or "", items=[], error="Obsidian vault path not configured in settings.")

    try:
        vault_base_path = FilePath(vault_base_path_str)

        # --- Resolve Target Path Safely ---
        # Use the utility function to resolve the subpath relative to the base path.
        # This handles security checks (traversal), existence checks, and symlink resolution.
        # It will raise HTTPException (403, 404, 500) on errors.
        target_path = utils._resolve_safe_path(vault_base_path, subpath or "") # Pass empty string for root

        # --- Check if Target is a Directory ---
        # _resolve_safe_path ensures target_path exists, now check if it's a directory we can list.
        if not target_path.is_dir():
             logger.warning(f"Requested browse path exists but is not a directory: {target_path}")
             # Return a valid response indicating the path isn't browseable
             # Calculate relative path for the response even if it's not a dir
             current_relative_path = target_path.relative_to(vault_base_path).as_posix() if target_path != vault_base_path else ""
             return models.VaultBrowseResponse(path=current_relative_path, items=[], error="Requested path is not a browseable directory.")

        # --- List Directory Contents ---
        items = []
        # Iterate through items directly within the resolved target directory
        for item_path in target_path.iterdir():
            try:
                # Use lstat to get info without following symlinks for the item itself
                item_stat = item_path.lstat()
                item_type: Optional[Literal['file', 'directory']] = None # Type hint

                # Determine item type (directory or file)
                if stat.S_ISDIR(item_stat.st_mode):
                    item_type = 'directory'
                elif stat.S_ISREG(item_stat.st_mode):
                    item_type = 'file'
                else:
                    # Skip other types like sockets, pipes, broken symlinks etc.
                    logger.debug(f"Skipping item '{item_path.name}' with unsupported type in vault browse.")
                    continue # Skip to the next item

                # Calculate relative path from the *vault base*, not the current target_path
                relative_path_to_base = item_path.relative_to(vault_base_path)
                relative_path_str = relative_path_to_base.as_posix() # Use forward slashes

                # Create the VaultItem model
                items.append(models.VaultItem(
                    name=item_path.name,
                    path=relative_path_str, # Path relative to vault root
                    type=item_type,
                    modified_time=datetime.datetime.fromtimestamp(item_stat.st_mtime, tz=datetime.timezone.utc),
                    size=item_stat.st_size if item_type == 'file' else None,
                    extension=item_path.suffix.lower() if item_type == 'file' else None
                ))
            except (FileNotFoundError, PermissionError) as item_err:
                 # Handle cases where item disappears between iterdir and lstat, or permission issues for the item
                 logger.warning(f"Skipping vault item '{item_path.name}' due to error: {item_err}")
                 continue # Skip this specific item, continue browsing others
            except Exception as e:
                 # Log unexpected errors during processing of a single item but continue browsing
                 logger.warning(f"Error processing vault item '{item_path.name}': {e}", exc_info=True)
                 continue # Skip this item

        # --- Sort Items ---
        # Sort items: Directories first, then files, alphabetically within each group
        items.sort(key=lambda x: (x.type != 'directory', x.name.lower()))

        # --- Return Success Response ---
        # Calculate the relative path of the directory *being browsed* for the response
        current_relative_path = target_path.relative_to(vault_base_path).as_posix() if target_path != vault_base_path else ""
        return models.VaultBrowseResponse(path=current_relative_path, items=items)

    except HTTPException as e:
        # Re-raise HTTP exceptions from _resolve_safe_path or other checks
        raise e
    except Exception as e:
        # Catch unexpected errors during the overall browse process (e.g., iterdir failure)
        logger.exception(f"Error browsing Obsidian vault subpath '{subpath}'")
        raise HTTPException(status_code=500, detail=f"Internal server error while browsing vault: {e}")


@router.get("/check/obsidian", response_model=models.VaultCheckResponse, tags=["Vault"])
async def check_obsidian_vault_path(path: str = Query(..., description="URL-encoded absolute path to check for Obsidian vault validity.")):
    """
    Validates if a given absolute path points to a directory containing an
    '.obsidian' subdirectory, indicating a likely Obsidian vault.
    """
    try:
        # URL-decode the path parameter provided in the query string
        decoded_path_str = urllib.parse.unquote_plus(path)
        path_obj = FilePath(decoded_path_str)
        logger.info(f"Checking Obsidian path validity for: '{path_obj}'")

        # --- Basic Path Validation ---
        if not path_obj.is_absolute():
            # Enforce absolute paths for clarity and security. Relative paths depend on CWD.
            logger.warning(f"Path check failed: Path '{path_obj}' is not absolute.")
            # Return 400 Bad Request for invalid input format
            raise HTTPException(status_code=400, detail="Path must be absolute.")

        # --- Existence and Type Check ---
        # Resolve the path strictly to ensure it exists and follow symlinks.
        # This will raise FileNotFoundError if the path doesn't exist.
        resolved_path = path_obj.resolve(strict=True)

        # Check if the resolved path points to a directory.
        if not resolved_path.is_dir():
             logger.info(f"Path check failed: '{resolved_path}' exists but is not a directory.")
             # Path exists, but isn't the required type. Not a "Not Found" error.
             return models.VaultCheckResponse(valid=False, error="Path exists but is not a directory.")

        # --- Check for '.obsidian' Marker Directory ---
        # Construct the path to the expected '.obsidian' subdirectory.
        obsidian_marker_path = resolved_path / ".obsidian"

        # Check if the '.obsidian' path exists *and* is a directory.
        if not obsidian_marker_path.is_dir():
            error_msg = "Required '.obsidian' subdirectory not found within the path."
            # Check if it exists but isn't a directory (e.g., it's a file)
            if obsidian_marker_path.exists():
                error_msg = "Path contains an item named '.obsidian', but it is not a directory."
            logger.info(f"Path check failed: {error_msg} for path '{resolved_path}'")
            return models.VaultCheckResponse(valid=False, error=error_msg)

        # --- Success ---
        # If all checks pass, the path is considered a valid Obsidian vault directory.
        logger.info(f"Path check successful: '{resolved_path}' appears to be a valid Obsidian vault.")
        return models.VaultCheckResponse(valid=True)

    # --- Error Handling ---
    except FileNotFoundError:
        # Raised by resolve(strict=True) if the input path doesn't exist.
        logger.info(f"Path check failed: Path '{decoded_path_str}' does not exist.")
        # Return 404 Not Found.
        raise HTTPException(status_code=404, detail="Specified path does not exist.")
    except PermissionError:
        # Raised if OS denies access during path resolution or checks.
        logger.warning(f"Path check failed: Permission denied accessing '{decoded_path_str}'.")
        # Return 403 Forbidden.
        raise HTTPException(status_code=403, detail="Permission denied accessing the specified path.")
    except ValueError as e:
        # Catch errors from FilePath creation (e.g., invalid characters) or potentially resolve().
        logger.warning(f"Path check failed: Invalid path format or characters in '{path}': {e}")
        # Return 400 Bad Request.
        raise HTTPException(status_code=400, detail="Invalid path format or characters.")
    except HTTPException as e:
        # Re-raise specific HTTP exceptions we might have raised (like the 400 for non-absolute path).
        raise e
    except Exception as e:
        # Catch any other unexpected OS or internal errors during the checks.
        logger.exception(f"Unexpected error checking Obsidian path '{path}': {e}")
        # Return 500 Internal Server Error.
        raise HTTPException(status_code=500, detail="Internal server error during path check.")


# --- File Serving Endpoints ---

# Note: These endpoints have '/api/' prefix hardcoded here. Consider if this should be part of the router prefix.
# They are tagged for documentation.

@router.get("/api/tartube_thumbnails/{channel}/{filename}", tags=["File Serving"])
async def get_tartube_thumbnail(
    channel: str = Path(..., description="URL-decoded channel name subdirectory."),
    filename: str = Path(..., description="URL-decoded thumbnail filename.")
):
    """
    Serves thumbnail images stored locally from the configured Tartube download directory.
    Assumes a structure like '[download_base_path]/[channel]/[filename]'.
    """
    # Path parameters (channel, filename) are automatically URL-decoded by FastAPI.
    logger.info(f"Requesting Tartube thumbnail: Decoded Channel='{channel}', Decoded Filename='{filename}'")
    settings = utils.load_settings()
    # Get the configured base path for Tartube downloads from settings.
    base_path_str = settings.get('personalSources', {}).get('youtube', {}).get('download_base_path')

    # Check if the base path is configured.
    if not base_path_str:
        logger.error("Cannot serve Tartube thumbnail: YouTube download_base_path not configured.")
        raise HTTPException(status_code=404, detail="YouTube thumbnail base path not configured in server settings.")

    try:
        base_path = FilePath(base_path_str)
        # Construct the relative subpath from the decoded channel and filename.
        # Use os.path.join for cross-platform compatibility. Channel/filename might still contain '/' if encoded that way.
        subpath = os.path.join(channel, filename)
        logger.debug(f"Attempting to resolve safe path for thumbnail: Base='{base_path}', Subpath='{subpath}'")

        # --- Resolve Path Safely ---
        # Use the utility function to get the final, validated, absolute path.
        # Raises HTTPException (403, 404, 500) on errors.
        target_file_path = utils._resolve_safe_path(base_path, subpath)

        # --- Check if it's a File ---
        # _resolve_safe_path ensures it exists, but we need to ensure it's a file, not a directory.
        if not target_file_path.is_file():
             logger.error(f"Resolved thumbnail path is not a file: {target_file_path}")
             # Treat as Not Found if the resolved path isn't a file.
             raise HTTPException(status_code=404, detail="Thumbnail resource is not a file.")

        # --- Determine Media Type ---
        # Guess media type based on file extension for the Content-Type header.
        media_type_map = {
             '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
             '.gif': 'image/gif', '.webp': 'image/webp'
             # Add other expected thumbnail types if necessary
        }
        media_type = media_type_map.get(target_file_path.suffix.lower()) # Case-insensitive suffix check

        logger.debug(f"Serving thumbnail file: {target_file_path}, Media Type: {media_type or 'application/octet-stream'}")

        # --- Serve the File ---
        # Use FastAPI's FileResponse for efficient streaming.
        return FileResponse(
            path=str(target_file_path), # Path must be string
            media_type=media_type, # Set Content-Type header, browser might infer if None
            filename=target_file_path.name # Suggest filename for download (optional)
        )

    except HTTPException as e:
        # Re-raise exceptions from _resolve_safe_path or the is_file check.
        raise e
    except Exception as e:
        # Catch unexpected errors during file serving preparation.
        logger.exception(f"Error serving thumbnail '{channel}/{filename}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error serving thumbnail.")


@router.get("/api/photos/{relative_path:path}", tags=["File Serving"])
async def get_photo(
    relative_path: str = Path(..., description="URL-encoded relative path of the photo within the configured photos directory.")
):
    """
    Serves photo files stored locally from the configured photos directory.
    The 'relative_path' parameter captures the entire path after '/api/photos/'.
    """
    # The 'relative_path' is automatically URL-decoded by FastAPI.
    logger.info(f"Requesting photo: Decoded Relative Path='{relative_path}'")
    settings = utils.load_settings()
    # Get the configured base path for photos.
    base_path_str = settings.get('personalSources', {}).get('photos', {}).get('path')

    # Check if the base path is configured.
    if not base_path_str:
        logger.error("Cannot serve photo: Photos base path not configured.")
        raise HTTPException(status_code=404, detail="Photo base path not configured in server settings.")

    try:
        base_path = FilePath(base_path_str)
        logger.debug(f"Attempting to resolve safe photo path: Base='{base_path}', Relative Path='{relative_path}'")

        # --- Resolve Path Safely ---
        # Use the utility function with the captured relative path.
        # Raises HTTPException (403, 404, 500) on errors.
        target_file_path = utils._resolve_safe_path(base_path, relative_path)

        # --- Check if it's a File ---
        if not target_file_path.is_file():
             logger.error(f"Resolved photo path is not a file: {target_file_path}")
             raise HTTPException(status_code=404, detail="Photo resource is not a file.")

        # --- Determine Media Type ---
        # Guess media type based on file extension.
        media_type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff'
            # Add more supported image types as needed
        }
        media_type = media_type_map.get(target_file_path.suffix.lower())

        logger.debug(f"Serving photo file: {target_file_path}, Media Type: {media_type or 'application/octet-stream'}")

        # --- Serve the File ---
        return FileResponse(
            path=str(target_file_path),
            media_type=media_type,
            filename=target_file_path.name # Suggest original filename
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error serving photo '{relative_path}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error serving photo.")


# --- SearXNG Engine Catalog Endpoint ---

# Note: The response model `List[models.EngineDetail]` uses the model defined in main_models.py
@router.get("/api/searxng/engines", response_model=List[models.EngineDetail], tags=["SearXNG Integration"])
async def get_searxng_engines_endpoint():
    """
    Retrieves the list of all known SearXNG engines as defined in the
    server's engine catalog file (e.g., engine_catalog.json).
    This represents the potential engines available, regardless of their
    enabled status in any specific loadout. Used for populating engine selection UIs.
    """
    logger.debug("Request received for /api/searxng/engines")
    # Use the utility function to load and validate engines from the catalog file.
    # This function handles file reading, JSON parsing, Pydantic validation, and error logging internally.
    available_engines: List[models.EngineDetail] = await utils.get_searxng_engines()

    # The response model List[models.EngineDetail] ensures the output matches the expected structure.
    # FastAPI will automatically convert the list of Pydantic models to JSON.
    # If get_searxng_engines encountered errors and returned [], this will correctly return an empty JSON array.
    return available_engines
