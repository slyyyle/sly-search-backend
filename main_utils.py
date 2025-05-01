# main_utils.py
import httpx
import json
import logging
import os
import re
import stat
import urllib.parse
import datetime
from pathlib import Path as FilePath
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import HTTPException
from urllib.parse import quote as url_quote

# Import models and config needed by utils
import main_models as models
import main_config as config

# External libraries
try:
    import mutagen
    import mutagen.mp3
    import mutagen.flac
    import mutagen.oggvorbis
    import mutagen.mp4
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning("Mutagen library not found. Music metadata searching will be disabled.")


logger = logging.getLogger(__name__)

# --- Settings Load/Save ---

def load_settings() -> Dict[str, Any]:
    """Loads settings from SETTINGS_FILE or returns defaults defined in SettingsData."""
    defaults = models.SettingsData().model_dump()

    if not os.path.exists(config.SETTINGS_FILE):
        logger.info(f"Settings file '{config.SETTINGS_FILE}' not found. Returning default settings.")
        return defaults
    try:
        with open(config.SETTINGS_FILE, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Ensure 'engines' structure exists and has defaults if missing
        if 'engines' not in loaded_data or not isinstance(loaded_data.get('engines'), dict):
            logger.warning("Loaded settings file missing 'engines' structure. Initializing.")
            loaded_data['engines'] = models.EngineSettings().model_dump()
        else:
            if 'loadouts' not in loaded_data['engines'] or not isinstance(loaded_data['engines'].get('loadouts'), list):
                logger.warning("Loaded settings file missing 'loadouts' list in 'engines'. Initializing with model default.")
                default_engines_model = models.EngineSettings()
                # Ensure loadouts are dictionaries when setting default
                loaded_data['engines']['loadouts'] = [lo.model_dump() for lo in default_engines_model.loadouts]
            if 'activeLoadoutId' not in loaded_data['engines']:
                 loaded_data['engines']['activeLoadoutId'] = 'starter'
                 logger.warning("Loaded settings file missing 'activeLoadoutId' in 'engines'. Initializing with 'starter'.")

        # --- Validate final structure using Pydantic ---
        validated_settings = models.SettingsData(**loaded_data)
        logger.info(f"Successfully loaded and validated settings from {config.SETTINGS_FILE}")
        # Return as dict, ensuring defaults for unset fields are included if needed
        return validated_settings.model_dump(exclude_unset=False) # Use exclude_unset=False to include defaults

    except (json.JSONDecodeError, IOError, models.ValidationError, Exception) as e:
        logger.error(f"Error loading or validating settings file '{config.SETTINGS_FILE}': {e}. Returning defaults.", exc_info=True)
        # Return defaults on any error during loading/validation
        return models.SettingsData().model_dump(exclude_unset=False) # Ensure full default structure

def save_settings(settings_to_save: Dict[str, Any]):
    """Saves the provided settings dictionary to SETTINGS_FILE atomically after validation."""
    temp_file_path = f"{config.SETTINGS_FILE}.tmp"
    try:
        # Ensure engines structure consistency before validation
        if 'engines' not in settings_to_save or not isinstance(settings_to_save.get('engines'), dict):
             settings_to_save['engines'] = models.EngineSettings().model_dump()
             logger.warning("Data passed to save_settings missing 'engines'. Initializing.")
        elif 'activeLoadoutId' not in settings_to_save['engines']:
             loadouts = settings_to_save['engines'].get('loadouts', [])
             # Ensure loadouts are dicts if they came from Pydantic models
             loadouts_as_dicts = [lo.model_dump(exclude_unset=True) if isinstance(lo, models.EngineLoadoutItem) else lo for lo in loadouts]
             settings_to_save['engines'] = models.EngineSettings(loadouts=loadouts_as_dicts, activeLoadoutId='starter').model_dump()
             logger.warning("Data passed to save_settings missing 'activeLoadoutId'. Adding 'starter'.")

        # Validate the entire structure - this raises ValidationError on failure
        validated_data = models.SettingsData(**settings_to_save)

        # Write validated data to temp file using Pydantic's JSON export
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            # Use model_dump_json for direct, validated JSON output
            f.write(validated_data.model_dump_json(indent=4))

        # Atomically replace the original file with the temp file
        os.rename(temp_file_path, config.SETTINGS_FILE)
        logger.info(f"Settings successfully validated and atomically saved to '{config.SETTINGS_FILE}'.")

    except (IOError, TypeError, models.ValidationError, Exception) as e:
        logger.error(f"Error validating or saving settings to '{config.SETTINGS_FILE}' via temporary file '{temp_file_path}': {e}", exc_info=True)
        # Clean up temp file if it exists after an error
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary settings file: {temp_file_path}")
            except OSError as remove_err:
                logger.error(f"Error removing temporary settings file {temp_file_path}: {remove_err}")
        # Re-raise specific exceptions for the API layer (main_routes.py) to handle
        if isinstance(e, models.ValidationError):
            # Provide detailed validation errors
            raise HTTPException(status_code=422, detail=f"Settings validation failed: {e.errors()}")
        else:
            # General server error for other issues
            raise HTTPException(status_code=500, detail=f"Failed to save settings due to {type(e).__name__}: {e}")
    except Exception as e: # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error during atomic settings save: {e}", exc_info=True)
        # Ensure cleanup even for truly unexpected errors
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError: pass # Ignore errors during cleanup here
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred saving settings: {e}")


# --- Path Resolution Helper ---

def _resolve_safe_path(base_path: FilePath, requested_subpath: str) -> FilePath:
    """
    Resolves the requested subpath relative to the base path,
    ensuring it doesn't escape the base path (prevents path traversal).
    Raises HTTPException on errors (403, 404, 500) or security violations.
    """
    if not isinstance(base_path, FilePath):
        logger.error(f"Invalid base_path type provided to _resolve_safe_path: {type(base_path)}")
        raise HTTPException(status_code=500, detail="Internal server error: Invalid base path type.")

    # --- Validate Base Path ---
    try:
        if not base_path.is_absolute():
             # If base path isn't absolute, resolve it relative to CWD (or handle differently if needed)
             # This might have security implications if CWD is unexpected. Forcing absolute seems safer.
             # For now, let's assume configured base paths should be absolute or resolvable.
             # base_path = base_path.resolve(strict=True) # Or raise error if relative base paths aren't allowed
             logger.warning(f"Base path '{base_path}' is not absolute. Resolving.")
             # Strict=True ensures the base path itself must exist.
             resolved_base = base_path.resolve(strict=True)
        else:
            # Resolve even absolute paths to handle symlinks and canonicalize '..' etc.
            resolved_base = base_path.resolve(strict=True)

        if not resolved_base.exists(): # Should be caught by strict=True, but double-check
            logger.error(f"Base path does not exist after resolving: {resolved_base}")
            raise FileNotFoundError # Trigger the FileNotFoundError handling below

        if not resolved_base.is_dir():
            logger.error(f"Resolved base path is not a directory: {resolved_base}")
            raise NotADirectoryError # Trigger specific handling or let it fall to generic Exception

    except FileNotFoundError:
         logger.error(f"Base path configuration error: Path not found at '{base_path}'")
         raise HTTPException(status_code=500, detail="Internal server error: Configured base path not found.")
    except PermissionError:
         logger.error(f"Base path configuration error: Permission denied for '{base_path}'")
         raise HTTPException(status_code=500, detail="Internal server error: Permission denied for configured base path.")
    except Exception as e: # Catch other resolution errors (NotADirectoryError, etc.)
         logger.exception(f"Error resolving base path '{base_path}': {e}")
         raise HTTPException(status_code=500, detail=f"Internal server error resolving base path: {type(e).__name__}")


    # --- Normalize and Validate Subpath ---
    # Strip leading/trailing slashes/backslashes for consistency
    # os.path.normpath handles '.' and redundant separators (e.g., '//')
    norm_subpath = os.path.normpath(requested_subpath.strip('/\\'))

    # If normalized subpath is empty or '.', return the resolved base path safely
    if not norm_subpath or norm_subpath == '.':
        return resolved_base

    # Security Check: Prevent directory traversal using '..' components
    if '..' in norm_subpath.split(os.path.sep):
        logger.warning(f"Path traversal attempt detected in subpath: '{requested_subpath}' (normalized: '{norm_subpath}') relative to base '{resolved_base}'")
        raise HTTPException(status_code=403, detail="Access forbidden: Invalid path components ('..').")

    # --- Combine and Resolve Final Path ---
    try:
        # Combine the resolved base path and normalized subpath
        combined_path = resolved_base / norm_subpath

        # Resolve the combined path to handle symlinks and get the absolute canonical path
        # strict=False initially: allows checking the path relationship even if the final part doesn't exist yet
        final_resolved_path = combined_path.resolve(strict=False)

        # --- Final Security Check: is_relative_to ---
        # Ensure the fully resolved final path is *still* within the resolved base path directory tree.
        # This is crucial for preventing escape via symlinks or complex '..' sequences not caught earlier.
        if not final_resolved_path.is_relative_to(resolved_base):
            logger.warning(f"Path traversal detected or path outside base after resolving symlinks: Base='{resolved_base}', Requested='{requested_subpath}', Resolved='{final_resolved_path}'")
            raise HTTPException(status_code=403, detail="Access forbidden: Path resolves outside the allowed base directory.")

        # --- Existence Check (strict=True) ---
        # Now, resolve strictly again to ensure the *final target path* actually exists.
        # This raises FileNotFoundError if the final file/directory is missing.
        final_existing_path = final_resolved_path.resolve(strict=True)

        return final_existing_path

    except FileNotFoundError:
        # The final combined path does not point to an existing file or directory
        logger.info(f"Resource not found at resolved path: '{final_resolved_path}' (from request '{requested_subpath}')")
        raise HTTPException(status_code=404, detail="Requested resource path not found.")
    except PermissionError:
        # Permission denied accessing the final combined path
        logger.warning(f"Permission denied accessing resolved path: '{final_resolved_path}' (from request '{requested_subpath}')")
        raise HTTPException(status_code=403, detail="Permission denied accessing resource path.")
    except ValueError as e: # Catch potential errors from path operations like joining invalid chars
        logger.error(f"Path processing error for subpath '{requested_subpath}' relative to '{base_path}': {e}")
        raise HTTPException(status_code=400, detail="Invalid path format or component.")
    except HTTPException as e: # Re-raise our specific HTTP exceptions
        raise e
    except Exception as e: # Catch any other unexpected errors during resolution
        logger.exception(f"Unexpected error resolving path: Base='{base_path}', Subpath='{requested_subpath}'")
        raise HTTPException(status_code=500, detail=f"Internal server error resolving path: {type(e).__name__}")


# --- Search Helpers ---

async def search_obsidian_vault(vault_path: FilePath, query: str) -> Tuple[List[models.ObsidianResultItem], List[str]]:
    """Searches *.md files in the vault for the query, generating snippets."""
    results = []
    errors = []
    query_lower = query.lower()
    max_snippet_length = 250
    yaml_separator = "---"
    logger.info(f"Starting Obsidian search in '{vault_path}' for query '{query}'")

    if not vault_path.is_dir(): # Check should be redundant if called after _resolve_safe_path
        errors.append(f"Configured Obsidian path is not a valid directory: {vault_path}")
        logger.error(f"Obsidian path error: {errors[-1]}")
        return [], errors

    try:
        # Use rglob for recursive search of Markdown files
        for md_file_path in vault_path.rglob('*.md'):
            # Skip files within the .obsidian config directory and its subdirectories
            if '.obsidian' in md_file_path.parts:
                continue
            # Ensure it's actually a file
            if not md_file_path.is_file():
                continue

            relative_path_str = str(md_file_path.relative_to(vault_path)).replace('\\', '/') # Use posix paths for URLs/IDs
            query_found_in_file = False
            content_for_snippet = ""
            first_content_line_num = -1 # Track where content starts
            is_in_yaml = False
            yaml_block_ended = False

            try:
                with md_file_path.open('r', encoding='utf-8', errors='ignore') as f:
                    # Using enumerate allows tracking line numbers easily
                    for line_num, line in enumerate(f, 1):
                        stripped_line = line.strip()

                        # --- YAML Front Matter Handling ---
                        if line_num == 1 and stripped_line == yaml_separator:
                            is_in_yaml = True
                            continue # Skip the first separator line
                        if is_in_yaml:
                            if stripped_line == yaml_separator:
                                is_in_yaml = False
                                yaml_block_ended = True
                            continue # Skip all content within YAML block

                        # --- Content Processing (after YAML or if no YAML) ---
                        # This block executes only for lines outside the YAML front matter
                        if first_content_line_num == -1:
                            first_content_line_num = line_num # Mark the first line of actual content

                        # Build snippet from initial content lines (up to max length)
                        if len(content_for_snippet) < max_snippet_length:
                            # Add space between lines unless snippet is empty
                            content_for_snippet += (" " if content_for_snippet else "") + stripped_line
                            # Truncate immediately if max length reached/exceeded
                            if len(content_for_snippet) >= max_snippet_length:
                                content_for_snippet = content_for_snippet[:max_snippet_length].rstrip() + "..."
                                # Stop adding to snippet once truncated

                        # --- Query Matching (check all lines outside YAML) ---
                        # We perform the query check independently of snippet generation
                        # This ensures a match is recorded even if it occurs after the snippet is filled.
                        if not query_found_in_file: # Stop checking once found
                            if query_lower in line.lower(): # Case-insensitive check
                                query_found_in_file = True

                # --- Result Creation (if query was found anywhere in the content) ---
                if query_found_in_file:
                    # Finalize snippet: Ensure it's stripped and handle edge cases
                    final_snippet = content_for_snippet.strip()

                    # Handle cases where snippet remained empty (e.g., file only had YAML or was empty)
                    if not final_snippet:
                        if first_content_line_num > 1 or yaml_block_ended:
                             final_snippet = "[no content found after YAML]"
                        else: # File was likely empty or contained only the first YAML separator
                             final_snippet = "[empty file or only YAML]"
                    # Ensure snippet doesn't exceed max length if truncation happened exactly at the limit without ellipsis
                    elif not final_snippet.endswith("...") and len(final_snippet) > max_snippet_length:
                        final_snippet = final_snippet[:max_snippet_length].rstrip() + "..."

                    try:
                        # Use file stem (name without extension) as title
                        title = md_file_path.stem
                        # Create the Pydantic model for the result
                        results.append(models.ObsidianResultItem(
                            title=title,
                            url=relative_path_str, # Use the relative path as the URL/identifier
                            snippet=final_snippet
                        ))
                    except Exception as e: # Catch errors during Pydantic model creation if any
                        error_msg = f"Error creating result item for {relative_path_str}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            except PermissionError:
                error_msg = f"Permission denied reading file: {relative_path_str}"
                logger.warning(error_msg)
                errors.append(error_msg)
            except OSError as e: # Catch file system errors (e.g., file deleted during read)
                error_msg = f"OS error reading file {relative_path_str}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
            except Exception as e: # Catch unexpected errors during file processing
                error_msg = f"Unexpected error processing file {relative_path_str}: {e}"
                logger.exception(error_msg) # Log full traceback for unexpected errors
                errors.append(error_msg)

    except PermissionError:
        error_msg = f"Permission denied listing files in vault path: {vault_path}"
        logger.error(error_msg)
        errors.append(error_msg)
    except Exception as e: # Catch errors during rglob iteration itself
        error_msg = f"Unexpected error walking vault directory {vault_path}: {e}"
        logger.exception(error_msg)
        errors.append(error_msg)

    logger.info(f"Obsidian search completed. Found {len(results)} results. Encountered {len(errors)} errors.")
    # Sort results alphabetically by title for consistent ordering
    results.sort(key=lambda x: x.title.lower())
    return results, errors

async def search_tartube_json_export(json_export_path: FilePath, query: str, youtube_settings: models.YouTubeSourceConfig) -> Tuple[List[models.YouTubeResultItem], List[str]]:
    """Searches a Tartube JSON export file for videos matching the query."""
    results = []
    errors = []
    query_lower = query.lower()
    download_base_path = None
    download_base_path_str = youtube_settings.download_base_path

    if not json_export_path.is_file(): # Basic check before proceeding
        errors.append(f"Tartube JSON export file not found or is not a file: {json_export_path}")
        logger.error(errors[-1])
        return [], errors

    # --- Resolve and validate the download base path for thumbnails ---
    if download_base_path_str:
        try:
            # Resolve strictly to ensure it exists and get absolute path
            resolved_path = FilePath(download_base_path_str).resolve(strict=True)
            if resolved_path.is_dir():
                download_base_path = resolved_path
                logger.info(f"Using YouTube download base path for thumbnails: {download_base_path}")
            else:
                warn_msg = f"Configured YouTube download_base_path exists but is not a directory: {resolved_path}. Thumbnails disabled."
                logger.warning(warn_msg)
                errors.append(f"Warning: YouTube download path is not a directory: {resolved_path}")
        except FileNotFoundError:
            warn_msg = f"Configured YouTube download_base_path not found: '{download_base_path_str}'. Thumbnails disabled."
            logger.warning(warn_msg)
            errors.append(f"Warning: YouTube download path not found: {download_base_path_str}")
        except PermissionError:
            warn_msg = f"Permission denied accessing YouTube download_base_path '{download_base_path_str}'. Thumbnails disabled."
            logger.warning(warn_msg)
            errors.append(f"Warning: Permission denied for YouTube download path: {download_base_path_str}")
        except Exception as e: # Catch other potential errors during path resolution
            warn_msg = f"Error resolving YouTube download_base_path '{download_base_path_str}': {e}. Thumbnails disabled."
            logger.warning(warn_msg)
            errors.append(f"Warning: Error accessing YouTube download path: {e}")
    else:
        logger.info("YouTube download_base_path not configured. Thumbnails from local files disabled.")

    logger.info(f"Starting Tartube JSON export search in '{json_export_path}' for query '{query}'. Local Thumbnails: {'Enabled' if download_base_path else 'Disabled'}")

    try:
        # --- Load and Parse JSON ---
        with open(json_export_path, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)

        # --- Navigate Tartube JSON structure ---
        # Tartube structure might vary (e.g., root dict, or nested under 'db_dict')
        data_to_iterate = None
        if isinstance(loaded_json, dict):
            if 'db_dict' in loaded_json and isinstance(loaded_json['db_dict'], dict):
                logger.debug("Found top-level 'db_dict'. Iterating inside it.")
                data_to_iterate = loaded_json['db_dict']
            else:
                 # Check if the root dictionary seems to contain channels/videos directly
                 # Heuristic: Check if values look like channel or video dicts
                 is_likely_data_dict = False
                 if loaded_json:
                     first_value = next(iter(loaded_json.values()), None)
                     if isinstance(first_value, dict) and first_value.get("type") in ["channel", "video"]:
                         is_likely_data_dict = True

                 if is_likely_data_dict:
                      logger.debug("No top-level 'db_dict', but root seems to contain data. Iterating over root keys.")
                      data_to_iterate = loaded_json
                 else:
                     # If structure is unrecognized, report error
                     err_msg = "Could not find recognizable Tartube data structure ('db_dict' or direct channel/video entries) in JSON."
                     logger.error(err_msg)
                     errors.append(err_msg)
                     return [], errors # Stop processing if structure is wrong
        else:
             # If JSON root is not a dictionary
             err_msg = f"Invalid JSON structure: Expected a top-level dictionary, but got {type(loaded_json).__name__}."
             logger.error(err_msg)
             errors.append(err_msg)
             return [], errors # Stop processing

        # --- Iterate through items (channels or videos) ---
        for item_id, item_data in data_to_iterate.items():
            if isinstance(item_data, dict):
                item_type = item_data.get("type")

                # Process Channels: Iterate through their nested videos
                if item_type == "channel":
                    channel_name = item_data.get("name", f"Unnamed Channel {item_id}")
                    logger.debug(f"Processing channel: {channel_name} (ID: {item_id})")
                    videos_dict = item_data.get("db_dict") # Videos are expected here
                    if isinstance(videos_dict, dict):
                        for video_id, video_data in videos_dict.items():
                             # Ensure nested item is a video dictionary
                             if isinstance(video_data, dict) and video_data.get("type") == "video":
                                 hit, result_item, error = _process_tartube_video_item(
                                     video_data, query_lower, channel_name, download_base_path
                                 )
                                 if error: errors.append(error) # Collect errors from processing
                                 if hit and result_item: results.append(result_item) # Collect valid results
                             else:
                                 logger.debug(f"Skipping non-video item inside channel '{channel_name}' db_dict (ID: {video_id}, Type: {type(video_data).__name__})")
                    else:
                        # Log if a channel doesn't contain the expected video structure
                        logger.debug(f"Channel '{channel_name}' has no 'db_dict' or it's not a dictionary. Skipping nested videos.")

                # Process top-level Videos (if structure allows)
                elif item_type == "video":
                    logger.debug(f"Processing top-level video (ID: {item_id})")
                    # Try to get channel name if available directly on video item (might not be standard)
                    toplevel_channel_name = item_data.get("channel_name")
                    hit, result_item, error = _process_tartube_video_item(
                        item_data, query_lower, toplevel_channel_name, download_base_path
                    )
                    if error: errors.append(error)
                    if hit and result_item: results.append(result_item)

                # Skip other item types if necessary
                # else:
                #    logger.debug(f"Skipping item with type '{item_type}' (ID: {item_id})")

            else:
                # Log if we encounter non-dictionary items at the main data level
                logger.debug(f"Skipping non-dictionary item at data level (ID: {item_id}, Type: {type(item_data).__name__})")


        logger.info(f"Tartube JSON export search completed. Found {len(results)} results. Encountered {len(errors)} non-fatal processing errors.")
        # Sort results alphabetically by title for consistency
        results.sort(key=lambda x: x.title.lower())
        return results, errors # Return results and any accumulated errors

    except PermissionError:
        err_msg = f"Permission denied reading Tartube JSON export file: {json_export_path}"
        logger.error(err_msg)
        errors.append(err_msg)
        return [], errors # Return empty results and the error
    except json.JSONDecodeError as e:
        err_msg = f"Error decoding JSON from file '{json_export_path}': {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return [], errors # Return empty results and the error
    except Exception as e: # Catch any other unexpected errors during file processing/iteration
        logger.exception(f"Unexpected error during Tartube JSON export search for file {json_export_path}")
        errors.append(f"Unexpected error processing Tartube file: {e}")
        return [], errors # Return empty results and the error

def _process_tartube_video_item(
    video_data: Dict[str, Any],
    query_lower: str,
    channel_name: Optional[str],
    download_base_path: Optional[FilePath]
) -> Tuple[bool, Optional[models.YouTubeResultItem], Optional[str]]:
    """
    Helper to process a single video dictionary from Tartube export.
    Checks for query match and constructs a YouTubeResultItem.
    Returns: (match_found: bool, result_item: YouTubeResultItem | None, error_message: str | None)
    """
    video_name_for_log = video_data.get("name", "[MISSING NAME]")
    video_id_for_log = video_data.get("vid", "[MISSING VID]")
    logger.debug(f"Processing video item: '{video_name_for_log}' (VID: {video_id_for_log}) for query '{query_lower}'")

    error_message = None
    search_hit = False
    result_item = None

    try:
        # --- Extract fields from video data ---
        video_name = video_data.get("name")
        video_url = video_data.get("source") # Usually the original YouTube URL
        video_desc = video_data.get("description")
        video_vid = video_data.get("vid") # YouTube video ID
        video_id_internal = video_data.get("db_id") # Tartube's internal ID (optional)

        # --- Basic Validation ---
        # Check essential fields needed for a result
        if not video_name: return False, None, f"Skipped video ID {video_id_internal or video_vid or 'Unknown'}: Missing 'name'"
        if not video_url: return False, None, f"Skipped video '{video_name}': Missing 'source' (URL)"
        if not video_vid: return False, None, f"Skipped video '{video_name}': Missing 'vid' (YouTube ID)"
        # Channel name is helpful but not strictly essential
        if not channel_name: logger.warning(f"Video '{video_name}' (VID: {video_vid}) has no associated channel name.")

        # --- Query Matching ---
        # Check if query exists in title, channel name, or description (case-insensitive)
        if query_lower in video_name.lower(): search_hit = True
        if not search_hit and channel_name and query_lower in channel_name.lower(): search_hit = True
        if not search_hit and video_desc and query_lower in video_desc.lower(): search_hit = True

        # If no match found in relevant fields, return early (no result, no error)
        if not search_hit:
            return False, None, None

        # --- Thumbnail Logic (only if query matched) ---
        thumbnail_url_to_send = None
        if download_base_path and channel_name and video_name:
            # Sanitize channel and video names to create safe file/directory names
            # Replace characters potentially problematic in file systems with underscores
            safe_channel = re.sub(r'[<>:"/\\|?*]', '_', channel_name)
            safe_filename_base = re.sub(r'[<>:"/\\|?*]', '_', video_name)

            # Look for common thumbnail extensions in the expected location
            found_filename = None
            common_thumb_extensions = [".webp", ".jpg", ".jpeg", ".png", ".gif"]
            for ext in common_thumb_extensions:
                try:
                    # Construct potential path: base / sanitized_channel / sanitized_video_name.ext
                    thumb_path = download_base_path / safe_channel / f"{safe_filename_base}{ext}"
                    # Check if the file exists and is readable
                    if thumb_path.is_file() and os.access(thumb_path, os.R_OK):
                        found_filename = f"{safe_filename_base}{ext}" # Store the name with extension
                        logger.debug(f"Found thumbnail for '{video_name}': {thumb_path}")
                        break # Stop searching once a thumbnail is found
                except (OSError, ValueError, TypeError) as path_err:
                    # Log errors during path construction/checking but continue trying other extensions
                    # These errors are common if filenames have very unusual characters
                    logger.debug(f"Error checking potential thumbnail path {thumb_path}: {path_err}")
                    pass # Ignore and try next extension

            # If a thumbnail file was found, construct the API URL for it
            if found_filename:
                try:
                    # URL-encode the channel and filename components separately
                    # safe='' ensures slashes are encoded if any remain after sanitization (unlikely but possible)
                    encoded_channel = url_quote(safe_channel, safe='')
                    encoded_filename = url_quote(found_filename, safe='')
                    # Construct the API URL for the frontend to use to fetch the thumbnail
                    thumbnail_url_to_send = f"/api/tartube_thumbnails/{encoded_channel}/{encoded_filename}"
                    logger.debug(f"Generated thumbnail URL: {thumbnail_url_to_send}")
                except Exception as enc_err:
                    # Log error during URL encoding but don't fail the whole item
                    logger.error(f"Error URL encoding thumbnail parts for video '{video_name}': {enc_err}")
                    thumbnail_url_to_send = None # Ensure URL is None if encoding failed


        # --- Create Result Item (using Pydantic model) ---
        # This step also validates the data types
        result_item = models.YouTubeResultItem(
            title=video_name,
            url=video_url,
            vid=video_vid,
            snippet=video_desc, # Can be None
            channel_name=channel_name, # Can be None
            file_path=video_data.get("file"), # Include local file path if available in export (can be None)
            thumbnail_url=thumbnail_url_to_send # Will be None if not found/configured/errored
        )
        # Return success: match found, result created, no error message
        return True, result_item, None

    except models.ValidationError as pydantic_err:
        # Handle errors if the video_data doesn't match the YouTubeResultItem model
        error_message = f"Data validation error for video '{video_name_for_log}': {pydantic_err}"
        logger.error(f"Pydantic validation error creating YouTubeResultItem: {pydantic_err} for data: {video_data}")
        # Return failure: match status uncertain, no result, error message
        return False, None, error_message
    except Exception as e:
        # Catch any other unexpected errors during processing of this single item
        error_message = f"Internal error processing video '{video_name_for_log}': {e}"
        logger.exception(f"Unexpected error processing Tartube video item: {video_data}")
        # Return failure: match status uncertain, no result, error message
        return False, None, error_message


async def search_photo_directory(photo_base_path: FilePath, query: str) -> Tuple[List[models.PhotoResultItem], List[str]]:
    """Searches image filenames in the photo directory for the query."""
    results = []
    errors = []
    query_lower = query.lower()
    # Define allowed image extensions (case-insensitive check using .lower())
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
    logger.info(f"Starting Photo search in '{photo_base_path}' for query '{query}'")

    if not photo_base_path.is_dir():
        errors.append(f"Configured Photos path is not a valid directory: {photo_base_path}")
        logger.error(errors[-1])
        return [], errors

    try:
        # Recursively walk the directory using rglob
        for item_path in photo_base_path.rglob('*'):
            # Check if it's a file and has an allowed extension
            if item_path.is_file() and item_path.suffix.lower() in allowed_extensions:
                # Check if query appears in the filename (case-insensitive)
                if query_lower in item_path.name.lower():
                    try:
                        # Calculate relative path from the base for the API response/URL
                        relative_path = item_path.relative_to(photo_base_path)
                        relative_path_str = relative_path.as_posix() # Use forward slashes for consistency

                        # Get file metadata (last modified time)
                        item_stat = item_path.lstat() # Use lstat to avoid following symlinks for stats
                        # Convert timestamp to timezone-aware datetime object (UTC)
                        modified_time = datetime.datetime.fromtimestamp(item_stat.st_mtime, tz=datetime.timezone.utc)

                        # Create result item using Pydantic model
                        results.append(models.PhotoResultItem(
                            filename=item_path.name,
                            relative_path=relative_path_str,
                            modified_time=modified_time
                        ))
                    # Handle potential errors during processing of a single file
                    except PermissionError:
                        # Try to get relative path for error message, fallback to name
                        rel_path_err_str = item_path.name
                        try: rel_path_err_str = item_path.relative_to(photo_base_path).as_posix()
                        except: pass
                        err_msg = f"Permission denied accessing metadata for photo: {rel_path_err_str}"
                        logger.warning(err_msg)
                        errors.append(err_msg) # Add non-fatal error to list
                    except FileNotFoundError: # Should be rare after is_file(), but handles race conditions
                         rel_path_err_str = item_path.name
                         try: rel_path_err_str = item_path.relative_to(photo_base_path).as_posix()
                         except: pass
                         err_msg = f"Photo file disappeared during processing: {rel_path_err_str}"
                         logger.warning(err_msg)
                         errors.append(err_msg)
                    except Exception as e: # Catch other errors (e.g., during relative_to, timestamp conversion)
                         rel_path_err_str = item_path.name
                         try: rel_path_err_str = item_path.relative_to(photo_base_path).as_posix()
                         except: pass
                         err_msg = f"Error processing photo file {rel_path_err_str}: {e}"
                         logger.exception(err_msg) # Log stack trace for unexpected errors
                         errors.append(err_msg)

    except PermissionError:
        # Error iterating through the base directory itself
        err_msg = f"Permission denied listing files in photo path: {photo_base_path}"
        logger.error(err_msg)
        errors.append(err_msg) # Add fatal error to list
    except Exception as e:
        # Catch other errors during the directory walk (rglob)
        err_msg = f"Unexpected error scanning photo directory: {e}"
        logger.exception(f"Unexpected error walking photo directory {photo_base_path}")
        errors.append(err_msg)

    logger.info(f"Photo search completed. Found {len(results)} results. Encountered {len(errors)} errors.")
    # Sort results alphabetically by filename
    results.sort(key=lambda x: x.filename.lower())
    return results, errors


async def search_music_directory(music_base_path: FilePath, query: str) -> Tuple[List[models.MusicResultItem], List[str]]:
    """Searches music file tags (title, artist, album) and filename for the query."""
    results = []
    errors = []
    query_lower = query.lower()
    # Supported audio file extensions for mutagen
    allowed_extensions = {'.mp3', '.flac', '.ogg', '.m4a', '.opus', '.wav', '.aac', '.wma', '.ape'}
    logger.info(f"Starting Music search in '{music_base_path}' for query '{query}'")

    # Check if Mutagen is available before proceeding
    if not MUTAGEN_AVAILABLE:
         errors.append("Music search disabled: Mutagen library is not installed.")
         logger.error(errors[-1])
         return [], errors # Return immediately if dependency is missing

    if not music_base_path.is_dir():
        errors.append(f"Configured Music path is not a valid directory: {music_base_path}")
        logger.error(errors[-1])
        return [], errors

    try:
        # Recursively walk the directory
        for item_path in music_base_path.rglob('*'):
            # Check if it's a file and has an allowed audio extension
            if item_path.is_file() and item_path.suffix.lower() in allowed_extensions:
                try:
                    # Initialize variables for tags and status
                    title = None; artist = None; album = None
                    metadata_error = None # Store any error during tag reading
                    search_hit = False

                    # --- Read Metadata using Mutagen ---
                    try:
                        # Use mutagen.File for automatic format detection
                        # easy=True provides simple common tags like 'title', 'artist', 'album'
                        audio = mutagen.File(item_path, easy=True)
                        if audio:
                            # Access tags safely using .get(), providing default None
                            # Mutagen EasyID3/EasyMP4 tags often return lists, take the first element
                            title_list = audio.get('title')
                            artist_list = audio.get('artist')
                            album_list = audio.get('album')
                            title = title_list[0] if title_list else None
                            artist = artist_list[0] if artist_list else None
                            album = album_list[0] if album_list else None

                    except mutagen.MutagenError as me:
                        # Catch errors specific to Mutagen (e.g., invalid file format, corrupt tags)
                        metadata_error = f"Mutagen error reading tags for {item_path.name}: {me}"
                        logger.debug(metadata_error) # Debug level as some files might lack tags or be slightly corrupt
                    except Exception as meta_e:
                        # Catch other unexpected errors during metadata reading
                        metadata_error = f"Unexpected error reading metadata for {item_path.name}: {meta_e}"
                        logger.warning(metadata_error) # Warn for unexpected errors

                    # --- Query Matching (Filename and Tags) ---
                    # 1. Check filename (case-insensitive)
                    if query_lower in item_path.name.lower():
                        search_hit = True
                    # 2. Check tags only if filename didn't match (avoid redundant checks)
                    if not search_hit:
                         if title and query_lower in title.lower(): search_hit = True
                         elif artist and query_lower in artist.lower(): search_hit = True
                         elif album and query_lower in album.lower(): search_hit = True

                    # --- Create Result if Match Found ---
                    if search_hit:
                        # Calculate relative path
                        relative_path = item_path.relative_to(music_base_path)
                        relative_path_str = relative_path.as_posix()

                        # Get modified time
                        item_stat = item_path.lstat()
                        modified_time = datetime.datetime.fromtimestamp(item_stat.st_mtime, tz=datetime.timezone.utc)

                        # Use title tag if available, otherwise fallback to filename stem for display
                        display_title = title if title else item_path.stem

                        # Create result item using Pydantic model
                        results.append(models.MusicResultItem(
                            filename=item_path.name,
                            relative_path=relative_path_str,
                            title=display_title, # Use the determined display title
                            artist=artist, # Will be None if not found
                            album=album,   # Will be None if not found
                            modified_time=modified_time
                        ))
                        # If there was a metadata reading error for this matched file, log it as warning
                        if metadata_error:
                             logger.warning(f"Metadata issue noted for matched music file '{item_path.name}': {metadata_error}")

                # Handle potential errors during processing of a single music file
                except PermissionError:
                    rel_path_err_str = item_path.name
                    try: rel_path_err_str = item_path.relative_to(music_base_path).as_posix()
                    except: pass
                    err_msg = f"Permission denied accessing music file: {rel_path_err_str}"
                    logger.warning(err_msg)
                    errors.append(err_msg)
                except FileNotFoundError: # Race condition check
                     rel_path_err_str = item_path.name
                     try: rel_path_err_str = item_path.relative_to(music_base_path).as_posix()
                     except: pass
                     err_msg = f"Music file disappeared during processing: {rel_path_err_str}"
                     logger.warning(err_msg)
                     errors.append(err_msg)
                except Exception as e: # Catch other errors (path, time conversion, etc.)
                     rel_path_err_str = item_path.name
                     try: rel_path_err_str = item_path.relative_to(music_base_path).as_posix()
                     except: pass
                     err_msg = f"Error processing music file {rel_path_err_str}: {e}"
                     logger.exception(err_msg)
                     errors.append(err_msg)

    except PermissionError:
        # Error iterating through the base music directory
        err_msg = f"Permission denied listing files in music path: {music_base_path}"
        logger.error(err_msg)
        errors.append(err_msg)
    except Exception as e:
        # Catch other errors during the directory walk (rglob)
        err_msg = f"Unexpected error scanning music directory: {e}"
        logger.exception(f"Unexpected error walking music directory {music_base_path}")
        errors.append(err_msg)

    logger.info(f"Music search completed. Found {len(results)} results. Encountered {len(errors)} errors.")
    # Sort primarily by artist, then album, then title for standard music library sorting
    results.sort(key=lambda x: (
        x.artist.lower() if x.artist else 'zzz', # Put items without artist last
        x.album.lower() if x.album else 'zzz',   # Put items without album last within artist
        x.title.lower() if x.title else 'zzz'    # Sort by title (including fallback) within album
    ))
    return results, errors


# --- SearXNG Client ---

async def query_searxng(query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Queries the configured SearXNG instance using httpx."""
    base_url_str = str(config.SEARXNG_URL).rstrip('/') + '/' # Ensure base URL ends with a slash
    # Default parameters always sent to SearXNG
    searxng_query_params = {
        "q": query,
        "format": "json" # Always request JSON format
    }

    # Merge additional parameters from the request if provided
    if params:
        # Filter out any params with None values before merging
        filtered_params = {k: v for k, v in params.items() if v is not None}
        searxng_query_params.update(filtered_params)
        # Ensure format remains json even if passed in params mistakenly
        searxng_query_params["format"] = "json"

    # Configure timeout (consider making these values configurable via settings/env)
    timeout = httpx.Timeout(15.0, connect=5.0) # 15s total wait, 5s for connection establishment

    # Use httpx.AsyncClient for asynchronous requests, enabling HTTP/2 if available
    async with httpx.AsyncClient(follow_redirects=True, http2=True, timeout=timeout) as client:
        logger.info(f"Querying SearXNG instance at {base_url_str} with effective params: {searxng_query_params}")
        try:
            # Perform the GET request
            response = await client.get(base_url_str, params=searxng_query_params)

            # Raise an exception for non-2xx status codes (4xx, 5xx)
            response.raise_for_status()

            # Log success and response preview
            logger.info(f"SearXNG responded with status: {response.status_code}")
            response_text_preview = response.text[:500] + ('...' if len(response.text) > 500 else '')
            logger.debug(f"SearXNG Raw Response Preview: {response_text_preview}")

            # Try parsing the JSON response body
            try:
                return response.json()
            except json.JSONDecodeError as json_err:
                # Handle cases where SearXNG returns 2xx but invalid JSON
                logger.error(f"SearXNG returned non-JSON response ({response.status_code}). Body preview: {response_text_preview}. Error: {json_err}")
                raise HTTPException(status_code=502, detail=f"SearXNG returned invalid JSON response (status {response.status_code}).")

        # Handle specific httpx network/request errors
        except httpx.TimeoutException as exc:
            logger.error(f"Timeout error connecting to SearXNG at {base_url_str}: {exc}")
            raise HTTPException(status_code=504, detail="SearXNG request timed out.")
        except httpx.ConnectError as exc:
            logger.error(f"Connection error contacting SearXNG at {base_url_str}: {exc}")
            raise HTTPException(status_code=503, detail="Could not connect to SearXNG instance.")
        except httpx.RequestError as exc: # More general request errors (DNS, protocol errors etc.)
            logger.error(f"Request error contacting SearXNG: {exc}")
            raise HTTPException(status_code=502, detail=f"Network error contacting SearXNG: {exc}")

        # Handle non-2xx responses raised by response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_text = exc.response.text[:500] + ('...' if len(exc.response.text) > 500 else '')
            logger.error(f"SearXNG returned error status {status_code}. Response preview: {response_text}")

            # Attempt to extract a more specific error message from SearXNG JSON response if available
            detail_msg = f"SearXNG instance returned error: {status_code}"
            try:
                error_json = exc.response.json()
                if isinstance(error_json, dict):
                    if 'detail' in error_json:
                        detail_msg = f"SearXNG Error ({status_code}): {error_json['detail']}"
                    elif 'message' in error_json: # Check for 'message' as another common error key
                        detail_msg = f"SearXNG Error ({status_code}): {error_json['message']}"
            except json.JSONDecodeError:
                # Ignore if the error response body isn't valid JSON
                pass
            # Raise as 502 Bad Gateway, indicating an issue with the upstream service (SearXNG)
            raise HTTPException(status_code=502, detail=detail_msg)

        # Catch any other unexpected errors during the request/response cycle
        except Exception as e:
            logger.exception("An unexpected error occurred during SearXNG query.")
            raise HTTPException(status_code=500, detail=f"An internal error occurred communicating with SearXNG: {e}")


# --- Engine Catalog and Active Config Helpers ---

async def get_searxng_engines() -> List[models.EngineDetail]:
    """
    Loads, validates, and returns the list of known SearXNG engines from the engine catalog file.
    Returns an empty list if the file is missing, invalid, or empty after validation.
    """
    logger.info(f"Attempting to load engine catalog from '{config.ENGINE_CATALOG_FILE}'.")
    catalog_path = FilePath(config.ENGINE_CATALOG_FILE)

    if not catalog_path.is_file():
        logger.error(f"Engine catalog file not found: {catalog_path}")
        return [] # Return empty list if file doesn't exist

    validated_engines = []
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            engines_data = json.load(f)

        # Ensure the loaded data is a list
        if not isinstance(engines_data, list):
            logger.error(f"Engine catalog file '{catalog_path}' does not contain a valid JSON list.")
            return [] # Return empty list if structure is wrong

        # Validate each engine entry against the Pydantic model
        for i, item in enumerate(engines_data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dictionary item #{i+1} in '{catalog_path}'.")
                continue
            try:
                # Pydantic will use default values for missing optional fields (like enabled, weight, etc.)
                # Validate the dictionary against the EngineDetail model
                validated_engine = models.EngineDetail(**item)
                validated_engines.append(validated_engine)
            except models.ValidationError as val_err:
                # Log validation errors for specific entries but continue processing others
                engine_id = item.get('id', f'Entry #{i+1}') # Get ID for logging if possible
                logger.warning(f"Validation error for engine '{engine_id}' in '{catalog_path}': {val_err}. Skipping.")
            except TypeError as type_err: # Catch potential issues if data types are wrong before Pydantic validation
                engine_id = item.get('id', f'Entry #{i+1}')
                logger.warning(f"Type error processing engine data for '{engine_id}' in '{catalog_path}': {type_err}. Skipping.")

        # Sort the successfully validated engines alphabetically by name
        validated_engines.sort(key=lambda x: x.name.lower())
        logger.info(f"Successfully loaded and validated {len(validated_engines)} engines from '{config.ENGINE_CATALOG_FILE}'.")
        return validated_engines # Return the list of validated Pydantic models

    # Handle file reading and JSON parsing errors
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from engine catalog file '{catalog_path}': {e}")
        return [] # Return empty list on JSON error
    except IOError as e:
        logger.error(f"I/O error reading engine catalog file '{catalog_path}': {e}")
        return [] # Return empty list on file read error
    except Exception as e:
        # Catch any other unexpected errors during loading/validation
        logger.exception(f"Unexpected error loading engine catalog from '{catalog_path}'")
        return [] # Return empty list on unexpected error


def get_active_engine_config(settings: Dict[str, Any], available_engines: List[models.EngineDetail]) -> List[Dict[str, Any]]:
    """
    Determines the active engine configuration based on saved settings and available engine definitions.
    Returns a list of *dictionaries*, where each dictionary represents the final merged
    configuration for an engine in the active loadout (including enabled status, weight, timeout, etc.).
    Handles merging: Starts with catalog defaults, overrides with saved loadout specifics.
    Synthesizes a 'starter' config (based on catalog defaults) if active loadout is 'starter' or invalid.
    """
    engine_settings_dict = settings.get('engines', {})
    active_loadout_id = engine_settings_dict.get('activeLoadoutId', 'starter')
    all_loadouts_raw = engine_settings_dict.get('loadouts', []) # Could be list of dicts or Pydantic models

    # Convert available_engines (list of Pydantic models) into a dictionary map {id: engine_dict}
    # for efficient lookup. Use model_dump to get dictionary representation, include default values.
    available_engines_map: Dict[str, Dict[str, Any]] = {
        eng.id: eng.model_dump(exclude_unset=False) for eng in available_engines
    }

    # Helper to parse loadouts array which might contain dicts or Pydantic models from loading
    def parse_loadouts(raw_loadouts):
        parsed = []
        if not isinstance(raw_loadouts, list): return []
        for idx, lo in enumerate(raw_loadouts):
            if isinstance(lo, models.EngineLoadoutItem):
                # If already a model, dump it to dict, include defaults
                parsed.append(lo.model_dump(exclude_unset=False))
            elif isinstance(lo, dict):
                try:
                    # If it's a dict, validate it against the model and dump it
                    # This ensures consistency even if saved data was slightly off
                    parsed.append(models.EngineLoadoutItem(**lo).model_dump(exclude_unset=False))
                except models.ValidationError as e:
                   loadout_name = lo.get('name', f'Loadout #{idx+1}')
                   logger.warning(f"Skipping invalid loadout structure in settings for '{loadout_name}': {e}")
                   continue # Skip invalid loadout structures
            else:
                logger.warning(f"Skipping unexpected item type in loadouts list: {type(lo)}")
        return parsed

    # Parse the raw loadouts into a consistent list of dictionaries
    all_loadouts = parse_loadouts(all_loadouts_raw)

    # --- Find the target saved loadout based on activeLoadoutId ---
    target_loadout_dict = None
    if active_loadout_id != 'starter':
        # Find the dictionary in the parsed list that matches the active ID
        target_loadout_dict = next((l for l in all_loadouts if l.get('id') == active_loadout_id), None)

    # --- Determine if we need to use the 'starter' config ---
    # Use starter if:
    # 1. activeLoadoutId is explicitly 'starter'
    # 2. activeLoadoutId is set to something else, but that ID wasn't found in the parsed loadouts
    use_starter_config = (active_loadout_id == 'starter') or (target_loadout_dict is None and active_loadout_id != 'starter')

    if use_starter_config:
        if active_loadout_id != 'starter' and target_loadout_dict is None:
             # Log clearly if falling back due to missing ID
             logger.warning(f"Active engine loadout ID '{active_loadout_id}' not found in saved loadouts. Falling back to synthesized 'Starter' configuration.")

        logger.debug("Synthesizing 'Starter' engine config using defaults from engine catalog.")
        # Create starter config based purely on the 'enabled' status and other defaults defined in the engine_catalog.json
        starter_config = []
        for engine_id, catalog_engine_details in available_engines_map.items():
             # The starter config directly reflects the defaults loaded from the catalog
             starter_config.append({
                **catalog_engine_details, # Include all details from catalog (name, categories, description, etc.)
                # Ensure critical fields have fallbacks if somehow missing in catalog data
                "enabled": catalog_engine_details.get('enabled', False),
                "weight": catalog_engine_details.get('weight', 1.0),
                "timeout": catalog_engine_details.get('timeout', 3.0),
             })
        return starter_config

    # --- Case 2: Use a found saved loadout (target_loadout_dict is valid) ---
    else:
        loadout_name = target_loadout_dict.get('name', active_loadout_id) # Use name for logging
        logger.debug(f"Using saved engine loadout config: '{loadout_name}' (ID: {active_loadout_id})")

        # Extract the list of engine configurations from the saved loadout
        saved_engine_configs_list = target_loadout_dict.get('config', []) # Should be a list of dicts/models

        # Normalize the saved engine config items into a dictionary map {id: config_dict} for efficient lookup
        saved_engine_configs_map = {}
        if isinstance(saved_engine_configs_list, list):
            for idx, lc in enumerate(saved_engine_configs_list):
                engine_id = None
                config_dict = {}
                if isinstance(lc, models.EngineDetail): # If loaded as model
                    engine_id = lc.id
                    config_dict = lc.model_dump(exclude_unset=False) # Use dict from model
                elif isinstance(lc, dict) and lc.get('id'): # If loaded as dict
                    engine_id = lc.get('id')
                    # Validate the dictionary against the model for safety? Optional.
                    try:
                        config_dict = models.EngineDetail(**lc).model_dump(exclude_unset=False)
                    except models.ValidationError:
                        logger.warning(f"Skipping invalid engine config item #{idx+1} in loadout '{loadout_name}': {lc}")
                        continue
                else:
                    logger.warning(f"Skipping invalid engine config item structure in loadout '{loadout_name}': {lc}")
                    continue # Skip malformed entries

                if engine_id:
                     # Store the validated/dumped dictionary
                     saved_engine_configs_map[str(engine_id)] = config_dict
        else:
            # Log if the 'config' field in the saved loadout wasn't a list
            logger.warning(f"Loadout '{loadout_name}' has invalid 'config' (expected list, got {type(saved_engine_configs_list).__name__}). Treating loadout as empty.")
            saved_engine_configs_map = {} # Ensure it's an empty map

        # --- Merge catalog defaults with saved loadout specifics ---
        final_merged_config = []
        # Iterate through *all engines available in the catalog* as the source of truth for existence
        for avail_id, catalog_engine_dict in available_engines_map.items():
            # Find the corresponding configuration for this engine *within the saved loadout*
            engine_config_from_saved_loadout = saved_engine_configs_map.get(avail_id)

            # Start with the full, default details from the catalog as a base
            merged_engine = {**catalog_engine_dict}

            if engine_config_from_saved_loadout:
                # If this engine *is* mentioned in the saved loadout:
                # Override specific fields based on the saved configuration.
                # The 'enabled' status comes *only* from the saved loadout's entry for this engine.
                merged_engine['enabled'] = engine_config_from_saved_loadout.get('enabled', False) # Default to disabled if 'enabled' is missing in saved item

                # Weight and timeout are also taken from the saved loadout if present,
                # otherwise, they retain the default value from the catalog dictionary.
                merged_engine['weight'] = engine_config_from_saved_loadout.get('weight', catalog_engine_dict.get('weight', 1.0))
                merged_engine['timeout'] = engine_config_from_saved_loadout.get('timeout', catalog_engine_dict.get('timeout', 3.0))
                # Note: Other fields like name, categories, description always come from the catalog baseline.
            else:
                # If this engine (available in the catalog) is *not* mentioned in the saved loadout's 'config' list,
                # it must be considered disabled for this specific loadout.
                merged_engine['enabled'] = False
                # Weight/timeout retain their catalog defaults but are irrelevant if disabled.

            # Add the final merged dictionary for this engine to the result list
            final_merged_config.append(merged_engine)

        # Return the list of dictionaries representing the complete configuration for the active loadout
        return final_merged_config


# --- AI Placeholder ---
# Keep this simple placeholder function as defined in the original refactoring. Phase 2 will build upon this.
async def enhance_results_with_ai(results: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Placeholder for AI processing of search results (e.g., RAG)."""
    # In a real scenario, this would involve:
    # 1. Getting AI config from settings.
    # 2. Initializing an LLM chain (potentially using Langchain).
    # 3. Formatting the results and query into a prompt.
    # 4. Calling the LLM.
    # 5. Parsing the LLM response and integrating it into the results dict.
    logger.info(f"Placeholder: AI processing invoked for query '{query}'. No actual enhancement performed.")
    # Add a marker to indicate it was called, even if it did nothing yet
    results['ai_processed_placeholder'] = True
    return results
