# main_config.py
import logging
import os
import json
from typing import Dict, Any
from pathlib import Path as FilePath
from decouple import config, UndefinedValueError
from starlette.datastructures import URL

# --- Core Application Configuration ---
try:
    # Using default values that match the original main.py if not set in .env
    SEARXNG_URL: URL = config('SEARXNG_URL', cast=URL, default='http://localhost:8088')
    ALLOWED_ORIGINS: list[str] = config(
        'ALLOWED_ORIGINS',
        cast=lambda v: [s.strip() for s in v.split(',')],
        default="http://localhost:3000,http://127.0.0.1:3000"
    )
    APP_HOST: str = config("APP_HOST", default="127.0.0.1")
    APP_PORT: int = config("APP_PORT", cast=int, default=8000)

except UndefinedValueError as e:
    logging.error(f"Missing critical configuration in environment variables: {e}")
    # You might want to raise an exception here to prevent startup
    # For now, provide safe defaults or indicate missing config
    SEARXNG_URL = URL('http://localhost:8088') # Example default
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    APP_HOST = "127.0.0.1"
    APP_PORT = 8000
    logging.warning(f"Using default core config due to missing environment variables. Please check setup.")
except Exception as e:
     logging.exception(f"Error loading core configuration: {e}")
     # Handle unexpected errors during config loading if necessary
     raise RuntimeError(f"Failed to load core configuration: {e}")


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Logger specific to this module if needed

# Log loaded config
logger.info(f"--- Backend Core Config ---")
logger.info(f"SearXNG URL: {SEARXNG_URL}")
logger.info(f"Allowed CORS Origins: {ALLOWED_ORIGINS}")
logger.info(f"App Host: {APP_HOST}")
logger.info(f"App Port: {APP_PORT}")
logger.info(f"---------------------------")


# --- File Path Constants ---
# Determine script directory to make paths relative to the script location
# Assuming this file is in the same directory as the original main.py
script_dir = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(script_dir, "app_settings.json")
ENGINE_CATALOG_FILE = os.path.join(script_dir, "engine_catalog.json")

# --- Default values for settings from app_settings.json ---
# Keeping OLLAMA_BASE_CHAT_MODEL_DEFAULT for chat functionality
OLLAMA_BASE_CHAT_MODEL_DEFAULT = "llama3:8b-instruct-q8_0"

# --- Load Settings from app_settings.json ---
app_settings: Dict[str, Any] = {}
try:
    with open(SETTINGS_FILE, 'r') as f:
        app_settings = json.load(f)
    logger.info(f"Successfully loaded settings from {SETTINGS_FILE}")
except FileNotFoundError:
    logger.warning(f"Settings file not found at {SETTINGS_FILE}. Using defaults for user-configurable settings.")
except json.JSONDecodeError:
    logger.error(f"Error decoding JSON from {SETTINGS_FILE}. Using defaults for user-configurable settings.")
except Exception as e:
    logger.exception(f"Unexpected error loading {SETTINGS_FILE}: {e}. Using defaults.")

# Extract Auto Navigation settings or use defaults
auto_nav_settings = app_settings.get("auto_navigation", {}) # Get the section, default to empty dict

# Only keep the base chat model, remove engine selector related configs
OLLAMA_BASE_CHAT_MODEL: str = auto_nav_settings.get("base_chat_model", OLLAMA_BASE_CHAT_MODEL_DEFAULT)

logger.info("--- Auto Navigation Settings (from app_settings.json or defaults) ---")
logger.info(f"Base Chat Model: {OLLAMA_BASE_CHAT_MODEL}")
logger.info("-------------------------------------------------------------------")


# --- Load Remaining Config from Environment ---
# Ollama Base URL remains in .env
try:
    OLLAMA_BASE_URL = config('OLLAMA_BASE_URL', cast=str)
    logger.info(f"Ollama Base URL (from environment): '{OLLAMA_BASE_URL}'")
except UndefinedValueError:
    logger.error("Missing OLLAMA_BASE_URL in environment variables. Using default.")
    OLLAMA_BASE_URL = "http://localhost:11434" # Example default

# Maintain ENGINE_CATALOG_PATH for loadout functionality
try:
    ENGINE_CATALOG_PATH = config('ENGINE_CATALOG_PATH', default='engine_catalog.json')
    logger.info("--- Engine Paths (from environment or defaults) ---")
    logger.info(f"Engine Catalog Path: {ENGINE_CATALOG_PATH}")
    logger.info("--------------------------------------------------------------")
except UndefinedValueError as e:
    # This shouldn't happen if defaults are provided in config()
    logger.error(f"Unexpected UndefinedValueError for engine paths: {e}")
    # Set explicit defaults again just in case
    ENGINE_CATALOG_PATH = 'engine_catalog.json'
except Exception as e:
    logger.exception(f"Error loading engine paths configuration: {e}")
    raise RuntimeError(f"Failed to load engine paths configuration: {e}")

# --- LLM Configuration ---
# Moved from engine_round_robin.py for centralized config
LLM_MODEL = "llama3:8b-instruct-q8_0"
LLM_TEMPERATURE = 0.2
LLM_REQUEST_TIMEOUT = 120.0
LLM_CTX = 4096
