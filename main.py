# main.py (Corrected Imports & Entry Point)
import logging
import sys
import os

# Set up logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # REMOVED - Let uvicorn handle basic config

# Add the current directory to Python path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Corrected Imports ---
# Use direct imports now that backend is not treated as a package
import main_config
import main_routes
# Import directly from the routers directory/file
from routers import chat as chat_router
from routers import engine_routes as engine_router
# from routers import freshrss as freshrss_router # <<< REMOVED Import the new FreshRSS router correctly
# --- End Corrected Imports ---

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SlySearch Backend API",
    description="Modular backend service for SlySearch, including SearXNG proxy, AI features, settings management, local source searching, and more.",
    version="1.1.1" # Version bump for import fix
)

# --- Configure CORS Middleware ---
logger.info(f"Configuring CORS for origins: {main_config.ALLOWED_ORIGINS}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=main_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(main_routes.router)
logger.info("Included main API router (search, settings, vault, files, searxng_engines).")

# <<< REMOVED ENTIRE TRY/EXCEPT BLOCK for freshrss_router >>>
# try:
#     app.include_router(
#         freshrss_router.router, # Use the imported freshrss_router object
#         tags=["FreshRSS"] # Use the tag defined in the router or a new one
#         # prefix="/freshrss" # Prefix is already defined in the router itself
#     )
#     logger.info("Included FreshRSS router.") # Log inclusion
# except RuntimeError as e:
#      logger.error(f"--- FATAL: Failed to initialize and include FreshRSS router: {e} ---", exc_info=True)

try:
    app.include_router(
        chat_router.router, # Use the imported chat_router object
        prefix="/api/v1", # All routes in chat.py will be prefixed with /api/v1
        tags=["AI Chat"]  # Group these endpoints under 'AI Chat' in docs
    )
    logger.info("Included v1 chat router under /api/v1.")
except RuntimeError as e:
     # Catch initialization errors from the chat router if it uses Depends for setup
     logger.error(f"--- FATAL: Failed to initialize and include chat router: {e} ---", exc_info=True)
     # Depending on criticality, you might want the app to exit or prevent startup.
     # raise e # Optional: prevent startup if chat is critical

try:
    app.include_router(
        engine_router.router, # Use the imported engine_router object
        tags=["Engine Selection"]  # Group these endpoints under 'Engine Selection' in docs
    )
    logger.info("Included engine selection router.")
except RuntimeError as e:
    logger.error(f"--- FATAL: Failed to initialize and include engine router: {e} ---", exc_info=True)

# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/health", tags=["Status"])
async def health_check():
    """Basic health check endpoint."""
    # Add checks here later (e.g., can connect to SearXNG? Ollama? Database?)
    return {"status": "ok"}


# --- Main Execution Block ---
# Use this block ONLY if you intend to run `python main.py` directly.
# Running `uvicorn main:app ...` from within the backend directory is now the expected way.
if __name__ == "__main__":
    # Package handling warning removed as it's no longer relevant
    logger.info(f"Starting Uvicorn server directly...")
    logger.info(f" >> Listening on: {main_config.APP_HOST}:{main_config.APP_PORT}")
    logger.info(f" >> Allowed CORS Origins: {main_config.ALLOWED_ORIGINS}")
    logger.info(f" >> Access API Docs at: http://{main_config.APP_HOST}:{main_config.APP_PORT}/docs")

    uvicorn.run(
        "main:app",        # Point uvicorn to the FastAPI app instance ('app') in *this* file ('main.py')
        host=main_config.APP_HOST,     # Use host loaded from config module
        port=main_config.APP_PORT,     # Use port loaded from config module
        log_level="debug", # Set uvicorn's logging level to debug
        reload=True        # Enable auto-reload for development
    )
