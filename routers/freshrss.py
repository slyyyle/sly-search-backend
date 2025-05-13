import httpx
import base64
import time
import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
# from dotenv import load_dotenv # Remove if previously used here

# Load settings utility
from main_utils import load_settings # Keep this

router = APIRouter()

# Pydantic model for the structure of a FreshRSS search result item (adjust as needed)
class FreshRSSResultItemModel(BaseModel):
    id: str
    crawlTimeMsec: str
    timestampUsec: str
    title: str
    published: int
    updated: int
    canonical: List[Dict[str, str]]
    alternate: List[Dict[str, str]]
    summary: Dict[str, str]
    author: str
    likingUsers: List[Any] # Replace Any with a more specific model if structure is known
    comments: List[Any] # Replace Any with a more specific model if structure is known
    annotations: List[Any] # Replace Any with a more specific model if structure is known
    origin: Dict[str, Any] # Replace Any with a more specific model if structure is known
    categories: List[str]
    # Map backend fields to frontend fields
    url: Optional[str] = None # Will be populated from alternate or canonical
    snippet: Optional[str] = Field(None, alias='summary_content') # Map summary.content to snippet
    published_time: Optional[int] = Field(None, alias='published')
    feed_title: Optional[str] = Field(None, alias='origin_title') # Map origin.title
    
    # Add fields expected by the frontend SearchResultItem union type
    source: str = 'freshrss'
    result_type: str = 'freshrss'
    
    class Config:
        populate_by_name = True # Allows using alias for population

    def model_post_init(self, __context: Any) -> None:
        # More robust URL extraction
        self.url = '' # Default to empty string
        try:
            html_link = next((link.get('href') for link in self.alternate if link and isinstance(link, dict) and link.get('type') == 'text/html'), None)
            if html_link:
                self.url = html_link
            elif self.canonical and isinstance(self.canonical, list) and len(self.canonical) > 0 and isinstance(self.canonical[0], dict) and self.canonical[0].get('href'):
                self.url = self.canonical[0]['href']
            # If URL is still empty, log a warning or leave it empty
            if not self.url:
                print(f"Warning: Could not extract URL for item ID: {self.id}")
        except Exception as e:
            print(f"Error extracting URL for item ID {self.id}: {e}")
            self.url = '' # Ensure url is empty string on error

        # Extract snippet/content from summary
        try:
            self.snippet = self.summary.get('content') if isinstance(self.summary, dict) else None
        except Exception as e:
            print(f"Error extracting snippet for item ID {self.id}: {e}")
            self.snippet = None

        # Extract feed title
        try:
            self.feed_title = self.origin.get('title') if isinstance(self.origin, dict) else None
        except Exception as e:
            print(f"Error extracting feed_title for item ID {self.id}: {e}")
            self.feed_title = None
            
        # Ensure categories is always a list of strings (last part)
        try:
            processed_categories = []
            if isinstance(self.categories, list):
                for cat in self.categories:
                    if isinstance(cat, str) and cat.startswith('user/'):
                        parts = cat.split('/')
                        if len(parts) > 0:
                            processed_categories.append(parts[-1])
            self.categories = processed_categories
        except Exception as e:
            print(f"Error processing categories for item ID {self.id}: {e}")
            self.categories = [] # Default to empty list on error


class FreshRSSSearchResponseModel(BaseModel):
    results: List[FreshRSSResultItemModel]
    query: str
    total_results: int
    source: str = 'freshrss'
    errors: Optional[List[str]] = None

# Function to get authentication token
def get_freshrss_token(username: str, password: str) -> str:
    """Generates the Basic Auth token for FreshRSS API."""
    credentials = f"{username}:{password}"
    token = base64.b64encode(credentials.encode()).decode()
    return f"Basic {token}"

# Timeout configuration for HTTP requests
TIMEOUT_CONFIG = httpx.Timeout(10.0, connect=5.0)

# +++ Add inspect import +++
# import inspect # <<< REMOVED

# <<< REMOVE/COMMENT OUT DECORATOR >>>
# @router.get("/search/freshrss", response_model=FreshRSSSearchResponseModel)
async def search_freshrss(
    query: str, # Remove Query(...) default if not an endpoint
    settings: dict = None,
    pageno: int = 1,
    results_per_page: int = 10
):
    """
    Searches articles in FreshRSS instance using the GReader API compatible endpoint.
    Accepts settings, pagination, and result limit parameters.
    """
    # +++ Add signature logging +++
    # print(f"[DEBUG freshrss] search_freshrss called with signature: {inspect.signature(search_freshrss)}") # <<< REMOVED
    # --- 

    # settings = load_settings() # <<< REMOVED internal loading
    if not settings:
        # Handle case where settings might not be passed (shouldn't happen via main_routes)
        raise HTTPException(status_code=500, detail="Settings dictionary not provided to search_freshrss.")
        
    freshrss_settings = settings.get('personalSources', {}).get('freshrss')

    if not freshrss_settings:
        raise HTTPException(status_code=400, detail="FreshRSS source is not configured.")

    base_url = freshrss_settings.get('base_url')
    username = freshrss_settings.get('username')
    api_password = freshrss_settings.get('api_password')

    if not all([base_url, username, api_password]):
        raise HTTPException(status_code=400, detail="FreshRSS configuration (base_url, username, api_password) is missing in settings.")

    # --- DEFINE BASE API URL *INSIDE* THE FUNCTION ---
    freshrss_base_api_url = f"{base_url.rstrip('/')}/api/greader.php/reader/api/0"
    # --- END DEFINE ---

    # --- GET AUTH TOKEN *INSIDE* THE FUNCTION ---
    auth_token = get_freshrss_token(username, api_password)
    headers = {'Authorization': auth_token}
    # --- END GET AUTH TOKEN ---
    
    search_url = f"{freshrss_base_api_url}/stream/contents/search/items/ids"
    params = {'q': query, 'output': 'json'}
    
    all_item_details = []
    errors = []

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
            # 1. Search for item IDs
            print(f"Querying FreshRSS search IDs: {search_url} with query: {query}")
            response_ids = await client.get(search_url, headers=headers, params=params)
            response_ids.raise_for_status()
            search_data = response_ids.json()
            
            item_ids = [item['id'] for item in search_data.get('results', [])]
            print(f"Found {len(item_ids)} matching item IDs.")

            if not item_ids:
                return FreshRSSSearchResponseModel(results=[], query=query, total_results=0, errors=errors)
            
            # --- TODO: Implement pagination/limit using item_ids, pageno, results_per_page --- START
            # Currently fetches all found IDs. Need to slice item_ids based on pageno and results_per_page
            # Example (Needs testing with API continuation if necessary):
            total_found_ids = len(item_ids)
            start_index = (pageno - 1) * results_per_page
            end_index = start_index + results_per_page
            ids_to_fetch = item_ids[start_index:end_index]
            
            if not ids_to_fetch:
                # Page number is out of range
                return FreshRSSSearchResponseModel(results=[], query=query, total_results=total_found_ids, errors=errors)
            # --- TODO: Implement pagination/limit using item_ids, pageno, results_per_page --- END

            # 2. Fetch full content for found item IDs (Use ids_to_fetch)
            content_url = f"{freshrss_base_api_url}/stream/items/contents"
            # The API expects multiple 'i' parameters for item IDs
            content_params = [('i', item_id) for item_id in ids_to_fetch] # <<< MODIFIED to use sliced list
            content_params.append(('output', 'json'))

            print(f"Fetching content for {len(ids_to_fetch)} items from: {content_url}") # <<< MODIFIED log message
            # Use POST for potentially long list of IDs
            response_content = await client.post(content_url, headers=headers, data=content_params) 
            response_content.raise_for_status()
            content_data = response_content.json()

            # Validate and map items
            raw_items = content_data.get('items', [])
            validated_items = []
            for item_data in raw_items:
                try:
                    # Add necessary fields if missing before validation
                    item_data['source'] = 'freshrss'
                    item_data['result_type'] = 'freshrss'
                    # Handle potential category format differences if necessary here
                    validated_item = FreshRSSResultItemModel.model_validate(item_data)
                    validated_items.append(validated_item)
                except Exception as e:
                    errors.append(f"Error validating item {item_data.get('id', 'N/A')}: {e}")
                    print(f"Validation Error for item: {item_data.get('id', 'N/A')}, Error: {e}")


            print(f"Successfully fetched and validated details for {len(validated_items)} items.")
            all_item_details = validated_items
            
    except httpx.TimeoutException:
        errors.append("Request to FreshRSS API timed out.")
        raise HTTPException(status_code=504, detail="Request to FreshRSS API timed out.")
    except httpx.RequestError as e:
        errors.append(f"HTTP Request error connecting to FreshRSS: {e}")
        raise HTTPException(status_code=502, detail=f"Could not connect to FreshRSS API: {e}")
    except httpx.HTTPStatusError as e:
         errors.append(f"FreshRSS API request failed: {e.response.status_code} - {e.response.text}")
         raise HTTPException(status_code=e.response.status_code, detail=f"FreshRSS API error: {e.response.text}")
    except json.JSONDecodeError:
        errors.append("Failed to decode JSON response from FreshRSS API.")
        raise HTTPException(status_code=500, detail="Invalid JSON response from FreshRSS API.")
    except Exception as e:
        errors.append(f"An unexpected error occurred: {e}")
        # Log the full error for debugging
        print(f"Unexpected error during FreshRSS search: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    return FreshRSSSearchResponseModel(
        results=all_item_details, 
        query=query, 
        total_results=total_found_ids, # <<< MODIFIED to return total found, not just fetched page count
        errors=errors if errors else None
    ) 
