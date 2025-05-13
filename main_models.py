# main_models.py
from pydantic import BaseModel, Field, HttpUrl, RedisDsn, field_validator, ValidationError
from typing import Any, Dict, List, Optional, Union, Literal
import datetime

# --- Settings Models ---

class GeneralSettings(BaseModel):
    instanceName: Optional[str] = "SlySearch"
    resultsPerPage: Optional[Union[str, int]] = "10"
    safeSearch: Optional[Union[str, int]] = "0"
    defaultLanguage: Optional[str] = "auto"
    openNewTab: Optional[bool] = True
    infiniteScroll: Optional[bool] = True
    ragEnabled: Optional[bool] = False
    autocomplete: Optional[bool] = True
    autocompleteMin: Optional[Union[str, int]] = "4"
    faviconResolver: Optional[str] = "off"
    banTime: Optional[Union[str, int]] = "5"
    maxBanTime: Optional[Union[str, int]] = "120"
    searchOnCategory: Optional[bool] = True

class EngineDetail(BaseModel):
    id: str
    name: str
    enabled: Optional[bool] = True
    weight: Optional[float] = 1.0
    shortcut: Optional[str] = None
    categories: Optional[List[str]] = None
    timeout: Optional[Union[float, int, str]] = None
    potential_bias: Optional[List[str]] = None
    description: Optional[str] = None

class EngineLoadoutItem(BaseModel):
    id: str
    name: str
    config: Optional[List[EngineDetail]] = Field(default_factory=list)

class EngineSettings(BaseModel):
    loadouts: Optional[List[EngineLoadoutItem]] = Field(
        default_factory=lambda: [
            EngineLoadoutItem(id="starter", name="Starter", config=[])
        ]
    )
    activeLoadoutId: Optional[str] = 'starter'

class PrivacySettings(BaseModel):
    proxyImages: Optional[bool] = True
    removeTrackers: Optional[bool] = True
    blockCookies: Optional[bool] = False
    queryInTitle: Optional[bool] = False
    method: Optional[str] = "POST"
    urlAnonymizer: Optional[bool] = False

class AppearanceSettings(BaseModel):
    resultsLayout: Optional[str] = "list"
    theme: Optional[str] = "cyberpunk"
    centerAlignment: Optional[bool] = False
    defaultLocale: Optional[str] = "auto"
    hotkeys: Optional[str] = "default"
    urlFormat: Optional[str] = "pretty"
    enableQuickLinks: Optional[bool] = True
    quickLinks: Optional[List[Dict[str, Any]]] = None

class AdvancedSettings(BaseModel):
    # Allowing string type for HttpUrl during validation, then casting
    instanceUrl: Union[HttpUrl, str, None] = Field(default=HttpUrl("http://localhost:8888"))
    requestTimeout: Optional[Union[str, int]] = "5"
    maxRequestTimeout: Optional[Union[str, int]] = "10"
    formats: Optional[List[str]] = Field(default_factory=lambda: ["json", "html"])
    headlessMode: Optional[bool] = False
    enableResultProxy: Optional[bool] = False
    resultProxyUrl: Optional[str] = ""
    resultProxyKey: Optional[str] = ""
    poolConnections: Optional[Union[str, int]] = "100"
    poolMaxsize: Optional[Union[str, int]] = "20"
    enableHttp2: Optional[bool] = True
    customCss: Optional[str] = ""
    debugMode: Optional[bool] = False
    redisUrl: Optional[Union[RedisDsn, str, None]] = None # Allow string for flexibility
    limiter: Optional[bool] = False
    publicInstance: Optional[bool] = False

    @field_validator('instanceUrl', mode='before')
    @classmethod
    def validate_instance_url(cls, v):
        if isinstance(v, str):
            try:
                # Attempt to parse as HttpUrl, will raise ValidationError if invalid
                return HttpUrl(v)
            except ValidationError:
                # Re-raise as ValueError for FastAPI to catch as a 422 error
                raise ValueError(f"Invalid URL format for instanceUrl: {v}")
        return v # Return as is if already HttpUrl or None

    @field_validator('redisUrl', mode='before')
    @classmethod
    def validate_redis_url(cls, v):
        if isinstance(v, str):
             # Basic check, Pydantic's RedisDsn will do more thorough validation
             if not v.startswith(('redis://', 'rediss://')):
                 raise ValueError("Redis URL must start with redis:// or rediss://")
        # Allow None or already valid RedisDsn
        return v


# Personal Sources Models
class SourceListItem(BaseModel):
    id: str
    label: str
    icon: str
    color: str
    gradient: str
    type: Optional[str] = None # e.g., 'music', 'obsidian'

class SourceLoadoutItem(BaseModel):
    id: str
    name: str
    config: Optional[List[SourceListItem]] = Field(default_factory=list)

class ObsidianSourceConfig(BaseModel):
    path: Optional[str] = None
    vaultName: Optional[str] = None
    useLocalPlugin: Optional[bool] = False
    apiPort: Optional[Union[str, int]] = None
    pluginApiKey: Optional[str] = None
    resultsPerPage: Optional[int] = 10

class LocalFilesSourceConfig(BaseModel):
    path: Optional[str] = None
    fileTypes: Optional[str] = "md,txt,pdf"
    useIndexer: Optional[bool] = False
    resultsPerPage: Optional[int] = 10

class AISourceConfig(BaseModel):
    provider: Optional[str] = "openai" 
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None 
    model: Optional[str] = "gpt-4o" 
    temperature: Optional[Union[str, float, int]] = "0.7"
    maxTokens: Optional[Union[str, int]] = "1000"
    resultsPerPage: Optional[int] = 1

class YouTubeSourceConfig(BaseModel):
    path: Optional[str] = None
    download_base_path: Optional[str] = None
    apiKey: Optional[str] = None
    includeChannels: Optional[bool] = True
    includePlaylists: Optional[bool] = True
    resultsPerPage: Optional[int] = 10
    resultsColumns: Optional[int] = 4

class MusicSourceConfig(BaseModel):
    path: Optional[str] = None
    resultsPerPage: Optional[int] = 10

class PhotosSourceConfig(BaseModel):
    path: Optional[str] = None
    resultsPerPage: Optional[int] = 10
    resultsColumns: Optional[int] = 4

# +++ Add FreshRSS Config Model +++
class FreshRSSSourceConfig(BaseModel):
    base_url: Optional[Union[HttpUrl, str, None]] = None 
    username: Optional[str] = None
    api_password: Optional[str] = None
    resultsPerPage: Optional[int] = 10
    openNewTab: Optional[bool] = True
    
    @field_validator('base_url', mode='before')
    @classmethod
    def validate_base_url(cls, v):
        if isinstance(v, str) and v.strip(): 
            try:
                return HttpUrl(v)
            except ValidationError:
                raise ValueError(f"Invalid URL format for FreshRSS base_url: {v}")
        return v 

# +++ Add Web Config Model (Assuming structure if needed, mirroring frontend schema) +++
class WebSourceConfig(BaseModel):
    resultsPerPage: Optional[int] = 10
    openNewTab: Optional[bool] = True
    searchOnCategory: Optional[bool] = True

class PersonalSourcesSettings(BaseModel):
    sources: List[SourceListItem] = [
        SourceListItem(id="web", label="Web", icon="Zap", color="#176BEF", gradient="from-[#176BEF]/70 to-[#FF3E30]/70", type="web"),
        SourceListItem(id="obsidian", label="Obsidian", icon="Brain", color="#7E6AD7", gradient="from-[#7E6AD7]/70 to-[#9C87E0]/70", type="obsidian"),
        SourceListItem(id="localFiles", label="Files", icon="FileText", color="#F7B529", gradient="from-[#FF3E30]/70 to-[#F7B529]/70", type="localFiles"),
        SourceListItem(id="ai", label="AI", icon="Bot", color="#10B981", gradient="from-[#10B981]/70 to-[#059669]/70", type="ai"),
        SourceListItem(id="youtube", label="YouTube", icon="Youtube", color="#FF0000", gradient="from-[#FF0000]/70 to-[#CC0000]/70", type="youtube"),
        SourceListItem(id="music", label="Music", icon="Library", color="#FF7700", gradient="from-[#FF7700]/70 to-[#FF3300]/70", type="music"),
        SourceListItem(id="photos", label="Photos", icon="Image", color="#3498DB", gradient="from-[#3498DB]/70 to-[#2980B9]/70", type="photos"),
        SourceListItem(id="freshrss", label="FreshRSS", icon="Rss", color="#FFA500", gradient="from-[#FFA500]/70 to-[#FF8C00]/70", type="freshrss"),
    ]
    loadouts: Optional[List[SourceLoadoutItem]] = Field(default_factory=list)
    obsidian: Optional[ObsidianSourceConfig] = Field(default_factory=lambda: ObsidianSourceConfig(resultsPerPage=10))
    localFiles: Optional[LocalFilesSourceConfig] = Field(default_factory=lambda: LocalFilesSourceConfig(resultsPerPage=10))
    ai: Optional[AISourceConfig] = Field(default_factory=lambda: AISourceConfig(resultsPerPage=1))
    youtube: Optional[YouTubeSourceConfig] = Field(default_factory=lambda: YouTubeSourceConfig(resultsPerPage=10, resultsColumns=4))
    music: Optional[MusicSourceConfig] = Field(default_factory=lambda: MusicSourceConfig(resultsPerPage=10))
    photos: Optional[PhotosSourceConfig] = Field(default_factory=lambda: PhotosSourceConfig(resultsPerPage=10, resultsColumns=4))
    freshrss: Optional[FreshRSSSourceConfig] = Field(default_factory=lambda: FreshRSSSourceConfig(resultsPerPage=10, openNewTab=True))

# Main Settings Data Model
class SettingsData(BaseModel):
    general: Optional[GeneralSettings] = Field(default_factory=GeneralSettings)
    engines: Optional[EngineSettings] = Field(default_factory=EngineSettings)
    privacy: Optional[PrivacySettings] = Field(default_factory=PrivacySettings)
    appearance: Optional[AppearanceSettings] = Field(default_factory=AppearanceSettings)
    advanced: Optional[AdvancedSettings] = Field(default_factory=AdvancedSettings)
    personalSources: Optional[PersonalSourcesSettings] = Field(default_factory=PersonalSourcesSettings)

# Vault Models
class VaultItem(BaseModel):
    name: str
    path: str
    type: Literal['file', 'directory']
    extension: Optional[str] = None
    modified_time: Optional[datetime.datetime] = None
    size: Optional[int] = None

class VaultBrowseResponse(BaseModel):
    path: str
    items: List[VaultItem]
    error: Optional[str] = None

class VaultCheckResponse(BaseModel):
    valid: bool
    error: Optional[str] = None

# Search Response Models
class ObsidianResultItem(BaseModel):
    title: str
    url: str # Relative path within vault
    snippet: Optional[str] = None
    source: Literal['obsidian'] = 'obsidian'

class ObsidianSearchResponse(BaseModel):
    query: str
    results: List[ObsidianResultItem]
    source: Literal['obsidian'] = 'obsidian'
    errors: List[str] = Field(default_factory=list)
    total_results: Optional[int] = None

class YouTubeResultItem(BaseModel):
    title: str
    url: str # Original YouTube URL
    vid: Optional[str] = None
    thumbnail_url: Optional[str] = None # URL to serve the thumbnail via backend
    snippet: Optional[str] = None
    channel_name: Optional[str] = None
    file_path: Optional[str] = None # Local path if downloaded
    source: Literal['youtube'] = 'youtube'

class YouTubeSearchResponse(BaseModel):
    query: str
    results: List[YouTubeResultItem]
    total_results: Optional[int] = None
    source: Literal["youtube"] = "youtube"
    errors: List[str] = Field(default_factory=list)

class PhotoResultItem(BaseModel):
    filename: str
    relative_path: str # Path relative to photo base
    modified_time: Optional[datetime.datetime] = None
    source: Literal['photos'] = 'photos'

class PhotosSearchResponse(BaseModel):
    query: str
    results: List[PhotoResultItem]
    total_results: Optional[int] = None
    source: Literal["photos"] = "photos"
    errors: List[str] = Field(default_factory=list)

class MusicResultItem(BaseModel):
    filename: str
    relative_path: str # Path relative to music base
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    modified_time: Optional[datetime.datetime] = None
    source: Literal['music'] = 'music'

class MusicSearchResponse(BaseModel):
    query: str
    results: List[MusicResultItem]
    total_results: Optional[int] = None
    source: Literal["music"] = "music"
    errors: List[str] = Field(default_factory=list)

# Chat Models (from Phase 1, keeping them grouped)
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message to the chatbot.")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's generated response.")
