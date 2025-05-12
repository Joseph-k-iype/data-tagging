"""
Application settings for the AI Tagging Service.
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List, Set
from pydantic import BaseModel, Field, validator
from app.config.environment import get_os_env

logger = logging.getLogger(__name__)

class SecuritySettings(BaseModel):
    """Security settings."""
    secret_key: str = Field(..., description="Secret key for JWT token encryption")
    algorithm: str = Field("HS256", description="Algorithm for JWT token encryption")
    access_token_expire_minutes: int = Field(30, description="Access token expiration time in minutes")
    refresh_token_expire_days: int = Field(7, description="Refresh token expiration time in days")
    cors_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")
    api_key_header: str = Field("X-API-Key", description="Header name for API key")
    use_api_key: bool = Field(False, description="Whether to use API key authentication")

class VectorDBSettings(BaseModel):
    """Vector database settings."""
    type: str = Field("chroma", description="Vector database type")
    persist_dir: str = Field("./data/chroma_db", description="Directory for vector database persistence")
    collection_name: str = Field("business_terms", description="Name of the collection in the vector database")
    distance_metric: str = Field("cosine", description="Distance metric for similarity search")
    embedding_dimension: int = Field(3072, description="Dimension of the embedding vectors")

class AzureSettings(BaseModel):
    """Azure settings."""
    tenant_id: str = Field(..., description="Azure tenant ID")
    client_id: str = Field(..., description="Azure client ID")
    client_secret: str = Field(..., description="Azure client secret")
    azure_endpoint: str = Field(..., description="Azure OpenAI endpoint")
    api_version: str = Field("2023-05-15", description="Azure OpenAI API version")
    model_name: str = Field("gpt-4o-mini", description="Azure OpenAI model name")
    embedding_model: str = Field("text-embedding-3-large", description="Azure OpenAI embedding model")
    embedding_deployment_name: str = Field("text-embedding-3-large", description="Azure OpenAI embedding deployment name")
    token_caching_enabled: bool = Field(True, description="Whether to cache Azure AD tokens")
    token_refresh_interval: int = Field(300, description="Token refresh interval in seconds")
    token_validation_threshold: int = Field(600, description="Token validation threshold in seconds")

class ClassificationSettings(BaseModel):
    """Classification settings."""
    default_method: str = Field("agent", description="Default classification method")
    available_methods: Set[str] = Field({"embeddings", "llm", "agent"}, description="Available classification methods")
    cache_enabled: bool = Field(True, description="Whether to cache classification results")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    similarity_threshold: float = Field(0.5, description="Similarity threshold for embedding search")
    top_n_results: int = Field(5, description="Number of top results to return")
    include_broader_terms: bool = Field(True, description="Whether to include broader terms in results")
    confidence_threshold: float = Field(70.0, description="Confidence threshold for automatic approval")

class RateLimitSettings(BaseModel):
    """Rate limit settings."""
    enabled: bool = Field(True, description="Whether to enable rate limiting")
    default_limit: int = Field(100, description="Default rate limit per minute")
    classification_limit: int = Field(50, description="Rate limit for classification endpoints per minute")
    batch_size_limit: int = Field(100, description="Maximum batch size for batch classification")

class Settings(BaseModel):
    """Application settings."""
    app_name: str = Field("AI Tagging Service", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("production", description="Environment (development, testing, production)")
    log_level: str = Field("INFO", description="Log level")
    proxy_enabled: bool = Field(False, description="Whether to use a proxy for external requests")
    api_prefix: str = Field("/api/v1", description="API prefix")
    
    # Component settings
    security: SecuritySettings
    vector_db: VectorDBSettings
    azure: AzureSettings
    classification: ClassificationSettings
    rate_limit: RateLimitSettings
    
    # Data paths
    data_dir: str = Field("./data", description="Data directory")
    pbt_csv_path: str = Field("./data/pbt_data.csv", description="Path to PBT CSV file")
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = {"development", "testing", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Application settings
    """
    env = get_os_env()
    
    # Security settings
    security_settings = SecuritySettings(
        secret_key=env.get("SECRET_KEY", "supersecretkey"),
        algorithm=env.get("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(env.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
        refresh_token_expire_days=int(env.get("REFRESH_TOKEN_EXPIRE_DAYS", "7")),
        cors_origins=env.get("CORS_ORIGINS", "*").split(","),
        api_key_header=env.get("API_KEY_HEADER", "X-API-Key"),
        use_api_key=str_to_bool(env.get("USE_API_KEY", "True"))
    )
    
    # Vector DB settings
    vector_db_settings = VectorDBSettings(
        type=env.get("VECTOR_DB_TYPE", "chroma"),
        persist_dir=env.get("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        collection_name=env.get("CHROMA_COLLECTION", "business_terms"),
        distance_metric=env.get("DISTANCE_METRIC", "cosine"),
        embedding_dimension=int(env.get("EMBEDDING_DIMENSION", "3072"))
    )
    
    # Azure settings
    azure_settings = AzureSettings(
        tenant_id=env.get("AZURE_TENANT_ID", ""),
        client_id=env.get("AZURE_CLIENT_ID", ""),
        client_secret=env.get("AZURE_CLIENT_SECRET", ""),
        azure_endpoint=env.get("AZURE_ENDPOINT", ""),
        api_version=env.get("API_VERSION", "2023-05-15"),
        model_name=env.get("MODEL_NAME", "gpt-4o-mini"),
        embedding_model=env.get("EMBEDDING_MODEL", "text-embedding-3-large"),
        embedding_deployment_name=env.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        token_caching_enabled=str_to_bool(env.get("TOKEN_CACHING_ENABLED", "True")),
        token_refresh_interval=int(env.get("TOKEN_REFRESH_INTERVAL", "300")),
        token_validation_threshold=int(env.get("TOKEN_VALIDATION_THRESHOLD", "600"))
    )
    
    # Classification settings
    classification_settings = ClassificationSettings(
        default_method=env.get("DEFAULT_CLASSIFICATION_METHOD", "agent"),
        available_methods=set(env.get("AVAILABLE_CLASSIFICATION_METHODS", "embeddings,llm,agent").split(",")),
        cache_enabled=str_to_bool(env.get("CLASSIFICATION_CACHE_ENABLED", "True")),
        cache_ttl=int(env.get("CLASSIFICATION_CACHE_TTL", "3600")),
        similarity_threshold=float(env.get("SIMILARITY_THRESHOLD", "0.5")),
        top_n_results=int(env.get("TOP_N_RESULTS", "5")),
        include_broader_terms=str_to_bool(env.get("INCLUDE_BROADER_TERMS", "True")),
        confidence_threshold=float(env.get("CONFIDENCE_THRESHOLD", "70.0"))
    )
    
    # Rate limit settings
    rate_limit_settings = RateLimitSettings(
        enabled=str_to_bool(env.get("RATE_LIMIT_ENABLED", "True")),
        default_limit=int(env.get("RATE_LIMIT_DEFAULT", "100")),
        classification_limit=int(env.get("RATE_LIMIT_CLASSIFICATION", "50")),
        batch_size_limit=int(env.get("BATCH_SIZE_LIMIT", "100"))
    )
    
    # Main settings
    settings = Settings(
        app_name=env.get("APP_NAME", "AI Tagging Service"),
        version=env.get("APP_VERSION", "1.0.0"),
        debug=str_to_bool(env.get("DEBUG", "False")),
        environment=env.get("ENVIRONMENT", "production"),
        log_level=env.get("LOG_LEVEL", "INFO"),
        proxy_enabled=str_to_bool(env.get("PROXY_ENABLED", "False")),
        api_prefix=env.get("API_PREFIX", "/api/v1"),
        
        # Component settings
        security=security_settings,
        vector_db=vector_db_settings,
        azure=azure_settings,
        classification=classification_settings,
        rate_limit=rate_limit_settings,
        
        # Data paths
        data_dir=env.get("DATA_DIR", "./data"),
        pbt_csv_path=env.get("PBT_CSV_PATH", "./data/pbt_data.csv")
    )
    
    return settings

def str_to_bool(s: str) -> bool:
    """
    Convert a string to a boolean value.
    
    Args:
        s: String to convert
        
    Returns:
        bool: Converted boolean value
    """
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    return False