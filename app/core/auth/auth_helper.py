"""
Authentication helper functions for working with Azure AD tokens and API security.
"""

import os
import logging
import time
import threading
import requests
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Models for token handling
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []


class TokenCache:
    """Thread-safe token cache for storing and retrieving Azure tokens."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TokenCache, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the token cache."""
        if self._initialized:
            return
            
        with self._lock:
            if not self._initialized:
                self._tokens = {}  # Format: {cache_key: (token, expiry_time)}
                self._initialized = True
                logger.info("Token cache initialized")
    
    def get(self, tenant_id: str, client_id: str, scope: str) -> Optional[str]:
        """
        Get a token from the cache if it exists and is not expired.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            scope: OAuth scope
            
        Returns:
            Token if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(tenant_id, client_id, scope)
        
        with self._lock:
            if cache_key in self._tokens:
                token, expiry_time = self._tokens[cache_key]
                # Allow 5 minute buffer before expiration
                if time.time() < expiry_time - 300:
                    logger.debug(f"Token cache hit for {client_id[:8]}...")
                    return token
                else:
                    logger.debug(f"Token expired for {client_id[:8]}...")
                    # Remove expired token
                    del self._tokens[cache_key]
        
        return None
    
    def set(self, tenant_id: str, client_id: str, scope: str, token: str, expires_in: int = 3600) -> None:
        """
        Store a token in the cache.
        
        Args:
            tenant_id: Azure tenant ID
            client_id: Azure client ID
            scope: OAuth scope
            token: The token to store
            expires_in: Token expiration time in seconds
        """
        cache_key = self._get_cache_key(tenant_id, client_id, scope)
        expiry_time = time.time() + expires_in
        
        with self._lock:
            self._tokens[cache_key] = (token, expiry_time)
            
        logger.debug(f"Token cached for {client_id[:8]}... (expires in {expires_in}s)")
    
    def _get_cache_key(self, tenant_id: str, client_id: str, scope: str) -> str:
        """Generate a cache key."""
        return f"{tenant_id}:{client_id}:{scope}"
    
    def clear(self) -> None:
        """Clear all tokens from the cache."""
        with self._lock:
            self._tokens.clear()
        logger.info("Token cache cleared")
    
    def remove(self, tenant_id: str, client_id: str, scope: str) -> None:
        """Remove a specific token from the cache."""
        cache_key = self._get_cache_key(tenant_id, client_id, scope)
        
        with self._lock:
            if cache_key in self._tokens:
                del self._tokens[cache_key]
                logger.debug(f"Token removed from cache for {client_id[:8]}...")


# Initialize global token cache
token_cache = TokenCache()

def get_azure_token_cached(tenant_id: str, client_id: str, client_secret: str, 
                          scope: str = "https://cognitiveservices.azure.com/.default") -> Optional[str]:
    """
    Get an Azure AD token with caching support.
    This function first checks the cache before making a new token request.
    
    Args:
        tenant_id: Azure tenant ID
        client_id: Azure client ID
        client_secret: Azure client secret
        scope: OAuth scope to request
        
    Returns:
        Access token if successful, None otherwise
    """
    # Check cache first
    token = token_cache.get(tenant_id, client_id, scope)
    if token:
        return token
    
    # Cache miss - get new token
    try:
        logger.info(f"Token cache miss for {client_id[:8]}... - fetching new token")
        
        # OAuth2 token endpoint
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        
        # Request body
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope,
            "grant_type": "client_credentials"
        }
        
        # Make the request
        response = requests.post(
            token_url, 
            data=data,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            token_data = response.json()
            if "access_token" in token_data:
                token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour
                
                # Cache the token
                token_cache.set(tenant_id, client_id, scope, token, expires_in)
                
                logger.info(f"New token acquired and cached (expires in {expires_in}s)")
                return token
            else:
                logger.error("Token response did not contain access_token")
                return None
        else:
            logger.error(f"Token request failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting Azure token: {e}")
        return None


def refresh_token_if_needed(tenant_id: str, client_id: str, client_secret: str, 
                           scope: str = "https://cognitiveservices.azure.com/.default",
                           min_validity_seconds: int = 600) -> bool:
    """
    Check if a token is about to expire and refresh it if needed.
    
    Args:
        tenant_id: Azure tenant ID
        client_id: Azure client ID
        client_secret: Azure client secret
        scope: OAuth scope
        min_validity_seconds: Minimum seconds of validity required
        
    Returns:
        True if token was refreshed or is valid, False on error
    """
    # Get token from cache to check expiry
    cache_key = token_cache._get_cache_key(tenant_id, client_id, scope)
    
    with token_cache._lock:
        if cache_key in token_cache._tokens:
            _, expiry_time = token_cache._tokens[cache_key]
            time_left = expiry_time - time.time()
            
            # If token expires soon, refresh it
            if time_left < min_validity_seconds:
                logger.info(f"Token for {client_id[:8]}... expires in {time_left:.0f}s, refreshing")
                
                # Remove old token
                del token_cache._tokens[cache_key]
    
    # Get a fresh token (will update cache)
    token = get_azure_token_cached(tenant_id, client_id, client_secret, scope)
    return token is not None


# Global refresh thread reference
_token_refresh_thread = None

def start_token_refresh_service(refresh_interval: int = 300) -> threading.Thread:
    """
    Start a background thread that refreshes tokens periodically.
    
    Args:
        refresh_interval: Interval between refresh checks in seconds
        
    Returns:
        The background thread
    """
    def _token_refresh_worker():
        settings = get_settings()
        
        tenant_id = settings.azure.tenant_id
        client_id = settings.azure.client_id
        client_secret = settings.azure.client_secret
        
        while True:
            try:
                # Refresh the token if it's going to expire soon
                refresh_token_if_needed(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            except Exception as e:
                logger.error(f"Error in token refresh worker: {e}")
            
            # Sleep for the specified interval
            time.sleep(refresh_interval)
    
    # Create and start the thread
    refresh_thread = threading.Thread(
        target=_token_refresh_worker,
        daemon=True,
        name="TokenRefreshThread"
    )
    refresh_thread.start()
    logger.info(f"Token refresh service started (interval: {refresh_interval}s)")
    
    return refresh_thread


# API Key security
settings = get_settings()
api_key_header = APIKeyHeader(name=settings.security.api_key_header, auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Hard-coded API keys for demo purposes - in production, store in a secure database
API_KEYS = {
    "dev": "dev-api-key-123",
    "test": "test-api-key-456",
    "admin": "admin-api-key-789"
}

# Generate a secret key if not provided
def generate_secret_key(length=32):
    """Generate a random secret key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

SECRET_KEY = settings.security.secret_key or generate_secret_key()
ALGORITHM = settings.security.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.security.access_token_expire_minutes


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    API key verification is disabled.
    This function always returns True for all API keys.
    """
    return True


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if token is None:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # In a real system, you would look up the user in a database
    # For this example, we'll just create a simple user object
    if token_data.username:
        user = {"username": token_data.username}
        return user
        
    raise credentials_exception