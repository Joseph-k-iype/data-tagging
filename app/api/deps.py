"""
Dependencies for FastAPI endpoints.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from app.config.settings import get_settings
from app.core.auth.auth_helper import verify_api_key, get_current_user
from app.core.services.classification import get_classification_service
from app.core.services.pbt_manager import get_pbt_manager

logger = logging.getLogger(__name__)

# API key header
settings = get_settings()
api_key_header = APIKeyHeader(name=settings.security.api_key_header, auto_error=False)

# Request ID header
X_REQUEST_ID = "X-Request-ID"

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Get API key without validation - API key is completely optional.
    This is for development purposes only and should not be used in production.
    """
    return api_key

async def get_request_id(request_id: Optional[str] = Header(None, alias=X_REQUEST_ID)):
    """Get or generate request ID."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    return request_id

async def get_classification_service_dep():
    """Get classification service dependency."""
    return get_classification_service()

async def get_pbt_manager_dep():
    """Get PBT manager dependency."""
    return get_pbt_manager()

# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
        self.window_size = 60  # 1 minute window
    
    def is_rate_limited(self, client_id: str, limit: int) -> bool:
        """
        Check if the client has exceeded the rate limit.
        
        Args:
            client_id: Client identifier (API key or IP)
            limit: Request limit per minute
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        now = time.time()
        client_requests = self.requests.get(client_id, [])
        
        # Remove expired timestamps
        client_requests = [ts for ts in client_requests if now - ts < self.window_size]
        
        # Check if limit is exceeded
        if len(client_requests) >= limit:
            return True
        
        # Update requests
        client_requests.append(now)
        self.requests[client_id] = client_requests
        
        # Cleanup old entries if too many clients
        if len(self.requests) > 1000:
            self._cleanup()
        
        return False
    
    def _cleanup(self):
        """Clean up expired entries."""
        now = time.time()
        expired_clients = []
        
        for client_id, timestamps in self.requests.items():
            if all(now - ts >= self.window_size for ts in timestamps):
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.requests[client_id]

# Create rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
    endpoint_type: str = "default"
):
    """
    Check rate limit for the request.
    
    Args:
        request: FastAPI request
        api_key: API key
        endpoint_type: Endpoint type for specific limits
    """
    if not settings.rate_limit.enabled:
        return
    
    # Use API key or client IP as identifier
    client_id = api_key if api_key else request.client.host
    
    # Get limit based on endpoint type
    if endpoint_type == "classification":
        limit = settings.rate_limit.classification_limit
    else:
        limit = settings.rate_limit.default_limit
    
    # Check rate limit
    if rate_limiter.is_rate_limited(client_id, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} requests per minute."
        )

# Request timing middleware
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    
    # Generate request ID if not present
    if X_REQUEST_ID not in request.headers:
        request_id = str(uuid.uuid4())
        request.scope["headers"].append((X_REQUEST_ID.lower().encode(), request_id.encode()))
    else:
        request_id = request.headers[X_REQUEST_ID]
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers[X_REQUEST_ID] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request
    method = request.method
    path = request.url.path
    status_code = response.status_code
    
    log_level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(log_level, f"{method} {path} {status_code} {process_time:.4f}s {request_id}")
    
    return response