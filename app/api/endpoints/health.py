"""
Health check endpoints for monitoring application status.
"""

import logging
import platform
import os
import sys
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config.settings import get_settings
from app.api.deps import get_request_id
from app.core.vector_store.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)
settings = get_settings()

# Start time for uptime calculation
START_TIME = time.time()

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    environment: str
    uptime_seconds: float
    components: Dict[str, Any]

router = APIRouter(
    prefix="/health",
    tags=["health"]
)

@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the application."
)
async def health_check(
    request_id: str = Depends(get_request_id)
):
    """
    Health check endpoint.
    
    Args:
        request_id: Request ID
        
    Returns:
        Health status of the application
    """
    logger.debug(f"Health check request (request_id={request_id})")
    
    # Calculate uptime
    uptime_seconds = time.time() - START_TIME
    
    # Check vector store
    vector_store_status = "healthy"
    vector_store_details = {}
    
    try:
        vector_store = get_chroma_store()
        vector_store_stats = vector_store.get_collection_stats()
        vector_store_details = {
            "collection_name": vector_store_stats.get("collection_name"),
            "document_count": vector_store_stats.get("document_count"),
            "persist_directory": vector_store_stats.get("persist_directory")
        }
    except Exception as e:
        logger.error(f"Error checking vector store health: {e}")
        vector_store_status = "unhealthy"
        vector_store_details = {"error": str(e)}
    
    # System info
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "processors": os.cpu_count()
    }
    
    # Compile health status
    health = HealthResponse(
        status="healthy",  # Overall status
        version=settings.version,
        environment=settings.environment,
        uptime_seconds=uptime_seconds,
        components={
            "vector_store": {
                "status": vector_store_status,
                "details": vector_store_details
            },
            "system": system_info
        }
    )
    
    # If any component is unhealthy, set overall status to unhealthy
    if vector_store_status == "unhealthy":
        health.status = "unhealthy"
    
    return health

@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the application is ready to receive traffic."
)
async def readiness_check(
    request_id: str = Depends(get_request_id)
):
    """
    Readiness check endpoint.
    
    Args:
        request_id: Request ID
        
    Returns:
        Simple OK message if the application is ready
    """
    logger.debug(f"Readiness check request (request_id={request_id})")
    
    # Check if vector store is available
    try:
        vector_store = get_chroma_store()
        vector_store_stats = vector_store.get_collection_stats()
    except Exception as e:
        logger.error(f"Vector store not ready: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application not ready: Vector store unavailable"
        )
    
    return {"status": "ok", "message": "Application is ready"}

@router.get(
    "/live",
    summary="Liveness check",
    description="Check if the application is alive."
)
async def liveness_check(
    request_id: str = Depends(get_request_id)
):
    """
    Liveness check endpoint.
    
    Args:
        request_id: Request ID
        
    Returns:
        Simple OK message if the application is alive
    """
    logger.debug(f"Liveness check request (request_id={request_id})")
    
    return {"status": "ok", "message": "Application is alive"}