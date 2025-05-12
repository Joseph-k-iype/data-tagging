"""
Main application entry point for the AI Tagging Service.
"""

import os
import logging
import asyncio
from typing import List

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from starlette_exporter import PrometheusMiddleware, handle_metrics

from app.config.settings import get_settings
from app.config.logging_config import configure_logging
from app.config.environment import get_os_env
from app.api.deps import add_process_time_header
from app.api.endpoints import classification, health
from app.core.auth.auth_helper import start_token_refresh_service
from app.core.services.pbt_manager import get_pbt_manager

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()
env = get_os_env()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=f"{settings.app_name} API",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(PrometheusMiddleware)

# Add request timing middleware
@app.middleware("http")
async def process_time_middleware(request: Request, call_next):
    return await add_process_time_header(request, call_next)

# Add metrics endpoint
app.add_route("/metrics", handle_metrics)

# Include routers
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(classification.router, prefix=settings.api_prefix)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.version,
        description=f"{settings.app_name} API",
        routes=app.routes,
    )
    
    # Add API key security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": settings.security.api_key_header
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.version} in {settings.environment} mode")
    
    # Start background tasks
    if settings.azure.token_caching_enabled:
        start_token_refresh_service(settings.azure.token_refresh_interval)
        logger.info(f"Token refresh service started with interval {settings.azure.token_refresh_interval}s")
    
    # Create data directory if it doesn't exist
    os.makedirs(settings.data_dir, exist_ok=True)
    
    # Load PBT data if CSV file exists
    if os.path.exists(settings.pbt_csv_path):
        logger.info(f"Loading PBT data from {settings.pbt_csv_path}")
        pbt_manager = get_pbt_manager()
        load_result = await pbt_manager.load_csv(settings.pbt_csv_path)
        logger.info(f"PBT data loading result: {load_result['status']} - {load_result['message']}")
    else:
        logger.warning(f"PBT data file not found: {settings.pbt_csv_path}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.app_name}")


# Run application with uvicorn when this file is executed directly
if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    logger.info(f"Running application on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )