"""
API endpoints for business term classification.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from starlette.concurrency import run_in_threadpool

from app.config.settings import get_settings
from app.api.deps import (
    get_api_key, 
    get_request_id, 
    get_classification_service_dep,
    get_pbt_manager_dep,
    check_rate_limit
)
from app.core.services.classification import ClassificationService
from app.core.services.pbt_manager import PBTManager
from app.core.models.pbt import (
    PBT, PBTClassificationRequest, PBTClassificationResponse,
    BatchPBTClassificationRequest, BatchPBTClassificationResponse,
    PBTLoadRequest, PBTLoadResponse, PBTStatistics
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/classification",
    tags=["classification"]
    # No API key dependency required
)

@router.post(
    "/classify",
    response_model=PBTClassificationResponse,
    summary="Classify a business term",
    description="Classify a business term against the Preferred Business Terms (PBT) database."
)
async def classify_term(
    request: PBTClassificationRequest,
    request_id: str = Depends(get_request_id),
    classification_service: ClassificationService = Depends(get_classification_service_dep),
    _: Any = Depends(lambda: check_rate_limit(endpoint_type="classification"))
):
    """
    Classify a business term against the PBT database.
    
    Args:
        request: Classification request with name and description
        request_id: Request ID
        classification_service: Classification service
        
    Returns:
        Classification response with best match and confidence
    """
    logger.info(f"Classification request: {request.name} (method={request.method}, request_id={request_id})")
    
    # Validate method
    if request.method not in settings.classification.available_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid classification method. Available methods: {settings.classification.available_methods}"
        )
    
    # Classify
    result = await classification_service.classify(
        name=request.name,
        description=request.description,
        method=request.method,
        include_broader_terms=request.include_broader_terms,
        top_n=request.top_n
    )
    
    # Add request ID
    result.request_id = request_id
    
    return result

@router.post(
    "/batch",
    response_model=BatchPBTClassificationResponse,
    summary="Classify multiple business terms",
    description="Classify multiple business terms in a single request."
)
async def batch_classify(
    request: BatchPBTClassificationRequest,
    request_id: str = Depends(get_request_id),
    classification_service: ClassificationService = Depends(get_classification_service_dep),
    _: Any = Depends(lambda: check_rate_limit(endpoint_type="classification"))
):
    """
    Batch classify multiple business terms.
    
    Args:
        request: Batch classification request with multiple items
        request_id: Request ID
        classification_service: Classification service
        
    Returns:
        Batch classification response with results for each item
    """
    # Check batch size limit
    if len(request.items) > settings.rate_limit.batch_size_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds limit of {settings.rate_limit.batch_size_limit} items"
        )
    
    logger.info(f"Batch classification request with {len(request.items)} items (method={request.method}, request_id={request_id})")
    
    # Validate method
    if request.method not in settings.classification.available_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid classification method. Available methods: {settings.classification.available_methods}"
        )
    
    # Batch classify
    results = await classification_service.batch_classify(
        items=[item.dict() for item in request.items],
        method=request.method
    )
    
    # Count successes and failures
    success_count = sum(1 for result in results if result.status == "success")
    
    # Create response
    response = BatchPBTClassificationResponse(
        status="success",
        items=results,
        request_id=request_id,
        total_processed=len(results),
        total_success=success_count,
        total_failure=len(results) - success_count
    )
    
    return response

@router.post(
    "/load-pbt",
    response_model=PBTLoadResponse,
    summary="Load PBT data",
    description="Load Preferred Business Terms (PBT) data from a CSV file."
)
async def load_pbt_data(
    request: PBTLoadRequest,
    request_id: str = Depends(get_request_id),
    pbt_manager: PBTManager = Depends(get_pbt_manager_dep)
):
    """
    Load PBT data from a CSV file.
    
    Args:
        request: Load request with CSV path
        request_id: Request ID
        pbt_manager: PBT manager
        
    Returns:
        Load response with status and count
    """
    logger.info(f"Loading PBT data from {request.csv_path} (reload={request.reload}, request_id={request_id})")
    
    # Load the data
    result = await pbt_manager.load_csv(request.csv_path, request.reload)
    
    # Add request ID
    response = PBTLoadResponse(
        status=result["status"],
        message=result["message"],
        total_loaded=result["total_loaded"],
        request_id=request_id
    )
    
    return response

@router.get(
    "/pbt/{pbt_id}",
    response_model=PBT,
    summary="Get PBT by ID",
    description="Get a Preferred Business Term (PBT) by its ID."
)
async def get_pbt_by_id(
    pbt_id: str,
    request_id: str = Depends(get_request_id),
    pbt_manager: PBTManager = Depends(get_pbt_manager_dep)
):
    """
    Get a PBT by its ID.
    
    Args:
        pbt_id: PBT ID
        request_id: Request ID
        pbt_manager: PBT manager
        
    Returns:
        PBT object
    """
    logger.info(f"Getting PBT by ID: {pbt_id} (request_id={request_id})")
    
    # Get the PBT
    pbt = await pbt_manager.get_pbt_by_id(pbt_id)
    
    if not pbt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PBT with ID {pbt_id} not found"
        )
    
    return pbt

@router.get(
    "/statistics",
    response_model=PBTStatistics,
    summary="Get PBT statistics",
    description="Get statistics about the loaded PBT data."
)
async def get_pbt_statistics(
    request_id: str = Depends(get_request_id),
    pbt_manager: PBTManager = Depends(get_pbt_manager_dep)
):
    """
    Get statistics about the loaded PBT data.
    
    Args:
        request_id: Request ID
        pbt_manager: PBT manager
        
    Returns:
        PBT statistics
    """
    logger.info(f"Getting PBT statistics (request_id={request_id})")
    
    # Get statistics
    stats = await pbt_manager.get_statistics()
    
    return stats