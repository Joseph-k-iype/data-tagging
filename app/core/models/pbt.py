"""
Preferred Business Term (PBT) models and related classes.
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator, UUID4


class MatchType(str, Enum):
    """Type of match between a query and a PBT."""
    SPECIFIC = "specific"
    BROADER = "broader"
    SYNONYM = "synonym"
    EXACT = "exact"


class PBT(BaseModel):
    """Preferred Business Term (PBT) model."""
    id: str
    name: str = Field(..., alias="PBT_NAME")
    definition: str = Field(..., alias="PBT_DEFINITION")
    cdm: Optional[str] = Field(None, alias="CDM")
    synonyms: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True  # Allow populating fields by alias or name


class SynonymMatch(BaseModel):
    """Match between a synonym and a PBT."""
    term_id: str
    term_name: str
    synonym: str
    score: float


class ConfidenceScore(BaseModel):
    """Confidence score for a match."""
    score: float = Field(..., ge=0, le=100)
    explanation: str


class MatchedPBT(BaseModel):
    """Matched PBT with additional information."""
    id: str
    name: str
    definition: str
    cdm: Optional[str] = None
    match_type: MatchType
    similarity_score: float
    synonym_match: Optional[bool] = False
    matched_synonym: Optional[str] = None


class PBTClassificationRequest(BaseModel):
    """Request model for PBT classification."""
    name: str = Field(..., description="Name of the business term to classify")
    description: str = Field(..., description="Description of the business term to classify")
    method: Optional[str] = Field("agent", description="Classification method to use (embeddings, llm, agent)")
    include_broader_terms: Optional[bool] = Field(True, description="Whether to include broader terms in the results")
    top_n: Optional[int] = Field(5, description="Number of results to return")


class BatchPBTClassificationRequest(BaseModel):
    """Request model for batch PBT classification."""
    items: List[PBTClassificationRequest] = Field(..., description="List of items to classify")
    method: Optional[str] = Field("agent", description="Classification method to use (embeddings, llm, agent)")


class PBTClassificationResponse(BaseModel):
    """Response model for PBT classification."""
    status: str
    best_match: Optional[MatchedPBT] = None
    specific_matches: List[MatchedPBT] = Field(default_factory=list)
    broader_matches: List[MatchedPBT] = Field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    agent_response: Optional[str] = None
    request_id: Optional[str] = None


class BatchPBTClassificationResponse(BaseModel):
    """Response model for batch PBT classification."""
    status: str
    items: List[PBTClassificationResponse] = Field(default_factory=list)
    request_id: Optional[str] = None
    total_processed: int
    total_success: int
    total_failure: int


class PBTLoadRequest(BaseModel):
    """Request model for loading PBT data."""
    csv_path: str = Field(..., description="Path to the CSV file with PBT data")
    reload: Optional[bool] = Field(False, description="Whether to reload the data if already loaded")


class PBTLoadResponse(BaseModel):
    """Response model for loading PBT data."""
    status: str
    message: str
    total_loaded: int
    request_id: Optional[str] = None


class PBTStatistics(BaseModel):
    """Statistics about PBT data."""
    total_pbt_count: int
    indexed_count: int
    cdm_categories: Dict[str, int]  # Count of PBTs by CDM category
    has_synonyms_count: int
    average_synonyms_per_pbt: float
    top_cdm_categories: List[Dict[str, Any]]  # Top CDM categories by count