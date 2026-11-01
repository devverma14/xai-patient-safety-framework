"""
Schemas for Lab Report modality requests and responses.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .common import CommonResponsePayload


class LabPredictionRequest(BaseModel):
    """Optional payload containing explicitly supplied structured clinical data."""
    explicit_structured_input: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of explicitly supplied structured EHR/clinical variables to complement OCR lab report extraction."
    )


class LabReportStatusResponse(BaseModel):
    """Lab modality status response."""
    modality: str = "lab"
    task: str = "report_analysis"
    model_status: str = "AVAILABLE"
    decision_threshold: Optional[float] = None
    xai_method: str = "Clinical Reference Intervals"
    model_features_count: int = 9
