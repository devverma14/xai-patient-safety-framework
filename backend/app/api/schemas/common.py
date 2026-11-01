"""
Common API response schemas enforcing unified response contract across all modalities.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class PredictionDetails(BaseModel):
    """Prediction outcome details."""
    label: Optional[str] = Field(None, description="Predicted target class label")
    probability: Optional[float] = Field(None, description="Positive class prediction probability (0.0 to 1.0)")
    threshold: Optional[float] = Field(None, description="Decision threshold applied")
    model_name: Optional[str] = Field(None, description="Inference model display name")
    model_architecture: Optional[str] = Field(None, description="Inference model underlying architecture")
    raw_output: Optional[Dict[str, Any]] = Field(None, description="Additional model output metrics")


class ExplanationDetails(BaseModel):
    """Modality-aware explanation engine details."""
    method: str = Field(..., description="Explanation algorithm used ('SHAP' or 'Grad-CAM')")
    available: bool = Field(..., description="Whether feature explanation was successfully computed")
    summary: Optional[str] = Field(None, description="Summary textual explanation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Feature contribution scores or heatmap data")


class QualityDetails(BaseModel):
    """Input quality and feature readiness assessment."""
    input_quality: str = Field(..., description="File quality check status ('passed', 'failed', 'warning')")
    ocr_confidence: Optional[float] = Field(None, description="OCR text extraction confidence score")
    extraction_confidence: Optional[float] = Field(None, description="Lab parameter extraction confidence score")
    explanation_quality: Dict[str, Any] = Field(default_factory=dict, description="Explanation metric quality")


class CommonResponsePayload(BaseModel):
    """Unified API response schema across all 4 V1 modalities."""
    status: str = Field(..., description="Status: 'success', 'review_required', 'insufficient_data', 'model_not_available', 'error'")
    modality: str = Field(..., description="Modality identifier: 'lab', 'xray', 'skin', 'retina'")
    task: str = Field(..., description="Specific clinical classification task")
    prediction: Optional[PredictionDetails] = Field(None, description="Prediction outcome if available")
    explanation: Optional[ExplanationDetails] = Field(None, description="Modality XAI explanation details")
    quality: QualityDetails = Field(..., description="Input quality and extraction metrics")
    review_required: bool = Field(False, description="Whether human clinician review is required")
    review_reasons: List[str] = Field(default_factory=list, description="Reasons triggering human review flag")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or system notices")
    payload_details: Optional[Dict[str, Any]] = Field(None, description="Modality-specific extra payload details")


class ErrorResponsePayload(BaseModel):
    """Standard error response model."""
    status: str = "error"
    error: Dict[str, Any]
