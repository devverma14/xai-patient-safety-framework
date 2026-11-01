"""
FastAPI route handlers for Lab Report modality.
"""

import json
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, status
from pydantic import ValidationError

from backend.app.api.schemas.common import CommonResponsePayload
from backend.app.api.schemas.lab import LabReportStatusResponse
from backend.app.core.exceptions import InvalidInputError
from backend.app.services.modality_router import modality_dispatcher
from backend.app.services.model_registry import model_registry

router = APIRouter(prefix="/lab", tags=["Lab Report Modality"])


@router.get("/status", response_model=LabReportStatusResponse)
async def get_lab_status():
    """Get Lab report pipeline status and reference range comparison method."""
    return LabReportStatusResponse(
        modality="lab",
        task="report_analysis",
        model_status="AVAILABLE",
        decision_threshold=None,
        xai_method="Clinical Reference Intervals",
        model_features_count=9,
    )


@router.post(
    "/analyze",
    response_model=CommonResponsePayload,
    status_code=status.HTTP_200_OK,
    summary="Analyze Lab Report (PDF/Image)",
    description=(
        "Processes uploaded Lab report file through OCR, document quality validation, "
        "parameter extraction, and deterministic clinical reference-range comparison."
    ),
)
async def analyze_lab_report(
    file: UploadFile = File(..., description="Uploaded PDF or Image laboratory report file"),
    explicit_structured_input: Optional[str] = Form(
        None,
        description="Optional JSON string containing explicitly supplied structured clinical/EHR features.",
    ),
):
    """Analyze Lab Report PDF/Image endpoint."""
    file_bytes = await file.read()

    structured_dict = None
    if explicit_structured_input:
        try:
            structured_dict = json.loads(explicit_structured_input)
            if not isinstance(structured_dict, dict):
                raise InvalidInputError("'explicit_structured_input' must be a valid JSON dictionary object.")
        except json.JSONDecodeError:
            raise InvalidInputError("Invalid JSON format in 'explicit_structured_input' form parameter.")

    return modality_dispatcher.dispatch(
        modality="lab",
        task="report_analysis",
        file_bytes=file_bytes,
        filename=file.filename or "lab_report.pdf",
        explicit_structured_input=None,
    )
