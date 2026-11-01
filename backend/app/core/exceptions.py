"""
Custom exception definitions and centralized error handlers for the FastAPI backend.
Prevents leaking internal stack traces or machine file paths to API clients.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("backend.exceptions")


class BaseAPIException(Exception):
    """Base class for custom API errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidInputError(BaseAPIException):
    """Raised when file or payload validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code="INVALID_INPUT",
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class UnsupportedModalityError(BaseAPIException):
    """Raised when an unsupported modality is requested."""

    def __init__(self, modality: str):
        super().__init__(
            error_code="UNSUPPORTED_MODALITY",
            message=f"Modality '{modality}' is not supported. Valid modalities: lab, xray, skin, retina.",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"requested_modality": modality},
        )


class UnsupportedTaskError(BaseAPIException):
    """Raised when an unsupported task is requested for a given modality."""

    def __init__(self, modality: str, task: str, valid_tasks: list):
        super().__init__(
            error_code="UNSUPPORTED_TASK",
            message=f"Task '{task}' is not supported for modality '{modality}'. Valid tasks: {valid_tasks}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"modality": modality, "requested_task": task, "valid_tasks": valid_tasks},
        )


class ModelNotAvailableError(BaseAPIException):
    """Raised when a model endpoint is called before its model is trained/validated."""

    def __init__(self, modality: str, task: str):
        super().__init__(
            error_code="MODEL_NOT_AVAILABLE",
            message=f"Model for modality '{modality}' and task '{task}' is not yet available. Training planned in future phase.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"modality": modality, "task": task, "status": "PLANNED"},
        )


class InsufficientInputDataError(BaseAPIException):
    """Raised when required model features are missing and patient safety blocks prediction."""

    def __init__(self, message: str, missing_features: list):
        super().__init__(
            error_code="INSUFFICIENT_INPUT_DATA",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"missing_features": missing_features, "human_review_required": True},
        )


class ProcessingError(BaseAPIException):
    """Raised when an internal processing failure occurs."""

    def __init__(self, message: str = "An internal error occurred while processing the request."):
        super().__init__(
            error_code="PROCESSING_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """JSON exception handler for BaseAPIException."""
    logger.warning(f"API Error [{exc.error_code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Unhandled internal exception handler masking stack traces."""
    logger.error(f"Unhandled Server Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. Please contact technical support.",
            },
        },
    )
