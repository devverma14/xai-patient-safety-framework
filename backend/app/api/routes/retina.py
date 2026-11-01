"""
FastAPI route handlers for Retina modality (Diabetic Retinopathy task).
"""

from fastapi import APIRouter, File, UploadFile, Query, status
from backend.app.api.schemas.common import CommonResponsePayload
from backend.app.api.schemas.retina import RetinaStatusResponse, RetinaTask
from backend.app.services.modality_router import modality_dispatcher

from backend.app.services.model_registry import model_registry

router = APIRouter(prefix="/retina", tags=["Retina Modality"])


@router.get("/status", response_model=RetinaStatusResponse)
async def get_retina_status():
    """Get Retina modality status."""
    is_available = model_registry.is_model_available("retina", "diabetic_retinopathy")
    return RetinaStatusResponse(
        modality="retina",
        supported_tasks=["diabetic_retinopathy"],
        model_status="AVAILABLE" if is_available else "MODEL_NOT_AVAILABLE",
        xai_method="Grad-CAM",
        note=(
            "Retina Diabetic Retinopathy EfficientNet-B0 CNN model loaded and operational."
            if is_available
            else "Retina Diabetic Retinopathy model artifact not available."
        ),
        model_architecture="EfficientNet-B0",
    )


@router.post(
    "/analyze",
    response_model=CommonResponsePayload,
    status_code=status.HTTP_200_OK,
    summary="Analyze Retinal Image (Diabetic Retinopathy Classification)",
    description=(
        "Analyzes uploaded Retinal fundus image for Diabetic Retinopathy classification using EfficientNet-B0 "
        "and computes Grad-CAM visual spatial attribution explanations."
    ),
)
async def analyze_retina_image(
    file: UploadFile = File(..., description="Uploaded Retinal fundus image file"),
    task: RetinaTask = Query(RetinaTask.DIABETIC_RETINOPATHY, description="Classification task: 'diabetic_retinopathy'"),
):
    """Analyze Retinal image endpoint."""
    file_bytes = await file.read()
    return modality_dispatcher.dispatch(
        modality="retina",
        task=task.value,
        file_bytes=file_bytes,
        filename=file.filename or "retina.png",
    )
