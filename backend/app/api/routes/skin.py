"""
FastAPI route handlers for Skin Lesion modality (HAM10000 task).
"""

from fastapi import APIRouter, File, UploadFile, Query, status
from backend.app.api.schemas.common import CommonResponsePayload
from backend.app.api.schemas.skin import SkinStatusResponse, SkinTask
from backend.app.services.modality_router import modality_dispatcher

from backend.app.services.model_registry import model_registry

router = APIRouter(prefix="/skin", tags=["Skin Lesion Modality"])


@router.get("/status", response_model=SkinStatusResponse)
async def get_skin_status():
    """Get Skin Lesion modality status."""
    available = model_registry.is_model_available("skin", "ham10000")
    return SkinStatusResponse(
        modality="skin",
        supported_tasks=["ham10000"],
        model_status="AVAILABLE" if available else "MODEL_NOT_AVAILABLE",
        xai_method="Grad-CAM",
        note="Skin Lesion HAM10000 Binary CNN model (DenseNet121, frozen threshold t=0.40) integrated with Grad-CAM.",
    )


@router.post(
    "/analyze",
    response_model=CommonResponsePayload,
    status_code=status.HTTP_200_OK,
    summary="Analyze Skin Lesion Image",
    description=(
        "Extensible analysis endpoint for Skin Lesion modality (HAM10000 task). "
        "Uses binary DenseNet121 model with frozen threshold t=0.40 (HIGHER-RISK / LOWER-RISK) and Grad-CAM visual explanations."
    ),
)
async def analyze_skin_image(
    file: UploadFile = File(..., description="Uploaded Skin Lesion image file"),
    task: SkinTask = Query(SkinTask.HAM10000, description="Classification task: 'ham10000'"),
):
    """Analyze Skin Lesion image endpoint."""
    file_bytes = await file.read()
    return modality_dispatcher.dispatch(
        modality="skin",
        task=task.value,
        file_bytes=file_bytes,
        filename=file.filename or "skin.png",
    )
