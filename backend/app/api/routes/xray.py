"""
FastAPI route handlers for Chest X-Ray modality (Pneumonia & TB tasks).
"""

from fastapi import APIRouter, File, UploadFile, Query, status
from backend.app.api.schemas.common import CommonResponsePayload
from backend.app.api.schemas.xray import XRayStatusResponse, XRayTask
from backend.app.services.modality_router import modality_dispatcher

from backend.app.services.model_registry import model_registry

router = APIRouter(prefix="/xray", tags=["Chest X-Ray Modality"])


@router.get("/status", response_model=XRayStatusResponse)
async def get_xray_status():
    """Get Chest X-Ray modality status and supported tasks."""
    p_avail = model_registry.is_model_available("xray", "pneumonia")
    tb_avail = model_registry.is_model_available("xray", "tb")
    status_str = "AVAILABLE" if (p_avail and tb_avail) else ("PARTIAL" if (p_avail or tb_avail) else "MODEL_NOT_AVAILABLE")
    p_meta = model_registry.get_metadata("xray", "pneumonia")
    tb_meta = model_registry.get_metadata("xray", "tb")
    return XRayStatusResponse(
        modality="xray",
        supported_tasks=["pneumonia", "tb"],
        model_status=status_str,
        xai_method="Grad-CAM",
        note="Chest X-Ray models: Pneumonia (DenseNet121) and TB (DenseNet121) integrated with Grad-CAM.",
        model_architecture="DenseNet121",
        task_architectures={
            "pneumonia": p_meta.architecture or "DenseNet121",
            "tb": tb_meta.architecture or "DenseNet121",
        },
    )


@router.post(
    "/analyze",
    response_model=CommonResponsePayload,
    status_code=status.HTTP_200_OK,
    summary="Analyze Chest X-Ray Image",
    description=(
        "Extensible analysis endpoint for Chest X-Ray modality. "
        "Supports tasks 'pneumonia' (DenseNet121) or 'tb' (DenseNet121) with Grad-CAM visual explanations."
    ),
)
async def analyze_xray_image(
    file: UploadFile = File(..., description="Uploaded Chest X-Ray DICOM or image file"),
    task: XRayTask = Query(XRayTask.PNEUMONIA, description="Classification task: 'pneumonia' or 'tb'"),
):
    """Analyze Chest X-Ray image endpoint."""
    file_bytes = await file.read()
    return modality_dispatcher.dispatch(
        modality="xray",
        task=task.value,
        file_bytes=file_bytes,
        filename=file.filename or "xray.png",
    )
