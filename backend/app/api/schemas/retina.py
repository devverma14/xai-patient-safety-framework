"""
Schemas for Retina modality requests and responses.
Task options: 'diabetic_retinopathy'.
"""

from enum import Enum
from pydantic import BaseModel, Field


from typing import Optional


class RetinaTask(str, Enum):
    DIABETIC_RETINOPATHY = "diabetic_retinopathy"


class RetinaStatusResponse(BaseModel):
    modality: str = "retina"
    supported_tasks: list = ["diabetic_retinopathy"]
    model_status: str = "AVAILABLE"
    xai_method: str = "Grad-CAM"
    note: str = "Retina Diabetic Retinopathy EfficientNet-B0 CNN model loaded and operational."
    model_architecture: Optional[str] = "EfficientNet-B0"
