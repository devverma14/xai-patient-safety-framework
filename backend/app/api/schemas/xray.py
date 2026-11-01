"""
Schemas for Chest X-Ray modality requests and responses.
Task options: 'pneumonia' or 'tb'.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class XRayTask(str, Enum):
    PNEUMONIA = "pneumonia"
    TB = "tb"


class XRayStatusResponse(BaseModel):
    modality: str = "xray"
    supported_tasks: list = ["pneumonia", "tb"]
    model_status: str = "MODEL_NOT_AVAILABLE"
    xai_method: str = "Grad-CAM"
    note: str = "Chest X-Ray models: Pneumonia (DenseNet121) and TB (DenseNet121) integrated with Grad-CAM."
    model_architecture: Optional[str] = "DenseNet121"
    task_architectures: Optional[dict] = Field(default_factory=lambda: {"pneumonia": "DenseNet121", "tb": "DenseNet121"})
