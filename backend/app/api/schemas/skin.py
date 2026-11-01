"""
Schemas for Skin Lesion modality requests and responses.
Task options: 'ham10000'.
"""

from enum import Enum
from pydantic import BaseModel, Field


class SkinTask(str, Enum):
    HAM10000 = "ham10000"


class SkinStatusResponse(BaseModel):
    modality: str = "skin"
    supported_tasks: list = ["ham10000"]
    model_status: str = "MODEL_NOT_AVAILABLE"
    xai_method: str = "Grad-CAM"
    note: str = "Skin Lesion HAM10000 CNN model planned in future development phase."
