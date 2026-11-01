"""
Modality Dispatcher Service routing API requests to appropriate modality processing pipelines.
"""

from typing import Dict, Any, Optional
import logging

from backend.app.core.exceptions import UnsupportedModalityError, UnsupportedTaskError
from backend.app.services.model_registry import model_registry
from backend.app.services.prediction_service import prediction_service
from backend.app.api.schemas.common import CommonResponsePayload

logger = logging.getLogger("backend.services.modality_router")


class ModalityDispatcher:
    """Central Modality Dispatcher."""

    @staticmethod
    def dispatch(
        modality: str,
        task: str,
        file_bytes: bytes,
        filename: str,
        explicit_structured_input: Optional[Dict[str, Any]] = None,
    ) -> CommonResponsePayload:
        """Route request by modality and task to appropriate service pipeline."""
        supported_modalities = model_registry.get_supported_modalities()
        if modality not in supported_modalities:
            raise UnsupportedModalityError(modality)

        supported_tasks = model_registry.get_supported_tasks(modality)
        if task not in supported_tasks:
            raise UnsupportedTaskError(modality, task, supported_tasks)

        logger.info(f"Dispatching request to modality pipeline: '{modality}' (task: '{task}')")

        if modality == "lab":
            return prediction_service.process_lab_analysis(
                file_bytes=file_bytes,
                filename=filename,
                explicit_structured_input=explicit_structured_input,
            )
        else:
            return prediction_service.process_image_analysis(
                modality=modality,
                task=task,
                file_bytes=file_bytes,
                filename=filename,
            )


modality_dispatcher = ModalityDispatcher()
