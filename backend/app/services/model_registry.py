"""
Centralized Model Registry mapping (modality, task) pairs to model artifacts, metadata, and status.
Supports controlled lazy-loading and prevents false model availability claims.
"""

import os
import joblib
import json
from typing import Dict, Any, Tuple, Optional
import logging

from backend.app.core.config import settings
from backend.app.core.exceptions import ModelNotAvailableError, UnsupportedModalityError, UnsupportedTaskError

logger = logging.getLogger("backend.services.model_registry")


def _resolve_architecture(metadata_path: Optional[str], default_arch: str) -> str:
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            arch = data.get("selected_architecture") or data.get("architecture")
            if arch:
                if arch.lower() == "densenet121":
                    return "DenseNet121"
                elif arch.lower() in ("efficientnet_b0", "efficientnet-b0"):
                    return "EfficientNet-B0"
                elif arch.lower() == "xgboost":
                    return "XGBoost"
                return arch
        except Exception:
            pass
    return default_arch


class ModelMetadata:
    def __init__(
        self,
        modality: str,
        task: str,
        status: str,
        xai_method: str,
        artifact_path: Optional[str] = None,
        architecture: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.modality = modality
        self.task = task
        self.status = status  # 'AVAILABLE' or 'MODEL_NOT_AVAILABLE'
        self.xai_method = xai_method  # 'SHAP' or 'Grad-CAM'
        self.artifact_path = artifact_path
        self.architecture = architecture
        self.model_name = model_name or architecture


class ModelRegistry:
    """Registry maintaining metadata and lazy loading for all V1 modality models."""

    def __init__(self):
        self._registry: Dict[Tuple[str, str], ModelMetadata] = {
            ("lab", "report_analysis"): ModelMetadata(
                modality="lab",
                task="report_analysis",
                status="DETERMINISTIC_ENGINE",
                xai_method="Clinical Reference Intervals",
                artifact_path=None,
                architecture="Deterministic Rules",
                model_name="Clinical Physiological Reference Interval Engine",
            ),
            ("xray", "pneumonia"): ModelMetadata(
                modality="xray",
                task="pneumonia",
                status="AVAILABLE" if os.path.exists(settings.PNEUMONIA_MODEL_PATH) else "MODEL_NOT_AVAILABLE",
                xai_method="Grad-CAM",
                artifact_path=settings.PNEUMONIA_MODEL_PATH,
                architecture="DenseNet121",
                model_name="DenseNet121",
            ),
            ("xray", "tb"): ModelMetadata(
                modality="xray",
                task="tb",
                status="AVAILABLE" if os.path.exists(settings.TB_MODEL_PATH) else "MODEL_NOT_AVAILABLE",
                xai_method="Grad-CAM",
                artifact_path=settings.TB_MODEL_PATH,
                architecture=_resolve_architecture(settings.TB_METADATA_PATH, "DenseNet121"),
                model_name=_resolve_architecture(settings.TB_METADATA_PATH, "DenseNet121"),
            ),
            ("skin", "ham10000"): ModelMetadata(
                modality="skin",
                task="ham10000",
                status="AVAILABLE" if os.path.exists(settings.SKIN_MODEL_PATH) else "MODEL_NOT_AVAILABLE",
                xai_method="Grad-CAM",
                artifact_path=settings.SKIN_MODEL_PATH,
                architecture="DenseNet121",
                model_name="DenseNet121 (Binary Triage)",
            ),
            ("retina", "diabetic_retinopathy"): ModelMetadata(
                modality="retina",
                task="diabetic_retinopathy",
                status="AVAILABLE" if os.path.exists(settings.RETINA_MODEL_PATH) else "MODEL_NOT_AVAILABLE",
                xai_method="Grad-CAM",
                artifact_path=settings.RETINA_MODEL_PATH,
                architecture="EfficientNet-B0",
                model_name="EfficientNet-B0",
            ),
        }
        self._cached_models: Dict[Tuple[str, str], Any] = {}
        self._cached_preprocessors: Dict[Tuple[str, str], Any] = {}

    def get_supported_modalities(self) -> list:
        return list(set(k[0] for k in self._registry.keys()))

    def get_supported_tasks(self, modality: str) -> list:
        if modality not in self.get_supported_modalities():
            raise UnsupportedModalityError(modality)
        return [k[1] for k in self._registry.keys() if k[0] == modality]

    def get_metadata(self, modality: str, task: str) -> ModelMetadata:
        key = (modality, task)
        if modality not in self.get_supported_modalities():
            raise UnsupportedModalityError(modality)
        if key not in self._registry:
            valid_tasks = self.get_supported_tasks(modality)
            raise UnsupportedTaskError(modality, task, valid_tasks)
        return self._registry[key]

    def is_model_available(self, modality: str, task: str) -> bool:
        meta = self.get_metadata(modality, task)
        return meta.status == "AVAILABLE" and bool(meta.artifact_path)

    def load_model(self, modality: str, task: str) -> Any:
        """Lazily load and return trained model artifact if available."""
        meta = self.get_metadata(modality, task)
        if meta.status != "AVAILABLE" or not meta.artifact_path or modality == "lab":
            raise ModelNotAvailableError(modality, task)

        key = (modality, task)
        if key in self._cached_models:
            return self._cached_models[key]

        logger.info(f"Lazy-loading model artifact for ({modality}, {task}) from {meta.artifact_path}")
        try:
            if modality == "xray" and task == "pneumonia":
                import torch
                import torch.nn as nn
                import torchvision.models as models
                model = models.densenet121(weights=None)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 2)
                ckpt = torch.load(meta.artifact_path, map_location="cpu")
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                model.eval()
            elif modality == "xray" and task == "tb":
                import torch
                import torch.nn as nn
                import torchvision.models as models
                model = models.densenet121(weights=None)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 2)
                ckpt = torch.load(meta.artifact_path, map_location="cpu")
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                model.eval()
            elif modality == "skin" and task == "ham10000":
                import torch
                import torch.nn as nn
                import torchvision.models as models
                model = models.densenet121(weights=None)
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features, 2),
                )
                ckpt = torch.load(meta.artifact_path, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else (ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
                model.load_state_dict(state_dict)
                model.eval()
            elif modality == "retina" and task == "diabetic_retinopathy":
                import hashlib
                import torch
                import torch.nn as nn
                import torchvision.models as models

                EXPECTED_RETINA_SHA = "4bf9260be9393f440d2a201dc763edb2eab52ab57c1f2d7c6780deddd1ac7a07"
                hasher = hashlib.sha256()
                with open(meta.artifact_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                actual_sha = hasher.hexdigest()

                if actual_sha != EXPECTED_RETINA_SHA:
                    logger.error(
                        f"Retina model SHA-256 integrity mismatch! Expected {EXPECTED_RETINA_SHA}, got {actual_sha}"
                    )
                    raise ModelNotAvailableError(modality, task)

                model = models.efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, 2)
                ckpt = torch.load(meta.artifact_path, map_location="cpu")
                state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else (
                    ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                )
                model.load_state_dict(state_dict)
                model.eval()
            else:
                raise ModelNotAvailableError(modality, task)

            self._cached_models[key] = model
            return model
        except Exception as e:
            logger.error(f"Failed loading model for ({modality}, {task}): {str(e)}", exc_info=True)
            if isinstance(e, ModelNotAvailableError):
                raise e
            raise ModelNotAvailableError(modality, task)

    def load_preprocessor(self, modality: str, task: str) -> Any:
        """Lazily load and return preprocessor artifact for specified modality/task."""
        raise ModelNotAvailableError(modality, task)


model_registry = ModelRegistry()
