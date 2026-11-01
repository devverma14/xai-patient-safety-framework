"""
Modality-Aware Explanation Engine Service Interface.
Decouples explanation generation into algorithm-specific implementations (SHAP vs Grad-CAM).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

from xai.shap import LabSHAPExplainer

logger = logging.getLogger("backend.services.explanation_service")


class BaseExplanationEngine(ABC):
    """Abstract base class for modality-aware XAI explanation engines."""

    @abstractmethod
    def explain(
        self,
        modality: str,
        task: str,
        input_data: Any,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute feature or visual explanation for model prediction."""
        pass


class SHAPExplanationEngine(BaseExplanationEngine):
    """SHAP Feature Explanation Engine for tabular/lab modalities."""

    def __init__(self):
        self._explainer: Optional[LabSHAPExplainer] = None

    def _get_explainer(self) -> LabSHAPExplainer:
        if self._explainer is None:
            logger.info("Initializing LabSHAPExplainer instance...")
            self._explainer = LabSHAPExplainer()
        return self._explainer

    def explain(
        self,
        modality: str,
        task: str,
        input_data: Any,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("SHAPExplanationEngine expects a single-row pandas DataFrame input_data.")

        explainer = self._get_explainer()
        shap_res = explainer.explain_local_encounter(input_data, top_k=top_k)

        top_increasing = shap_res.get("top_risk_increasing_factors", [])
        top_decreasing = shap_res.get("top_risk_decreasing_factors", [])

        summary_parts = []
        if top_increasing:
            factor_names = [f"{f['human_readable_name']} ({f.get('feature_value_scaled', '')})" for f in top_increasing[:3]]
            summary_parts.append(f"Primary risk drivers: {', '.join(factor_names)}.")
        if top_decreasing:
            factor_names = [f"{f['human_readable_name']} ({f.get('feature_value_scaled', '')})" for f in top_decreasing[:2]]
            summary_parts.append(f"Protective factors: {', '.join(factor_names)}.")

        return {
            "method": "SHAP",
            "available": True,
            "summary": " ".join(summary_parts) if summary_parts else "SHAP explanation computed.",
            "details": {
                "base_value": shap_res.get("base_value"),
                "prediction_summary": shap_res.get("prediction_summary"),
                "top_risk_increasing_factors": top_increasing,
                "top_risk_decreasing_factors": top_decreasing,
            },
        }


class GradCAMExplanationEngine(BaseExplanationEngine):
    """Grad-CAM Visual Explanation Engine for image modalities (Pneumonia, TB, Skin)."""

    def explain(
        self,
        modality: str,
        task: str,
        input_data: Any,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(input_data, dict) or "model" not in input_data or "input_tensor" not in input_data:
            return {
                "method": "Grad-CAM",
                "available": False,
                "summary": f"Grad-CAM visual explanation engine for modality '{modality}' ({task}) is pending model availability.",
                "details": {
                    "heatmap_available": False,
                    "note": "Grad-CAM engine will activate upon model loading.",
                },
            }

        model = input_data["model"]
        input_tensor = input_data["input_tensor"]
        target_class = input_data.get("target_class")
        predicted_label = input_data.get("predicted_label", "Unknown")

        try:
            import numpy as np
            import torch.nn.functional as F
            from xai.gradcam.gradcam import GradCAM, get_default_target_layer

            target_layer = get_default_target_layer(model)
            target_layer_name = target_layer.__class__.__name__
            for name, mod in model.named_modules():
                if mod is target_layer:
                    target_layer_name = name
                    break

            gradcam = GradCAM(model=model, target_layer=target_layer)
            heatmap, evaluated_cls, class_prob = gradcam.generate_heatmap(
                input_tensor=input_tensor,
                target_class=target_class,
            )
            gradcam.remove_hooks()

            min_val = float(np.min(heatmap))
            max_val = float(np.max(heatmap))
            mean_val = float(np.mean(heatmap))

            # Downsample 224x224 heatmap to a compact 7x7 spatial grid for API response serialization
            import torch
            heatmap_t = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            grid_t = F.adaptive_avg_pool2d(heatmap_t, (7, 7)).squeeze().numpy()
            grid_flat = [round(float(v), 4) for v in grid_t.flatten()]

            return {
                "method": "Grad-CAM",
                "available": True,
                "summary": f"Grad-CAM visual activation heatmap generated for {predicted_label} prediction.",
                "details": {
                    "target_layer": target_layer_name,
                    "target_class_index": evaluated_cls,
                    "predicted_label": predicted_label,
                    "evaluated_probability": round(float(class_prob), 4),
                    "heatmap_shape": list(heatmap.shape),
                    "heatmap_min": round(min_val, 4),
                    "heatmap_max": round(max_val, 4),
                    "heatmap_mean": round(mean_val, 4),
                    "spatial_summary_grid_7x7": grid_flat,
                },
            }
        except Exception as e:
            logger.error(f"Grad-CAM execution error for ({modality}, {task}): {str(e)}", exc_info=True)
            return {
                "method": "Grad-CAM",
                "available": False,
                "summary": f"Grad-CAM visual explanation computation failed: {str(e)}",
                "details": {"error": str(e)},
            }


class ExplanationService:
    """Modality-aware explanation service factory."""

    def __init__(self):
        self._shap_engine = SHAPExplanationEngine()
        self._gradcam_engine = GradCAMExplanationEngine()

    def get_engine(self, modality: str) -> BaseExplanationEngine:
        if modality == "lab":
            return self._shap_engine
        return self._gradcam_engine


explanation_service = ExplanationService()
