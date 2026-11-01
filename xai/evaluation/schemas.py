"""
XAI Common Evaluation Schemas & Data Structures.

Defines standardized, modality-aware evaluation data structures that represent:
- Modality (lab, pneumonia, tb, skin, retina)
- Task definition & clinical scope
- Model architecture & prediction context
- Explanation method (SHAP vs Grad-CAM)
- Explanation type (Quantitative Feature Attribution vs Visual Spatial Attribution)
- Explanation quality metrics (Fidelity, Stability, Robustness, Perturbation Drop, Spatial Range)
- Explicit limitations & disclaimers
- Evaluation status (VERIFIED_COMPLETE, PARTIALLY_EVALUATED, NOT_EVALUATED)
"""

from typing import Dict, Any, List, Optional
from enum import Enum


class Modality(str, Enum):
    LAB = "lab"
    PNEUMONIA = "pneumonia"
    TB = "tb"
    SKIN = "skin"
    RETINA = "retina"


class ExplanationMethod(str, Enum):
    SHAP = "SHAP (TreeExplainer)"
    GRAD_CAM = "Grad-CAM"
    NOT_EVALUATED = "NOT_EVALUATED"


class ExplanationType(str, Enum):
    QUANTITATIVE_FEATURE = "Quantitative Feature Attribution"
    VISUAL_SPATIAL = "Visual Spatial Attribution"
    NOT_EVALUATED = "NOT_EVALUATED"


class EvaluationStatus(str, Enum):
    VERIFIED_COMPLETE = "VERIFIED_COMPLETE"
    PARTIALLY_EVALUATED = "PARTIALLY_EVALUATED"
    NOT_EVALUATED = "NOT_EVALUATED"


# Common explicit XAI limitations list
GLOBAL_XAI_LIMITATIONS = [
    "XAI feature attributions and visual heatmaps indicate correlation, not causal reasoning.",
    "Grad-CAM spatial resolution is limited by feature map dimensions (e.g. 7x7 spatial grid).",
    "SHAP feature values depend on background dataset distributions and input representations.",
    "Quantitative and visual XAI metrics do not establish clinical validity or diagnostic proof.",
    "Autonomous clinical decisions must not be based solely on AI attributions; prospective clinical validation is required.",
]


def create_modality_xai_schema(
    modality: str,
    task: str,
    model: str,
    explanation_method: str,
    explanation_type: str,
    prediction_context: Dict[str, Any],
    explanation_quality_metrics: Dict[str, Any],
    limitations: List[str],
    evaluation_status: str,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Construct a standardized dictionary representation for a modality XAI evaluation.
    """
    return {
        "modality": modality,
        "task": task,
        "model": model,
        "explanation_method": explanation_method,
        "explanation_type": explanation_type,
        "prediction_context": prediction_context,
        "explanation_quality_metrics": explanation_quality_metrics,
        "limitations": limitations,
        "evaluation_status": evaluation_status,
        "artifact_paths": artifact_paths,
    }
