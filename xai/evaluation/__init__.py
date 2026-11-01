"""
XAI Quality Evaluation Framework Package.
Implements quantitative metrics for explainability quality:
- Fidelity / Faithfulness (perturbation drop analysis)
- Stability Index (input perturbation sensitivity)
- Robustness / Sanity Checks (feature shuffling and parameter randomization)
"""

from .fidelity import evaluate_shap_fidelity
from .stability import evaluate_shap_stability
from .robustness import evaluate_shap_robustness
from .evaluator import CommonXAIEvaluator
from .schemas import (
    Modality,
    ExplanationMethod,
    ExplanationType,
    EvaluationStatus,
    GLOBAL_XAI_LIMITATIONS,
    create_modality_xai_schema,
)

__all__ = [
    "evaluate_shap_fidelity",
    "evaluate_shap_stability",
    "evaluate_shap_robustness",
    "CommonXAIEvaluator",
    "Modality",
    "ExplanationMethod",
    "ExplanationType",
    "EvaluationStatus",
    "GLOBAL_XAI_LIMITATIONS",
    "create_modality_xai_schema",
]

