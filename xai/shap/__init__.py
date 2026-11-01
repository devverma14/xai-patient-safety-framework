"""
SHAP Explainability Package for Tabular Laboratory Data.
Contains modules for TreeExplainer execution, feature name mapping, local & global attributions,
and visual contribution rendering.
"""

from .explainer import LabSHAPExplainer
from .feature_mapping import format_human_readable_feature_name, build_feature_mapping
from .visualization import generate_global_shap_plots, generate_local_waterfall_plot

__all__ = [
    "LabSHAPExplainer",
    "format_human_readable_feature_name",
    "build_feature_mapping",
    "generate_global_shap_plots",
    "generate_local_waterfall_plot",
]
