"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability package.
Provides target layer attribution, heatmap generation, and visual overlay utilities for medical image models.
"""

from .gradcam import GradCAM, get_default_target_layer
from .visualization import (
    normalize_heatmap,
    apply_colormap_on_image,
    generate_gradcam_overlay,
    save_gradcam_comparison_figure,
)

__all__ = [
    "GradCAM",
    "get_default_target_layer",
    "normalize_heatmap",
    "apply_colormap_on_image",
    "generate_gradcam_overlay",
    "save_gradcam_comparison_figure",
]
