"""
Visualization utilities for Grad-CAM heatmaps and visual overlays on medical X-ray images.
Strictly ensures zero patient-identifying information is written into visualization titles or saved artifacts.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Safely normalize 2D heatmap values to [0.0, 1.0] float32 range.

    Args:
        heatmap: 2D numpy array.

    Returns:
        Normalized 2D float32 array in [0.0, 1.0].
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    max_val = np.max(heatmap)
    min_val = np.min(heatmap)

    if max_val > min_val:
        return (heatmap - min_val) / (max_val - min_val)
    return np.zeros_like(heatmap, dtype=np.float32)


def apply_colormap_on_image(
    heatmap: np.ndarray,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Map 2D normalized [0, 1] heatmap to RGB color array (H, W, 3) uint8.

    Args:
        heatmap: Normalized 2D float array in [0, 1].
        colormap_name: Matplotlib colormap name (default: 'jet').

    Returns:
        RGB numpy array with shape (H, W, 3) and dtype uint8.
    """
    heatmap_norm = normalize_heatmap(heatmap)
    cmap = plt.get_cmap(colormap_name)
    colored_rgba = cmap(heatmap_norm)
    colored_rgb = (colored_rgba[:, :, :3] * 255.0).astype(np.uint8)
    return colored_rgb


def generate_gradcam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend normalized Grad-CAM heatmap with original RGB image.

    Formula:
        Overlay = (1 - alpha) * Image + alpha * Heatmap_RGB

    Args:
        image_rgb: RGB image array of shape (H, W, 3), dtype uint8 or float [0, 1].
        heatmap: 2D float array of shape (H, W) in [0, 1].
        alpha: Blending weight for heatmap overlay (default: 0.4).
        colormap_name: Matplotlib colormap name.

    Returns:
        Tuple of:
            - heatmap_rgb: (H, W, 3) uint8 colored heatmap.
            - overlay_rgb: (H, W, 3) uint8 blended overlay image.
    """
    if image_rgb.dtype != np.uint8:
        if np.max(image_rgb) <= 1.0:
            image_rgb = (image_rgb * 255.0).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)

    heatmap_rgb = apply_colormap_on_image(heatmap, colormap_name=colormap_name)

    img_float = image_rgb.astype(np.float32)
    map_float = heatmap_rgb.astype(np.float32)

    blended = (1.0 - alpha) * img_float + alpha * map_float
    overlay_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    return heatmap_rgb, overlay_rgb


def save_gradcam_comparison_figure(
    image_rgb: np.ndarray,
    heatmap_norm: np.ndarray,
    overlay_rgb: np.ndarray,
    save_path: str,
    title_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save 3-panel comparison figure (Original X-Ray, Heatmap, Overlay) to disk.

    Ensures zero patient-identifying data (such as patient IDs or file paths) is rendered.

    Args:
        image_rgb: Original RGB X-ray image array (H, W, 3).
        heatmap_norm: Normalized 2D float heatmap (H, W).
        overlay_rgb: Blended overlay RGB image array (H, W, 3).
        save_path: File path to save output PNG figure.
        title_info: Dictionary containing 'true_label', 'pred_label', 'probability', 'target_layer'.

    Returns:
        String absolute path to saved figure artifact.
    """
    save_p = Path(save_path).resolve()
    save_p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original Image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Chest X-Ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Normalized Grad-CAM Heatmap
    im1 = axes[1].imshow(heatmap_norm, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Blended Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Heatmap Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    if title_info is not None:
        true_lbl = title_info.get("true_label", "Unknown")
        pred_lbl = title_info.get("pred_label", "Unknown")
        prob_val = title_info.get("probability", 0.0)
        target_layer_str = title_info.get("target_layer", "DenseNet121 Layer")

        fig_title = (
            f"AI-Assisted Pneumonia Classification Explanation (Grad-CAM)\n"
            f"True Label: {true_lbl} | Predicted: {pred_lbl} | "
            f"Probability: {prob_val:.4f} | Target Layer: {target_layer_str}"
        )
        fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(save_p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_p)
