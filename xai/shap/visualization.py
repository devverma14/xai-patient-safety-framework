"""
SHAP visualization module for laboratory report attributions.
Generates global SHAP summary/beeswarm plots, global feature importance bar plots,
and local waterfall/bar attribution plots.
"""

import os
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import shap

from .feature_mapping import format_human_readable_feature_name


def generate_global_shap_plots(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: str = os.path.join("evaluation", "xai", "lab", "plots"),
    top_n: int = 20,
) -> List[str]:
    """
    Generate and save global SHAP summary beeswarm plot and feature importance bar plot.

    Args:
        shap_values: Array of SHAP values (N, num_features).
        X_sample: Preprocessed feature matrix sample (N, num_features).
        feature_names: List of transformed feature names.
        output_dir: Directory path to save plot images.
        top_n: Number of top features to include in plots.

    Returns:
        List of generated plot filepaths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Format human-readable feature names for plot display
    human_feature_names = [format_human_readable_feature_name(f) for f in feature_names]

    saved_paths = []

    # 1. Generate Global Bar Plot (Mean Absolute SHAP values)
    plt.figure(figsize=(10, 8))
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    top_names = [human_feature_names[i] for i in top_indices][::-1]
    top_values = [mean_abs_shap[i] for i in top_indices][::-1]

    plt.barh(range(len(top_names)), top_values, color="#2b5c8f")
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds Output)", fontsize=10)
    plt.title(f"Top {top_n} Global Feature Importances (Lab Report Model)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    bar_path = os.path.join(output_dir, "shap_global_bar_plot.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_paths.append(bar_path)

    # 2. Generate Summary Beeswarm Plot
    plt.figure(figsize=(10, 8))
    try:
        shap.summary_plot(
            shap_values[:, top_indices[::-1]],
            X_sample[:, top_indices[::-1]],
            feature_names=[human_feature_names[i] for i in top_indices[::-1]],
            max_display=top_n,
            show=False,
        )
        plt.title(f"SHAP Summary Beeswarm Plot — Top {top_n} Features", fontsize=12, fontweight="bold")
        plt.tight_layout()

        summary_path = os.path.join(output_dir, "shap_global_summary_plot.png")
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(summary_path)
    except Exception as e:
        plt.close()
        print(f"Warning: Summary plot generation failed with error: {str(e)}")

    return saved_paths


def generate_local_waterfall_plot(
    local_explanation: dict,
    output_path: str = os.path.join("evaluation", "xai", "lab", "plots", "local_waterfall_sample.png"),
    top_k: int = 10,
) -> str:
    """
    Generate and save a local feature contribution bar plot for a single encounter.

    Args:
        local_explanation: Local explanation dictionary from LabSHAPExplainer.explain_local_encounter.
        output_path: Destination PNG image filepath.
        top_k: Number of top contributing features to render.

    Returns:
        Path to saved PNG image file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ranked_items = local_explanation.get("all_ranked_contributions", [])[:top_k]
    if not ranked_items:
        return ""

    # Sort items so highest positive is at top
    items = sorted(ranked_items, key=lambda x: x["shap_value"])

    names = [x["human_readable_name"] for x in items]
    s_vals = [x["shap_value"] for x in items]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in s_vals]  # Red for risk increase, Green for risk decrease

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), s_vals, color=colors)
    plt.yticks(range(len(names)), names, fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP Value (Log-Odds Contribution to 30-Day Readmission Risk)", fontsize=10)
    
    prob_str = f"Prob: {local_explanation['prediction_summary']['probability']:.2f} (Threshold: {local_explanation['prediction_summary']['decision_threshold']:.2f})"
    plt.title(f"Local Feature Attribution — {prob_str}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path
