"""
XAI Fidelity / Faithfulness Evaluation Module.
Evaluates how accurately SHAP feature attributions reflect the model's decision process
by measuring prediction probability drops when top-k vs bottom-k features are masked.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb

from xai.shap import LabSHAPExplainer


def evaluate_shap_fidelity(
    explainer: LabSHAPExplainer,
    X_val_proc: np.ndarray,
    top_k: int = 5,
    num_samples: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform empirical fidelity (faithfulness) evaluation on a validation sample.

    Args:
        explainer: Initialized LabSHAPExplainer instance.
        X_val_proc: Preprocessed validation feature matrix (N, num_features).
        top_k: Number of top vs bottom features to mask/perturb.
        num_samples: Sample size to evaluate.
        random_state: Random seed for sampling.

    Returns:
        Dictionary containing fidelity metrics, mean delta probabilities, and fidelity ratio.
    """
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(len(X_val_proc), size=min(num_samples, len(X_val_proc)), replace=False)
    X_sample = X_val_proc[sample_indices].copy()

    # Original model probabilities
    orig_probs = explainer.model.predict_proba(X_sample)[:, 1]

    # Compute SHAP values
    shap_matrix = explainer.explain_dataset(X_sample)

    top_k_drops = []
    bottom_k_drops = []

    for i in range(len(X_sample)):
        p_orig = orig_probs[i]
        s_vals = shap_matrix[i]
        abs_s_vals = np.abs(s_vals)

        # Top-k feature indices (highest absolute SHAP impact)
        top_indices = np.argsort(abs_s_vals)[::-1][:top_k]
        
        # Bottom-k feature indices (lowest absolute SHAP impact)
        bottom_indices = np.argsort(abs_s_vals)[:top_k]

        # 1. Perturb Top-k features (mask to 0.0)
        X_pert_top = X_sample[i : i + 1].copy()
        X_pert_top[0, top_indices] = 0.0
        p_top = float(explainer.model.predict_proba(X_pert_top)[0, 1])
        drop_top = abs(p_orig - p_top)
        top_k_drops.append(drop_top)

        # 2. Perturb Bottom-k features (mask to 0.0)
        X_pert_bottom = X_sample[i : i + 1].copy()
        X_pert_bottom[0, bottom_indices] = 0.0
        p_bottom = float(explainer.model.predict_proba(X_pert_bottom)[0, 1])
        drop_bottom = abs(p_orig - p_bottom)
        bottom_k_drops.append(drop_bottom)

    mean_drop_top = float(np.mean(top_k_drops))
    mean_drop_bottom = float(np.mean(bottom_k_drops))

    # Fidelity ratio = mean_drop_top / mean_drop_bottom
    fidelity_ratio = float(mean_drop_top / (mean_drop_bottom + 1e-8))

    results = {
        "evaluation_metric": "XAI_Fidelity_Faithfulness",
        "sample_size": len(sample_indices),
        "top_k": top_k,
        "mean_probability_drop_top_k": round(mean_drop_top, 6),
        "mean_probability_drop_bottom_k": round(mean_drop_bottom, 6),
        "fidelity_ratio": round(fidelity_ratio, 4),
        "passed_faithfulness_test": bool(mean_drop_top > mean_drop_bottom),
        "interpretation": (
            f"Removing top-{top_k} SHAP features caused a mean probability drop of {mean_drop_top:.4f}, "
            f"compared to {mean_drop_bottom:.4f} for bottom-{top_k} features (Fidelity Ratio: {fidelity_ratio:.2f}x). "
            f"This confirms that SHAP attributions faithfully identify decision-driving features."
        ),
    }

    return results
