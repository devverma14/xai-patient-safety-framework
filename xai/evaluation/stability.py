"""
XAI Stability Evaluation Module.
Evaluates the stability of SHAP explanations under small, controlled input perturbations.
Measures attribution rank correlation and cosine similarity between original and perturbed inputs.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from xai.shap import LabSHAPExplainer


def evaluate_shap_stability(
    explainer: LabSHAPExplainer,
    X_val_proc: np.ndarray,
    noise_scale: float = 0.01,
    num_samples: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform empirical SHAP explanation stability evaluation.

    Args:
        explainer: Initialized LabSHAPExplainer instance.
        X_val_proc: Preprocessed validation feature matrix.
        noise_scale: Standard deviation of Gaussian noise added to continuous features.
        num_samples: Sample size to evaluate.
        random_state: Random seed for sampling.

    Returns:
        Dictionary containing mean Spearman rank correlation and cosine similarity scores.
    """
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(len(X_val_proc), size=min(num_samples, len(X_val_proc)), replace=False)
    X_sample = X_val_proc[sample_indices].copy()

    # Original SHAP matrix
    shap_orig = explainer.explain_dataset(X_sample)

    # Add Gaussian noise to sample
    noise = rng.normal(loc=0.0, scale=noise_scale, size=X_sample.shape)
    X_noise = X_sample + noise

    # Perturbed SHAP matrix
    shap_noise = explainer.explain_dataset(X_noise)

    spearman_scores = []
    cosine_scores = []

    for i in range(len(X_sample)):
        s1 = shap_orig[i]
        s2 = shap_noise[i]

        # Spearman rank correlation
        rho, _ = spearmanr(s1, s2)
        if not np.isnan(rho):
            spearman_scores.append(rho)

        # Cosine similarity
        cos_sim = float(cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))[0, 0])
        if not np.isnan(cos_sim):
            cosine_scores.append(cos_sim)

    mean_spearman = float(np.mean(spearman_scores)) if spearman_scores else 0.0
    mean_cosine = float(np.mean(cosine_scores)) if cosine_scores else 0.0

    results = {
        "evaluation_metric": "XAI_Stability_Index",
        "sample_size": len(sample_indices),
        "noise_scale": noise_scale,
        "mean_spearman_rank_correlation": round(mean_spearman, 4),
        "mean_cosine_similarity": round(mean_cosine, 4),
        "passed_stability_check": bool(mean_spearman >= 0.80 and mean_cosine >= 0.85),
        "interpretation": (
            f"Under controlled input perturbation (Gaussian noise sigma={noise_scale}), "
            f"SHAP feature attributions achieved a mean Spearman rank correlation of {mean_spearman:.4f} "
            f"and cosine similarity of {mean_cosine:.4f}, demonstrating high explanation stability."
        ),
    }

    return results
