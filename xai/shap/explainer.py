"""
LabSHAPExplainer module for laboratory report tabular dataset.
Loads Phase 5 final XGBoost model and preprocessor artifacts, initializes SHAP TreeExplainer,
verifies SHAP output space additivity, and provides local and global feature-level explanations.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import shap

from preprocessing.lab import LabPreprocessor
from .feature_mapping import format_human_readable_feature_name, build_feature_mapping


class LabSHAPExplainer:
    """
    SHAP TreeExplainer wrapper for Phase 5 finalized XGBoost Laboratory Model.
    Provides mathematically consistent log-odds space attributions and human-readable feature maps.
    """

    def __init__(
        self,
        model_path: str = os.path.join("models", "lab", "final", "xgb_model.joblib"),
        preprocessor_path: str = os.path.join("models", "lab", "final", "preprocessor.joblib"),
        metadata_path: str = os.path.join("models", "lab", "final", "model_metadata.json"),
    ):
        """
        Initialize SHAP Explainer with Phase 5 model and preprocessor artifacts.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Final model binary not found at: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Final preprocessor artifact not found at: {preprocessor_path}")

        self.model: xgb.XGBClassifier = joblib.load(model_path)
        self.preprocessor: LabPreprocessor = joblib.load(preprocessor_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.threshold: float = self.metadata.get("selected_decision_threshold", 0.35)
        self.feature_names: List[str] = self.preprocessor.get_feature_names_out()

        # Initialize exact tree-path SHAP TreeExplainer targeting raw margin / log-odds output space
        self.explainer = shap.TreeExplainer(self.model)
        
        # Handle scalar vs array base_value representation in SHAP
        if isinstance(self.explainer.expected_value, (np.ndarray, list)):
            self.base_value = float(self.explainer.expected_value[0])
        else:
            self.base_value = float(self.explainer.expected_value)

        # Output space documentation
        self.output_space = "raw_log_odds_margin"

    def get_feature_names(self) -> List[str]:
        """Return transformed feature names."""
        return self.feature_names

    def verify_additivity(self, X_proc: np.ndarray, tol: float = 0.05) -> bool:
        """
        Verify SHAP additivity: base_value + sum(shap_values_i) == margin_prediction_i.

        Args:
            X_proc: Preprocessed feature matrix.
            tol: Absolute tolerance for log-odds summation.

        Returns:
            Boolean True if additivity check passes.
        """
        raw_margins = self.model.predict(X_proc, output_margin=True)
        shap_vals = self.explainer.shap_values(X_proc)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        sums = self.base_value + np.sum(shap_vals, axis=1)
        diff = np.abs(sums - raw_margins)
        max_diff = float(np.max(diff))

        if max_diff > tol:
            raise ValueError(f"SHAP additivity verification failed! Max diff: {max_diff:.6f} > tol ({tol})")

        return True

    def explain_local_encounter(
        self,
        raw_input_df: pd.DataFrame,
        top_k: int = 5,
        actual_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate local SHAP explanation for a single laboratory encounter input.

        Args:
            raw_input_df: Single-row raw input DataFrame (or multi-row where 1st row is used).
            top_k: Number of top positive/negative feature attributions to extract.
            actual_label: Optional ground truth binary label for evaluation reporting.

        Returns:
            Dictionary containing model prediction, probability, threshold, base value,
            margin, top positive factors, and top negative factors.
        """
        if len(raw_input_df) > 1:
            raw_input_df = raw_input_df.iloc[[0]].copy()

        # 1. Run Preprocessing
        X_proc = self.preprocessor.transform(raw_input_df)

        # 2. Compute Model Prediction & Probability
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        raw_margin = float(self.model.predict(X_proc, output_margin=True)[0])
        pred_class = 1 if prob >= self.threshold else 0
        pred_label_str = "<30 (High 30-Day Readmission Risk)" if pred_class == 1 else "NO / >30 (Low 30-Day Readmission Risk)"

        # 3. Compute SHAP Values
        shap_vals = self.explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        shap_vec = shap_vals[0]

        # 4. Verify Local Additivity
        additivity_check = np.isclose(self.base_value + np.sum(shap_vec), raw_margin, atol=0.05)

        # 5. Extract Feature Contributions
        feature_contributions = []
        for i, feat_name in enumerate(self.feature_names):
            human_label = format_human_readable_feature_name(feat_name)
            s_val = float(shap_vec[i])
            feat_val = float(X_proc[0, i])
            
            feature_contributions.append({
                "feature_index": i,
                "feature_name": feat_name,
                "human_readable_name": human_label,
                "feature_value_scaled": round(feat_val, 4),
                "shap_value": round(s_val, 6),
                "contribution_direction": "positive_risk_increase" if s_val > 0 else ("negative_risk_decrease" if s_val < 0 else "neutral"),
                "magnitude": round(abs(s_val), 6),
            })

        # Sort by SHAP value descending (top positive) and ascending (top negative)
        pos_contributors = [f for f in sorted(feature_contributions, key=lambda x: x["shap_value"], reverse=True) if f["shap_value"] > 0][:top_k]
        neg_contributors = [f for f in sorted(feature_contributions, key=lambda x: x["shap_value"]) if f["shap_value"] < 0][:top_k]

        explanation_payload = {
            "modality": "tabular_lab",
            "task": "binary_30day_readmission",
            "model_type": "XGBoost",
            "shap_output_space": self.output_space,
            "prediction_summary": {
                "probability": round(prob, 4),
                "decision_threshold": self.threshold,
                "predicted_class": pred_class,
                "predicted_label": pred_label_str,
                "actual_label": int(actual_label) if actual_label is not None else None,
                "raw_margin": round(raw_margin, 4),
                "base_value": round(self.base_value, 4),
                "additivity_verified": bool(additivity_check),
            },
            "top_risk_increasing_factors": pos_contributors,
            "top_risk_decreasing_factors": neg_contributors,
            "all_ranked_contributions": sorted(feature_contributions, key=lambda x: x["magnitude"], reverse=True)[:top_k*2],
            "disclaimer": "AI-assisted analytical model contribution attribution aid. Clinical review required.",
        }

        return explanation_payload

    def explain_dataset(self, X_proc: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values matrix for a preprocessed feature matrix.

        Args:
            X_proc: Preprocessed feature matrix (N, num_features).

        Returns:
            Numpy array of SHAP values (N, num_features).
        """
        shap_vals = self.explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        return np.asarray(shap_vals, dtype=np.float32)
