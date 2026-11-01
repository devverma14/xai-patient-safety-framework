"""
Unit and Integration Test Suite for Phase 6 Lab SHAP XAI Engine.
Verifies model loading, preprocessor loading, SHAP additivity, prediction-explanation consistency,
decision threshold (0.35), PII exclusion, and feature mapping.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from preprocessing.lab import patient_level_split, LabPreprocessor
from xai.shap import LabSHAPExplainer, format_human_readable_feature_name


class TestLabSHAPXAIEngine(unittest.TestCase):
    """Test suite verifying Phase 6 Lab SHAP XAI Engine requirements."""

    @classmethod
    def setUpClass(cls):
        """Set up explainer and sample validation data."""
        model_p = os.path.join("models", "lab", "final", "xgb_model.joblib")
        prep_p = os.path.join("models", "lab", "final", "preprocessor.joblib")
        meta_p = os.path.join("models", "lab", "final", "model_metadata.json")

        cls.explainer = LabSHAPExplainer(model_path=model_p, preprocessor_path=prep_p, metadata_path=meta_p)

        csv_path = os.path.join("datasets", "lab", "diabetic_data.csv")
        raw_df = pd.read_csv(csv_path)
        expired_ids = [11, 19, 20, 21]
        df_clean = raw_df[~raw_df["discharge_disposition_id"].isin(expired_ids)].copy().reset_index(drop=True)
        df_clean["target_30day"] = (df_clean["readmitted"] == "<30").astype(int)

        train_df, val_df, test_df = patient_level_split(
            df_clean, group_col="patient_nbr", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
        )
        cls.val_sample_df = val_df.head(10).copy()

    def test_model_and_preprocessor_loading(self):
        """Test 1-3: Verify model, preprocessor, and metadata load correctly."""
        self.assertIsNotNone(self.explainer.model, "XGBoost model failed to load!")
        self.assertIsNotNone(self.explainer.preprocessor, "Preprocessor failed to load!")
        self.assertEqual(self.explainer.threshold, 0.35, f"Expected threshold 0.35, got {self.explainer.threshold}")

    def test_feature_dimensions(self):
        """Test 4-5 & 8: Verify feature count and SHAP output dimension match."""
        feature_names = self.explainer.get_feature_names()
        self.assertTrue(len(feature_names) > 0, "Feature names list is empty!")

        X_proc = self.explainer.preprocessor.transform(self.val_sample_df.head(1))
        self.assertEqual(X_proc.shape[1], len(feature_names), "Processed feature dimension mismatch!")

        shap_matrix = self.explainer.explain_dataset(X_proc)
        self.assertEqual(shap_matrix.shape, (1, len(feature_names)), f"SHAP matrix shape {shap_matrix.shape} mismatch!")

    def test_prediction_explanation_consistency(self):
        """Test 6-7: Verify prediction before explanation equals prediction during explanation."""
        sample_row = self.val_sample_df.head(1)
        X_proc = self.explainer.preprocessor.transform(sample_row)

        direct_prob = float(self.explainer.model.predict_proba(X_proc)[0, 1])
        direct_class = 1 if direct_prob >= 0.35 else 0

        local_exp = self.explainer.explain_local_encounter(sample_row, top_k=5)
        exp_summary = local_exp["prediction_summary"]

        self.assertTrue(np.isclose(direct_prob, exp_summary["probability"], atol=1e-4), "Probability mismatch!")
        self.assertEqual(direct_class, exp_summary["predicted_class"], "Class prediction mismatch!")
        self.assertEqual(exp_summary["decision_threshold"], 0.35, "Threshold in explanation is not 0.35!")

    def test_shap_additivity_check(self):
        """Test 10: Verify SHAP additivity: base_value + sum(shap_values) == margin_prediction."""
        X_proc = self.explainer.preprocessor.transform(self.val_sample_df)
        self.assertTrue(self.explainer.verify_additivity(X_proc, tol=0.05), "SHAP additivity test failed!")

    def test_pii_exclusion(self):
        """Test 9: Verify no patient identifiers (patient_nbr, encounter_id) appear in SHAP outputs."""
        local_exp = self.explainer.explain_local_encounter(self.val_sample_df.head(1), top_k=10)

        for item in local_exp["top_risk_increasing_factors"] + local_exp["top_risk_decreasing_factors"]:
            fname = item["feature_name"].lower()
            hname = item["human_readable_name"].lower()
            self.assertNotIn("patient_nbr", fname, "PII 'patient_nbr' exposed in SHAP feature name!")
            self.assertNotIn("patient_nbr", hname, "PII 'patient_nbr' exposed in human-readable name!")
            self.assertNotIn("encounter_id", fname, "PII 'encounter_id' exposed in SHAP feature name!")
            self.assertNotIn("encounter_id", hname, "PII 'encounter_id' exposed in human-readable name!")

    def test_deterministic_feature_mapping(self):
        """Test 11: Verify feature mapping is deterministic."""
        name1 = format_human_readable_feature_name("num_inpatient")
        name2 = format_human_readable_feature_name("num_inpatient")
        self.assertEqual(name1, name2)
        self.assertEqual(name1, "Inpatient Visits in Prior Year")

        cat1 = format_human_readable_feature_name("medical_specialty_Cardiology")
        self.assertEqual(cat1, "Admitting Medical Specialty: Cardiology")


if __name__ == "__main__":
    unittest.main()
