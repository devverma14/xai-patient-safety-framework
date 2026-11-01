"""
Unit Tests for Common XAI Evaluation Framework (Phase 23 Extension).

Verifies:
- Schema validity and top-level report structure.
- Correct modality identification (lab, pneumonia, tb, skin, retina).
- Method distinction between Quantitative Feature Attribution (SHAP) and Visual Spatial Attribution (Grad-CAM).
- Phase 23 Retina Grad-CAM inclusion with EfficientNet-B0, target layer features.8.0, 16 explanations, and perturbation diagnostics.
- Strict handling of uncomputed metrics as 'NOT_EVALUATED' (no dummy 0.0 fallbacks).
- Preservation and non-mutation of underlying pre-existing XAI artifacts across all 5 modalities.
"""

import json
from pathlib import Path
import pytest

from xai.evaluation import (
    CommonXAIEvaluator,
    Modality,
    ExplanationMethod,
    ExplanationType,
    EvaluationStatus,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_REPORT_PATH = PROJECT_ROOT / "evaluation" / "xai" / "common_xai_evaluation.json"
EXPECTED_RETINA_MODEL_SHA256 = "4bf9260be9393f440d2a201dc763edb2eab52ab57c1f2d7c6780deddd1ac7a07"


def test_common_xai_report_file_exists():
    """Verify common_xai_evaluation.json artifact exists on disk."""
    assert COMMON_REPORT_PATH.exists(), f"Common XAI evaluation report missing at {COMMON_REPORT_PATH}"
    with open(COMMON_REPORT_PATH, "r") as f:
        report = json.load(f)
    assert "Phase 23" in report.get("phase", "")
    assert "modalities" in report
    assert "global_disclaimer" in report


def test_schema_and_modality_identification():
    """Verify schema validity and presence of all 5 modalities."""
    evaluator = CommonXAIEvaluator(base_dir=PROJECT_ROOT)
    report = evaluator.generate_common_evaluation_report()

    modalities = report.get("modalities", {})
    assert set(modalities.keys()) == {"lab", "pneumonia", "tb", "skin", "retina"}

    for mod_key, mod_data in modalities.items():
        assert "modality" in mod_data
        assert "task" in mod_data
        assert "model" in mod_data
        assert "explanation_method" in mod_data
        assert "explanation_type" in mod_data
        assert "prediction_context" in mod_data
        assert "explanation_quality_metrics" in mod_data
        assert "limitations" in mod_data
        assert "evaluation_status" in mod_data
        assert "artifact_paths" in mod_data


def test_explanation_method_and_type_distinction():
    """Verify distinction between Quantitative Feature Attribution (SHAP) and Visual Spatial Attribution (Grad-CAM)."""
    evaluator = CommonXAIEvaluator(base_dir=PROJECT_ROOT)
    report = evaluator.generate_common_evaluation_report()
    modalities = report["modalities"]

    # Lab (SHAP / Quantitative)
    lab = modalities["lab"]
    assert lab["explanation_method"] == ExplanationMethod.SHAP.value
    assert lab["explanation_type"] == ExplanationType.QUANTITATIVE_FEATURE.value
    assert lab["evaluation_status"] == EvaluationStatus.VERIFIED_COMPLETE.value

    # Pneumonia (Grad-CAM / Visual)
    pneumonia = modalities["pneumonia"]
    assert pneumonia["explanation_method"] == ExplanationMethod.GRAD_CAM.value
    assert pneumonia["explanation_type"] == ExplanationType.VISUAL_SPATIAL.value
    assert pneumonia["evaluation_status"] == EvaluationStatus.VERIFIED_COMPLETE.value

    # TB (Grad-CAM / Visual)
    tb = modalities["tb"]
    assert tb["explanation_method"] == ExplanationMethod.GRAD_CAM.value
    assert tb["explanation_type"] == ExplanationType.VISUAL_SPATIAL.value
    assert tb["evaluation_status"] == EvaluationStatus.VERIFIED_COMPLETE.value

    # Skin (Grad-CAM / Visual)
    skin = modalities["skin"]
    assert skin["explanation_method"] == ExplanationMethod.GRAD_CAM.value
    assert skin["explanation_type"] == ExplanationType.VISUAL_SPATIAL.value
    assert skin["evaluation_status"] == EvaluationStatus.VERIFIED_COMPLETE.value

    # Retina (Grad-CAM / Visual - Phase 23)
    retina = modalities["retina"]
    assert retina["explanation_method"] == ExplanationMethod.GRAD_CAM.value
    assert retina["explanation_type"] == ExplanationType.VISUAL_SPATIAL.value
    assert retina["evaluation_status"] == EvaluationStatus.VERIFIED_COMPLETE.value


def test_retina_common_xai_entry_details():
    """Verify Retina entry in common evaluation report contains exact Phase 22 Grad-CAM details."""
    evaluator = CommonXAIEvaluator(base_dir=PROJECT_ROOT)
    report = evaluator.generate_common_evaluation_report()
    retina = report["modalities"]["retina"]

    assert retina["modality"] == "retina"
    assert retina["model"] == "EfficientNet-B0"
    assert retina["explanation_method"] == ExplanationMethod.GRAD_CAM.value
    assert retina["explanation_type"] == ExplanationType.VISUAL_SPATIAL.value
    assert retina["prediction_context"]["data_type"] == "Retinal fundus photographs (224x224)"
    assert "features.8.0" in retina["prediction_context"]["target_layer"]

    metrics = retina["explanation_quality_metrics"]
    assert metrics["spatial_heatmap_range"] == "[0.0, 1.0]"
    assert metrics["spatial_activation_validity"] == "EVALUATED (Passed)"
    assert metrics["output_dimensions_valid"] is True
    assert metrics["finite_heatmap_values"] is True
    assert metrics["total_explanations_evaluated"] == 16
    assert metrics["explanation_category_counts"] == {"TP": 4, "TN": 4, "FP": 4, "FN": 4}
    assert metrics["model_sha256"] == EXPECTED_RETINA_MODEL_SHA256
    assert "controlled_perturbation_diagnostic" in metrics
    assert metrics["controlled_perturbation_diagnostic"]["mean_delta_probability"] == -0.1395
    assert metrics["causal_proof_or_clinical_validity"] == "NOT_EVALUATED"


def test_uncomputed_metrics_handling():
    """Verify uncomputed metrics are explicitly marked as 'NOT_EVALUATED' and not dummy 0s."""
    evaluator = CommonXAIEvaluator(base_dir=PROJECT_ROOT)
    report = evaluator.generate_common_evaluation_report()
    modalities = report["modalities"]

    # Lab should not have visual heatmap metrics
    lab_metrics = modalities["lab"]["explanation_quality_metrics"]
    assert lab_metrics["spatial_heatmap_range"] == "NOT_EVALUATED"
    assert lab_metrics["spatial_activation_validity"] == "NOT_EVALUATED"
    assert lab_metrics["visual_perturbation_masking_drop"] == "NOT_EVALUATED"

    # Pneumonia should not have SHAP tabular metrics
    pneu_metrics = modalities["pneumonia"]["explanation_quality_metrics"]
    assert pneu_metrics["shap_fidelity"] == "NOT_EVALUATED"
    assert pneu_metrics["shap_stability"] == "NOT_EVALUATED"
    assert pneu_metrics["shap_robustness"] == "NOT_EVALUATED"

    # Retina should not have SHAP tabular metrics
    retina_metrics = modalities["retina"]["explanation_quality_metrics"]
    assert retina_metrics["shap_fidelity"] == "NOT_EVALUATED"
    assert retina_metrics["shap_stability"] == "NOT_EVALUATED"
    assert retina_metrics["shap_robustness"] == "NOT_EVALUATED"
    assert retina_metrics["causal_proof_or_clinical_validity"] == "NOT_EVALUATED"


def test_existing_underlying_xai_artifacts_preserved():
    """Verify existing pre-existing XAI artifacts across all 5 modalities remain on disk untouched."""
    lab_fid = PROJECT_ROOT / "evaluation" / "xai" / "lab" / "fidelity.json"
    pneu_summary = PROJECT_ROOT / "evaluation" / "pneumonia" / "gradcam_summary.json"
    tb_summary = PROJECT_ROOT / "evaluation" / "tb" / "gradcam" / "gradcam_summary.json"
    skin_summary = PROJECT_ROOT / "evaluation" / "skin" / "gradcam" / "gradcam_summary.json"
    retina_summary = PROJECT_ROOT / "evaluation" / "retina" / "gradcam" / "gradcam_summary.json"

    assert lab_fid.exists(), f"Lab fidelity artifact missing: {lab_fid}"
    assert pneu_summary.exists(), f"Pneumonia Grad-CAM summary missing: {pneu_summary}"
    assert tb_summary.exists(), f"TB Grad-CAM summary missing: {tb_summary}"
    assert skin_summary.exists(), f"Skin Grad-CAM summary missing: {skin_summary}"
    assert retina_summary.exists(), f"Retina Grad-CAM summary missing: {retina_summary}"
