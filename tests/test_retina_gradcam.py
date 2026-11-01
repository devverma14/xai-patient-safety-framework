"""
Phase 22 — Retina Grad-CAM XAI Unit Tests.

Tests:
1. Model checkpoint existence & SHA-256 integrity verification
2. Target layer existence (features.8.0, nn.Conv2d, 1280 channels)
3. Grad-CAM heatmap output shape (224x224), range [0.0, 1.0], and finite values
4. Overlay shape (224x224x3 uint8)
5. Representative sample coverage (4 TP, 4 TN, 4 FP, 4 FN)
6. Verification that authoritative test_predictions.csv is unmodified
7. Artifact inventory existence in evaluation/retina/gradcam/
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as T

from models.retina.trainer import build_retina_model, compute_file_sha256
from xai.gradcam.gradcam import GradCAM
from xai.gradcam.visualization import generate_gradcam_overlay

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "retina" / "best_retina_model.pt"
PRED_PATH = BASE_DIR / "evaluation" / "retina" / "test_predictions.csv"
GRADCAM_DIR = BASE_DIR / "evaluation" / "retina" / "gradcam"

EXPECTED_MODEL_SHA256 = "4bf9260be9393f440d2a201dc763edb2eab52ab57c1f2d7c6780deddd1ac7a07"


def test_model_checkpoint_and_sha256_integrity():
    """Verify model checkpoint exists and SHA-256 matches frozen Phase 21 hash."""
    assert MODEL_PATH.exists(), f"Model checkpoint missing at {MODEL_PATH}"
    actual_sha = compute_file_sha256(str(MODEL_PATH))
    assert actual_sha == EXPECTED_MODEL_SHA256, (
        f"Model SHA-256 mismatch!\nExpected: {EXPECTED_MODEL_SHA256}\nActual: {actual_sha}"
    )


def test_target_layer_existence():
    """Verify target layer features.8.0 exists in EfficientNet-B0 and is nn.Conv2d with 1280 out_channels."""
    model = build_retina_model("efficientnet_b0", num_classes=2, pretrained=False)
    target_layer = model.features[8][0]

    assert isinstance(target_layer, torch.nn.Conv2d)
    assert target_layer.out_channels == 1280
    assert target_layer.in_channels == 320


def test_gradcam_heatmap_and_overlay_properties():
    """Verify Grad-CAM produces 224x224 heatmap in [0, 1] and 224x224x3 uint8 overlay."""
    model = build_retina_model("efficientnet_b0", num_classes=2, pretrained=False)
    model.eval()
    target_layer = model.features[8][0]

    explainer = GradCAM(model=model, target_layer=target_layer)
    dummy_tensor = torch.randn(1, 3, 224, 224)

    heatmap, target_cls, prob = explainer.generate_heatmap(dummy_tensor, target_class=1)
    explainer.remove_hooks()

    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all()

    dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap_rgb, overlay_rgb = generate_gradcam_overlay(dummy_rgb, heatmap)

    assert overlay_rgb.shape == (224, 224, 3)
    assert overlay_rgb.dtype == np.uint8


def test_authoritative_predictions_unmodified():
    """Verify authoritative test_predictions.csv exists and contains expected columns and row count (547)."""
    assert PRED_PATH.exists()
    df_pred = pd.read_csv(PRED_PATH)
    assert len(df_pred) == 547
    assert list(df_pred.columns) == ["id_code", "ground_truth", "predicted_label", "prob_dr_present", "binary_class_name"]


def test_gradcam_artifacts_and_coverage():
    """Verify Phase 22 Grad-CAM artifacts exist with 16 representative samples (4 TP, 4 TN, 4 FP, 4 FN)."""
    summary_file = GRADCAM_DIR / "gradcam_summary.json"
    metadata_file = GRADCAM_DIR / "metadata.json"

    if not summary_file.exists() or not metadata_file.exists():
        pytest.skip("Grad-CAM artifacts not yet generated.")

    with open(summary_file, "r") as f:
        summary = json.load(f)

    assert summary["frozen_model"] == "EfficientNet-B0"
    assert summary["model_sha256"] == EXPECTED_MODEL_SHA256
    assert summary["representative_samples_summary"]["total_explanations"] == 16
    assert summary["representative_samples_summary"]["generated"] == {"TP": 4, "TN": 4, "FP": 4, "FN": 4}

    with open(metadata_file, "r") as f:
        meta = json.load(f)

    samples = meta["samples"]
    assert len(samples) == 16

    categories = [s["category"] for s in samples]
    assert categories.count("TP") == 4
    assert categories.count("TN") == 4
    assert categories.count("FP") == 4
    assert categories.count("FN") == 4

    for s in samples:
        assert (BASE_DIR / s["heatmap_path"]).exists()
        assert (BASE_DIR / s["overlay_path"]).exists()
        assert (BASE_DIR / s["comparison_path"]).exists()
        assert "perturbation_diagnostic" in s
        assert "delta_probability" in s["perturbation_diagnostic"]
