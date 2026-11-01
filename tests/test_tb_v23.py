"""
Unit & Integration Tests for Phase V2.3 - Controlled Multi-Site TB Model Training & Governance
"""

import os
import json
import pytest
import torch
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_MANIFEST_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "v23_split_manifest.json"
EXPERIMENTS_SUMMARY_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "v23_experiments_summary.json"
FROZEN_BASELINE_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "frozen_v1_baseline_metrics.json"
VAL_MATRIX_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "validation_comparison_matrix.csv"
TEST_MATRIX_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "test_comparison_matrix.csv"
ROBUSTNESS_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "robustness_metrics.json"
GRADCAM_SUMMARY_PATH = BASE_DIR / "evaluation" / "tb" / "v23" / "gradcam" / "gradcam_validation_summary.json"
PROD_MODEL_PATH = BASE_DIR / "models" / "tb" / "best_tb_model.pt"
PROD_META_PATH = BASE_DIR / "models" / "tb" / "best_tb_model_metadata.json"


def test_split_manifest_governance_and_leakage_isolation():
    """Verify V2.3 split manifest exists and guarantees zero image hash and patient overlap across splits."""
    assert SPLIT_MANIFEST_PATH.exists(), "v23_split_manifest.json missing"
    with open(SPLIT_MANIFEST_PATH, "r") as f:
        data = json.load(f)

    train_hashes = set(data["split_hashes"]["train"])
    val_hashes = set(data["split_hashes"]["val"])
    test_hashes = set(data["split_hashes"]["test"])

    # Zero duplicate hash overlap across partitions
    assert len(train_hashes.intersection(val_hashes)) == 0, "Hash overlap detected between train and val!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Hash overlap detected between train and test!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Hash overlap detected between val and test!"

    train_pids = set(data["split_patients"]["train"])
    val_pids = set(data["split_patients"]["val"])
    test_pids = set(data["split_patients"]["test"])

    # Zero patient overlap across partitions
    assert len(train_pids.intersection(val_pids)) == 0, "Patient overlap detected between train and val!"
    assert len(train_pids.intersection(test_pids)) == 0, "Patient overlap detected between train and test!"
    assert len(val_pids.intersection(test_pids)) == 0, "Patient overlap detected between val and test!"

    assert data["summary"]["total_images"] == 800
    assert data["summary"]["train_count"] == 560
    assert data["summary"]["val_count"] == 120
    assert data["summary"]["test_count"] == 120


def test_frozen_baseline_metrics_integrity():
    """Verify frozen V1 production model baseline evaluation on new V2.3 held-out split is stored."""
    assert FROZEN_BASELINE_PATH.exists(), "frozen_v1_baseline_metrics.json missing"
    with open(FROZEN_BASELINE_PATH, "r") as f:
        metrics = json.load(f)

    assert "frozen_v1_held_out_test" in metrics
    test_m = metrics["frozen_v1_held_out_test"]
    assert test_m["threshold"] == 0.50
    assert "roc_auc" in test_m
    assert "site_metrics" in test_m
    assert "montgomery" in test_m["site_metrics"]
    assert "shenzhen" in test_m["site_metrics"]


def test_v23_experiments_matrices_existence():
    """Verify validation and test comparison matrices exist and contain all 4 experiments."""
    assert VAL_MATRIX_PATH.exists(), "validation_comparison_matrix.csv missing"
    assert TEST_MATRIX_PATH.exists(), "test_comparison_matrix.csv missing"

    val_df = pd.read_csv(VAL_MATRIX_PATH)
    test_df = pd.read_csv(TEST_MATRIX_PATH)

    exp_ids = {"V23_E01", "V23_E02", "V23_E03", "V23_E04"}
    assert all(any(exp_id in c for c in val_df["Candidate"].values) for exp_id in exp_ids)
    assert exp_ids.issubset(set(test_df["Candidate"].values))


def test_production_checkpoint_and_metadata_governance():
    """Verify updated production model checkpoint and metadata reflect the selected V2.3 candidate."""
    assert PROD_MODEL_PATH.exists(), "best_tb_model.pt missing"
    assert PROD_META_PATH.exists(), "best_tb_model_metadata.json missing"

    with open(PROD_META_PATH, "r") as f:
        meta = json.load(f)

    assert meta["threshold"] == 0.50
    assert meta["phase"] == "V2.3"
    assert meta["status"] == "MEANINGFUL_IMPROVEMENT"
    assert meta["architecture"] == "densenet121"
    assert "sha256" in meta


def test_densenet121_gradcam_target_layer_validity():
    """Verify DenseNet121 target layer exists for Grad-CAM generation."""
    from torchvision.models import densenet121
    model = densenet121(weights=None)
    
    target_layer_name = "features.denseblock4.denselayer16.conv2"
    target_module = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_module = module
            break

    assert target_module is not None, f"Target layer '{target_layer_name}' not found in DenseNet121!"


def test_tb_model_registry_and_api_architecture_metadata():
    """Verify TB model architecture propagates as DenseNet121 from registry and API."""
    from backend.app.services.model_registry import model_registry
    from backend.app.main import app
    from fastapi.testclient import TestClient

    tb_meta = model_registry.get_metadata("xray", "tb")
    assert tb_meta.architecture == "DenseNet121"
    assert tb_meta.model_name == "DenseNet121"

    client = TestClient(app)
    # Check status endpoint
    status_resp = client.get("/api/v1/xray/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["model_architecture"] == "DenseNet121"
    assert status_data["task_architectures"]["tb"] == "DenseNet121"

    # Check analyze endpoint
    import io
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill=(200, 200, 200))
    draw.line([0, 0, 224, 224], fill=(50, 50, 50), width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    analyze_resp = client.post(
        "/api/v1/xray/analyze?task=tb",
        files={"file": ("tb_test.jpg", buf, "image/jpeg")},
    )
    assert analyze_resp.status_code == 200
    analyze_data = analyze_resp.json()
    pred = analyze_data["prediction"]
    assert pred["model_name"] == "DenseNet121"
    assert pred["model_architecture"] == "DenseNet121"
    assert analyze_data["explanation"]["details"]["target_layer"] == "features.denseblock4.denselayer16.conv2"
