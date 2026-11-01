"""
Phase 21 — Retina Model & Dataset Unit Tests.

Tests:
1. Binary label mapping (Grade 0 -> 0, Grades 1-4 -> 1)
2. Duplicate-hash partition integrity (0 cross-partition duplicate image hash overlap)
3. Split ratio & row count integrity
4. Preprocessing output dimensions and RGB channel handling
5. Deterministic validation/testing transforms
6. Candidate model instantiation & output shape (Batch Size, 2)
7. Probability range [0.0, 1.0] and metric calculation logic
8. Saved artifact presence & prediction label consistency
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as T

from preprocessing.images.dataset_specific import (
    APTOS2019DataSpec,
    parse_aptos_dataset,
    split_aptos_dataset,
    map_aptos_binary_labels,
)
from models.retina.trainer import (
    APTOSDataset,
    get_retina_transforms,
    build_retina_model,
    compute_retina_metrics,
    compute_file_sha256,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "retina"
MODEL_DIR = BASE_DIR / "models" / "retina"
EVAL_DIR = BASE_DIR / "evaluation" / "retina"


def test_binary_label_mapping():
    """Verify APTOS 5-grade diagnosis mapping to V1 binary labels."""
    df_sample = pd.DataFrame({
        "id_code": ["a", "b", "c", "d", "e"],
        "diagnosis": [0, 1, 2, 3, 4]
    })
    df_mapped = map_aptos_binary_labels(df_sample)

    assert "binary_label" in df_mapped.columns
    assert "binary_class_name" in df_mapped.columns
    assert list(df_mapped["binary_label"]) == [0, 1, 1, 1, 1]
    assert df_mapped.loc[0, "binary_class_name"] == "No Diabetic Retinopathy"
    assert df_mapped.loc[1, "binary_class_name"] == "Diabetic Retinopathy Present"


def test_duplicate_hash_isolated_split():
    """Verify split_aptos_dataset produces 0 cross-partition duplicate image hash overlap."""
    if not DATASET_DIR.exists() or not (DATASET_DIR / "train.csv").exists():
        pytest.skip("Local APTOS dataset not found.")

    df, audit_info = parse_aptos_dataset(str(DATASET_DIR))
    assert df is not None

    train_df, val_df, test_df = split_aptos_dataset(df, random_state=42)

    total_len = len(df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Verify 0 cross-partition file path overlap
    tr_paths = set(train_df["file_path"])
    va_paths = set(val_df["file_path"])
    te_paths = set(test_df["file_path"])

    assert len(tr_paths.intersection(va_paths)) == 0
    assert len(tr_paths.intersection(te_paths)) == 0
    assert len(va_paths.intersection(te_paths)) == 0

    # Verify 0 cross-partition SHA-256 duplicate image hash overlap
    if "img_hash" in train_df.columns:
        tr_hashes = set(train_df["img_hash"])
        va_hashes = set(val_df["img_hash"])
        te_hashes = set(test_df["img_hash"])

        assert len(tr_hashes.intersection(va_hashes)) == 0
        assert len(tr_hashes.intersection(te_hashes)) == 0
        assert len(va_hashes.intersection(te_hashes)) == 0


def test_preprocessing_transforms():
    """Verify Retina preprocessing output dimensions, RGB handling, and deterministic eval behavior."""
    train_tf = get_retina_transforms(input_size=(224, 224), is_training=True)
    eval_tf = get_retina_transforms(input_size=(224, 224), is_training=False)

    # Dummy RGB image tensor (3, 224, 224)
    dummy_img = torch.rand(3, 224, 224)

    t_out = train_tf(dummy_img)
    e_out1 = eval_tf(dummy_img)
    e_out2 = eval_tf(dummy_img)

    assert t_out.shape == (3, 224, 224)
    assert e_out1.shape == (3, 224, 224)
    # Validation/Testing transform must be strictly deterministic
    assert torch.equal(e_out1, e_out2)


def test_candidate_models_output_shape():
    """Verify all 3 candidate CNN models produce expected (Batch Size, 2) output logits."""
    dummy_input = torch.randn(2, 3, 224, 224)

    for arch in ["densenet121", "efficientnet_b0", "resnet50"]:
        model = build_retina_model(arch, num_classes=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 2), f"{arch} output shape mismatch: {output.shape}"


def test_metrics_calculation():
    """Verify compute_retina_metrics returns valid probabilities, sensitivity, specificity, F1, and AUC."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6])

    metrics = compute_retina_metrics(y_true, y_prob, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "dr_sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "balanced_accuracy" in metrics
    assert "confusion_matrix" in metrics

    # Probabilities range check
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["dr_sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0

    # Specificity = 2/3 (0.6667), Sensitivity = 3/3 (1.0)
    assert metrics["dr_sensitivity"] == 1.0
    assert round(metrics["specificity"], 2) == 0.67


def test_artifacts_existence_and_prediction_consistency():
    """Verify Phase 21 model and evaluation artifacts exist and predictions are consistent."""
    model_file = MODEL_DIR / "best_retina_model.pt"
    meta_file = MODEL_DIR / "model_metadata.json"
    pred_file = EVAL_DIR / "test_predictions.csv"
    cm_file = EVAL_DIR / "confusion_matrix.png"

    if not model_file.exists():
        pytest.skip("Retina training artifacts not yet generated.")

    assert model_file.exists()
    assert meta_file.exists()
    assert pred_file.exists()
    assert cm_file.exists()

    with open(meta_file, "r") as f:
        meta = json.load(f)

    assert meta["task"] == "Binary Diabetic Retinopathy Classification"
    assert "selected_architecture" in meta
    assert "model_sha256" in meta

    df_pred = pd.read_csv(pred_file)
    assert "id_code" in df_pred.columns
    assert "ground_truth" in df_pred.columns
    assert "predicted_label" in df_pred.columns
    assert "prob_dr_present" in df_pred.columns

    # Verify probabilities are within [0.0, 1.0]
    assert (df_pred["prob_dr_present"] >= 0.0).all()
    assert (df_pred["prob_dr_present"] <= 1.0).all()

    # Verify prediction label consistency (prob >= 0.5 -> 1 else 0)
    expected_pred = (df_pred["prob_dr_present"] >= 0.5).astype(int)
    assert (df_pred["predicted_label"] == expected_pred).all()
