import json
from pathlib import Path
import pandas as pd
import pytest

from preprocessing.images.dataset_specific import (
    APTOS2019DataSpec,
    map_aptos_binary_labels,
    parse_aptos_dataset,
    split_aptos_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON_PATH = PROJECT_ROOT / "evaluation" / "retina" / "retina_dataset_audit.json"
RETINA_DIR = PROJECT_ROOT / "datasets" / "retina"


def test_aptos_data_spec():
    """Verify APTOS 2019 specification constants and class mappings."""
    assert APTOS2019DataSpec.RAW_CLASSES == [0, 1, 2, 3, 4]
    assert len(APTOS2019DataSpec.RAW_CLASS_NAMES) == 5
    assert APTOS2019DataSpec.RAW_CLASS_NAMES[0] == "No DR"
    assert APTOS2019DataSpec.RAW_CLASS_NAMES[4] == "Proliferative DR"
    assert len(APTOS2019DataSpec.BINARY_CLASSES) == 2
    assert APTOS2019DataSpec.NUM_CLASSES_V1 == 2


def test_map_aptos_binary_labels():
    """Verify binary mapping: Grade 0 -> 0 (No DR), Grades 1-4 -> 1 (DR Present)."""
    df_synth = pd.DataFrame({
        "id_code": ["img0", "img1", "img2", "img3", "img4"],
        "diagnosis": [0, 1, 2, 3, 4],
    })
    df_mapped = map_aptos_binary_labels(df_synth)

    assert "binary_label" in df_mapped.columns
    assert "binary_class_name" in df_mapped.columns
    assert list(df_mapped["binary_label"]) == [0, 1, 1, 1, 1]
    assert df_mapped.loc[0, "binary_class_name"] == "No Diabetic Retinopathy"
    assert df_mapped.loc[1, "binary_class_name"] == "Diabetic Retinopathy Present"


def test_split_aptos_dataset_stratified():
    """Verify stratified split logic on synthetic APTOS DataFrame."""
    ids = [f"id_{i}" for i in range(100)]
    diags = [0] * 50 + [2] * 50
    df_synth = pd.DataFrame({"id_code": ids, "diagnosis": diags})
    df_synth = map_aptos_binary_labels(df_synth)

    train_df, val_df, test_df = split_aptos_dataset(
        df_synth, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    assert len(train_df) == 70
    assert len(val_df) + len(test_df) == 30
    assert abs(len(val_df) - 15) <= 1
    assert abs(len(test_df) - 15) <= 1

    # Verify zero index overlap across splits
    tr_ids = set(train_df["id_code"])
    va_ids = set(val_df["id_code"])
    te_ids = set(test_df["id_code"])

    assert len(tr_ids.intersection(va_ids)) == 0
    assert len(tr_ids.intersection(te_ids)) == 0
    assert len(va_ids.intersection(te_ids)) == 0

    # Verify class proportions in splits
    assert (train_df["binary_label"] == 0).sum() == 35
    assert (train_df["binary_label"] == 1).sum() == 35


def test_parse_aptos_dataset_local():
    """Verify parse_aptos_dataset on local acquired APTOS dataset."""
    df, audit_info = parse_aptos_dataset(str(RETINA_DIR))
    assert df is not None
    assert len(df) == 3662
    assert audit_info["status"] == "VERIFIED_READY"
    assert audit_info["readable_images"] == 3662


def test_retina_audit_artifact_exists():
    """Verify evaluation/retina/retina_dataset_audit.json exists and contains correct metadata."""
    assert AUDIT_JSON_PATH.exists(), f"Audit artifact missing at {AUDIT_JSON_PATH}"
    with open(AUDIT_JSON_PATH, "r") as f:
        audit = json.load(f)

    assert audit.get("task") == "Diabetic Retinopathy Classification (Binary V1)"
    assert audit.get("dataset_name") == "APTOS 2019 Blindness Detection"
    assert audit.get("acquisition", {}).get("status") == "VERIFIED_COMPLETED"
    assert audit.get("local_inventory", {}).get("total_images_found") == 3662
    assert audit.get("local_inventory", {}).get("corrupted_images") == 0
    assert audit.get("measured_local_statistics", {}).get("total_train_images") == 3662
    assert len(audit.get("proposed_candidate_architectures", [])) == 3
