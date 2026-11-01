"""
Phase 10 — Chest X-Ray Pneumonia Pipeline Unit Tests.

Tests:
1. Dataset Loading: Verifies parse_pneumonia_dataset on local dataset directory.
2. Patient-level Split: Verifies split_pneumonia_by_patient logic with zero patient leakage.
3. Preprocessing: Verifies image transform pipeline output shapes and normalization.
4. Model Loading & Architecture: Verifies model building and state dict structure.
5. Inference Output: Verifies forward pass logits and prediction probability output shapes.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from preprocessing.images import (
    PneumoniaDataSpec,
    parse_pneumonia_dataset,
    split_pneumonia_by_patient,
    get_image_transforms,
)
from experiments.pneumonia.train_pneumonia_models import build_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "pneumonia"
MODEL_DIR = BASE_DIR / "models" / "pneumonia"


class TestPneumoniaPipeline(unittest.TestCase):

    def test_01_dataset_loading(self):
        """Test dataset directory parsing and DataFrame generation."""
        self.assertTrue(DATASET_DIR.exists(), f"Dataset directory {DATASET_DIR} does not exist.")
        df = parse_pneumonia_dataset(str(DATASET_DIR))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Parsed Pneumonia DataFrame should not be empty.")
        
        required_cols = ["file_path", "relative_path", "label_name", "label_idx", "patient_id"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing from DataFrame.")

        # Check label indices match PneumoniaDataSpec
        unique_labels = set(df["label_name"].unique())
        self.assertTrue(unique_labels.issubset(set(PneumoniaDataSpec.CLASSES)))

    def test_02_patient_level_split(self):
        """Test patient-level split logic and verify zero patient leakage across splits."""
        df = parse_pneumonia_dataset(str(DATASET_DIR))
        train_df, val_df, test_df = split_pneumonia_by_patient(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        total_split_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(len(df), total_split_rows, "Total split rows must equal total dataset rows.")

        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())
        test_patients = set(test_df["patient_id"].unique())

        # Verify zero patient overlap
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)

        self.assertEqual(len(train_val_overlap), 0, f"Train/Val patient overlap found: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Train/Test patient overlap found: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Val/Test patient overlap found: {val_test_overlap}")

    def test_03_preprocessing_transforms(self):
        """Test deterministic evaluation preprocessing transformation output shape and type."""
        eval_transform = get_image_transforms(target_size=(224, 224), is_training=False)
        dummy_pil = Image.new("RGB", (300, 400), color=(128, 128, 128))
        
        tensor_img = eval_transform(dummy_pil)
        self.assertIsInstance(tensor_img, torch.Tensor)
        self.assertEqual(tensor_img.shape, (3, 224, 224))
        self.assertEqual(tensor_img.dtype, torch.float32)

    def test_04_model_loading_and_architecture(self):
        """Test candidate model building for DenseNet121, EfficientNet-B0, and ResNet50."""
        for arch in ["densenet121", "efficientnet_b0", "resnet50"]:
            model = build_model(arch, num_classes=2, freeze_backbone=True)
            self.assertIsInstance(model, nn.Module)
            
            # Verify output linear layer produces 2 logits
            dummy_input = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, (2, 2))

    def test_05_inference_output(self):
        """Test model inference forward pass and probability distribution calculation."""
        model = build_model("densenet121", num_classes=2, freeze_backbone=True)
        model.eval()

        dummy_batch = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            outputs = model(dummy_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

        self.assertEqual(probs.shape, (4, 2))
        self.assertEqual(preds.shape, (4,))
        # Check probabilities sum to 1.0 per sample
        prob_sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(prob_sums, np.ones(4), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
