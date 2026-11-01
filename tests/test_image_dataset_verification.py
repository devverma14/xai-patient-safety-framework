"""
Unit and Verification Test Suite for Phase 9 Image Dataset Specifications, Label Mappings,
Lesion-Level Leakage Prevention Splitting, and Inventory Audit.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from preprocessing.images.dataset_specific import (
    PneumoniaDataSpec,
    TuberculosisDataSpec,
    HAM10000DataSpec,
    APTOS2019DataSpec,
    map_aptos_binary_labels,
    split_ham10000_by_lesion,
)
from experiments.datasets.run_phase9_experiments import (
    audit_pneumonia_dataset,
    audit_tb_dataset,
    audit_ham10000_dataset,
    audit_aptos2019_dataset,
)


class TestPhase9ImageDatasetVerification(unittest.TestCase):
    """Test suite verifying Phase 9 image dataset specifications and utilities."""

    def test_01_pneumonia_spec(self):
        """Test Pneumonia dataset specification constants."""
        self.assertEqual(len(PneumoniaDataSpec.CLASSES), 2)
        self.assertEqual(PneumoniaDataSpec.CLASS_TO_IDX["NORMAL"], 0)
        self.assertEqual(PneumoniaDataSpec.CLASS_TO_IDX["PNEUMONIA"], 1)

    def test_02_tb_spec(self):
        """Test Tuberculosis dataset specification constants."""
        self.assertEqual(len(TuberculosisDataSpec.CLASSES), 2)
        self.assertIn("montgomery", TuberculosisDataSpec.SUBSETS)
        self.assertIn("shenzhen", TuberculosisDataSpec.SUBSETS)

    def test_03_ham10000_spec(self):
        """Test HAM10000 7-class specification constants."""
        self.assertEqual(HAM10000DataSpec.NUM_CLASSES, 7)
        self.assertIn("mel", HAM10000DataSpec.CLASSES)
        self.assertIn("nv", HAM10000DataSpec.CLASSES)
        self.assertEqual(len(HAM10000DataSpec.CLASS_NAMES), 7)

    def test_04_aptos_binary_label_mapping(self):
        """Test APTOS 2019 5-grade to binary DR label mapping."""
        df_dummy = pd.DataFrame({"id_code": ["a", "b", "c", "d", "e"], "diagnosis": [0, 1, 2, 3, 4]})
        df_mapped = map_aptos_binary_labels(df_dummy)
        
        self.assertIn("binary_label", df_mapped.columns)
        self.assertEqual(df_mapped.iloc[0]["binary_label"], 0)  # Grade 0 -> 0
        self.assertEqual(df_mapped.iloc[1]["binary_label"], 1)  # Grade 1 -> 1
        self.assertEqual(df_mapped.iloc[2]["binary_label"], 1)  # Grade 2 -> 1
        self.assertEqual(df_mapped.iloc[3]["binary_label"], 1)  # Grade 3 -> 1
        self.assertEqual(df_mapped.iloc[4]["binary_label"], 1)  # Grade 4 -> 1

    def test_05_ham10000_lesion_id_grouped_split(self):
        """Test HAM10000 lesion_id grouped split preventing lesion leakage."""
        # Create dummy metadata with 100 images across 50 unique lesions
        lesions = [f"HAM_lesion_{i//2}" for i in range(100)]
        df_dummy = pd.DataFrame({
            "image_id": [f"ISIC_{i:05d}" for i in range(100)],
            "lesion_id": lesions,
            "dx": ["nv"] * 100,
        })
        
        train_df, val_df, test_df = split_ham10000_by_lesion(
            df_dummy, lesion_id_col="lesion_id", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
        )
        
        train_lesions = set(train_df["lesion_id"].unique())
        val_lesions = set(val_df["lesion_id"].unique())
        test_lesions = set(test_df["lesion_id"].unique())
        
        # Verify ZERO lesion ID overlap
        self.assertEqual(len(train_lesions.intersection(val_lesions)), 0)
        self.assertEqual(len(train_lesions.intersection(test_lesions)), 0)
        self.assertEqual(len(val_lesions.intersection(test_lesions)), 0)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 100)

    def test_06_audit_functions(self):
        """Test programmatic dataset audit functions."""
        pneumonia = audit_pneumonia_dataset("datasets")
        self.assertIn("local_status", pneumonia)
        self.assertEqual(pneumonia["target_task"], "Pneumonia vs Normal (Binary Classification)")

        tb = audit_tb_dataset("datasets")
        self.assertIn("local_status", tb)
        self.assertIn("Montgomery County Chest X-ray Set", tb["expected_datasets"][0])

        ham = audit_ham10000_dataset("datasets")
        self.assertIn("local_status", ham)
        self.assertEqual(ham["expected_classes"], HAM10000DataSpec.CLASSES)

        aptos = audit_aptos2019_dataset("datasets")
        self.assertIn("local_status", aptos)
        self.assertEqual(aptos["v1_binary_target_classes"], APTOS2019DataSpec.BINARY_CLASSES)


if __name__ == "__main__":
    unittest.main()
