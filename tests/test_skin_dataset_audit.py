"""
Unit tests for Phase 15 HAM10000 Skin Lesion Dataset Audit & Split Strategy.
"""

import os
import unittest
import pandas as pd
from PIL import Image

from preprocessing.images.dataset_specific import (
    HAM10000DataSpec,
    parse_ham10000_dataset,
    split_ham10000_by_lesion,
)


class TestPhase15SkinDatasetAudit(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.skin_dir = os.path.join(self.root_dir, "datasets", "skin")
        self.metadata_path = os.path.join(self.skin_dir, "HAM10000_metadata.csv")

    def test_01_ham10000_data_spec_constants(self):
        """Test HAM10000 dataset specification constants."""
        self.assertEqual(HAM10000DataSpec.NUM_CLASSES, 7)
        self.assertEqual(len(HAM10000DataSpec.CLASSES), 7)
        self.assertEqual(
            HAM10000DataSpec.CLASSES,
            ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
        )
        self.assertEqual(HAM10000DataSpec.CLASS_NAMES["mel"], "Melanoma")
        self.assertEqual(HAM10000DataSpec.CLASS_TO_IDX["nv"], 0)
        self.assertEqual(HAM10000DataSpec.IDX_TO_CLASS[0], "nv")

    def test_02_parse_ham10000_dataset_counts(self):
        """Test parsing HAM10000 directory and verifying 10,015 images match metadata."""
        self.assertTrue(os.path.exists(self.metadata_path), "HAM10000_metadata.csv missing.")

        df = parse_ham10000_dataset(self.skin_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10015, f"Expected 10,015 images, found {len(df)}")
        self.assertEqual(df["lesion_id"].nunique(), 7470, "Expected 7,470 unique lesions.")

        # Check required columns
        for col in ["file_path", "relative_path", "lesion_id", "image_id", "dx", "class_name", "label_idx"]:
            self.assertIn(col, df.columns)

    def test_03_image_readability_and_dimensions(self):
        """Test reading images, verifying RGB channels and 600x450 resolution."""
        df = parse_ham10000_dataset(self.skin_dir)

        # Inspect first 20 images
        sample_df = df.head(20)
        for _, row in sample_df.iterrows():
            fpath = row["file_path"]
            self.assertTrue(os.path.exists(fpath))
            with Image.open(fpath) as img:
                self.assertEqual(img.size, (600, 450))
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.format, "JPEG")

    def test_04_lesion_level_split_zero_leakage(self):
        """Test split_ham10000_by_lesion for zero lesion overlap across splits."""
        df = parse_ham10000_dataset(self.skin_dir)
        tr, va, te = split_ham10000_by_lesion(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

        self.assertEqual(len(tr) + len(va) + len(te), 10015)

        tr_lesions = set(tr["lesion_id"])
        va_lesions = set(va["lesion_id"])
        te_lesions = set(te["lesion_id"])

        self.assertEqual(len(tr_lesions.intersection(va_lesions)), 0)
        self.assertEqual(len(tr_lesions.intersection(te_lesions)), 0)
        self.assertEqual(len(va_lesions.intersection(te_lesions)), 0)

        self.assertEqual(len(tr_lesions) + len(va_lesions) + len(te_lesions), 7470)

    def test_05_reproducibility(self):
        """Verify that split_ham10000_by_lesion is deterministic given random_state."""
        df = parse_ham10000_dataset(self.skin_dir)
        tr1, va1, te1 = split_ham10000_by_lesion(df, random_state=42)
        tr2, va2, te2 = split_ham10000_by_lesion(df, random_state=42)

        self.assertTrue(tr1["image_id"].equals(tr2["image_id"]))
        self.assertTrue(va1["image_id"].equals(va2["image_id"]))
        self.assertTrue(te1["image_id"].equals(te2["image_id"]))


if __name__ == "__main__":
    unittest.main()
