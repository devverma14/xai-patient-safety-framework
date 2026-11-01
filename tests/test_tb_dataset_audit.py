import unittest
import os
import pandas as pd
from PIL import Image

from preprocessing.images.dataset_specific import parse_tb_dataset, split_tb_dataset


class TestTBDatasetAudit(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tb_dir = os.path.join(self.root_dir, "datasets", "tb")
        self.montgomery_dir = os.path.join(self.tb_dir, "montgomery")
        self.shenzhen_dir = os.path.join(self.tb_dir, "shenzhen")

    def test_tb_dataset_existence(self):
        """Verify that Montgomery and Shenzhen dataset folders exist."""
        self.assertTrue(os.path.exists(self.montgomery_dir), "Montgomery directory missing.")
        self.assertTrue(os.path.exists(self.shenzhen_dir), "Shenzhen directory missing.")

    def test_parse_tb_dataset_counts_and_labels(self):
        """Verify total counts and site-specific label distributions."""
        df = parse_tb_dataset(self.tb_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 800, f"Expected 800 total images, found {len(df)}")

        # Montgomery counts
        mont_df = df[df["site"] == "montgomery"]
        self.assertEqual(len(mont_df), 138, "Montgomery should have 138 images")
        self.assertEqual(sum(mont_df["label"] == 0), 80, "Montgomery Normal should be 80")
        self.assertEqual(sum(mont_df["label"] == 1), 58, "Montgomery TB should be 58")

        # Shenzhen counts
        shen_df = df[df["site"] == "shenzhen"]
        self.assertEqual(len(shen_df), 662, "Shenzhen should have 662 images")
        self.assertEqual(sum(shen_df["label"] == 0), 326, "Shenzhen Normal should be 326")
        self.assertEqual(sum(shen_df["label"] == 1), 336, "Shenzhen TB should be 336")

    def test_patient_leakage_and_id_extraction(self):
        """Ensure patient IDs are correctly extracted and mapped 1-to-1."""
        df = parse_tb_dataset(self.tb_dir)
        self.assertEqual(df["patient_id"].nunique(), len(df), "Each image should belong to a unique patient ID")

    def test_image_readability_and_integrity(self):
        """Check that all 800 images can be opened and have valid dimensions."""
        df = parse_tb_dataset(self.tb_dir)
        for _, row in df.iterrows():
            filepath = row["file_path"]
            self.assertTrue(os.path.exists(filepath), f"File does not exist: {filepath}")
            with Image.open(filepath) as img:
                w, h = img.size
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)

    def test_split_tb_dataset_cross_site(self):
        """Verify cross_site split logic (Shenzhen train/val, Montgomery held-out test)."""
        df = parse_tb_dataset(self.tb_dir)
        train_df, val_df, test_df = split_tb_dataset(df, strategy="cross_site", val_ratio=0.20, random_state=42)

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 800)

        # Montgomery should be 100% test set
        self.assertTrue((test_df["site"] == "montgomery").all())
        self.assertEqual(len(test_df), 138)

        # Shenzhen should be in train/val only
        self.assertTrue((train_df["site"] == "shenzhen").all())
        self.assertTrue((val_df["site"] == "shenzhen").all())

        # Check zero patient overlap between splits
        train_pids = set(train_df["patient_id"])
        val_pids = set(val_df["patient_id"])
        test_pids = set(test_df["patient_id"])

        self.assertEqual(len(train_pids.intersection(val_pids)), 0)
        self.assertEqual(len(train_pids.intersection(test_pids)), 0)
        self.assertEqual(len(val_pids.intersection(test_pids)), 0)

    def test_split_tb_dataset_site_stratified(self):
        """Verify pooled site-stratified group split logic."""
        df = parse_tb_dataset(self.tb_dir)
        train_df, val_df, test_df = split_tb_dataset(df, strategy="site_stratified", random_state=42)

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 800)

        # Check zero patient overlap between splits
        train_pids = set(train_df["patient_id"])
        val_pids = set(val_df["patient_id"])
        test_pids = set(test_df["patient_id"])

        self.assertEqual(len(train_pids.intersection(val_pids)), 0)
        self.assertEqual(len(train_pids.intersection(test_pids)), 0)
        self.assertEqual(len(val_pids.intersection(test_pids)), 0)


if __name__ == "__main__":
    unittest.main()
