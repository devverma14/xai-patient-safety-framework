"""
Unit tests for Phase 13 Tuberculosis (TB) Model Development Pipeline.
"""

import os
import json
import unittest
import torch
import pandas as pd
from PIL import Image
import tempfile

from experiments.tb.train_tb_models import build_model, TuberculosisDataset
from preprocessing.images.dataset_specific import parse_tb_dataset, split_tb_dataset, TuberculosisDataSpec


class TestPhase13TBModelPipeline(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tb_dir = os.path.join(self.root_dir, "datasets", "tb")
        self.model_dir = os.path.join(self.root_dir, "models", "tb")
        self.eval_dir = os.path.join(self.root_dir, "evaluation", "tb")

    def test_tb_data_spec_constants(self):
        """Test Tuberculosis dataset specification constants."""
        self.assertEqual(TuberculosisDataSpec.CLASSES, ["Normal", "Tuberculosis"])
        self.assertEqual(TuberculosisDataSpec.CLASS_TO_IDX["Normal"], 0)
        self.assertEqual(TuberculosisDataSpec.CLASS_TO_IDX["Tuberculosis"], 1)
        self.assertEqual(TuberculosisDataSpec.IDX_TO_CLASS[0], "Normal")
        self.assertEqual(TuberculosisDataSpec.IDX_TO_CLASS[1], "Tuberculosis")

    def test_model_architectures_output_shape(self):
        """Test candidate model building and forward pass output shape."""
        architectures = ["densenet121", "efficientnet_b0", "resnet50"]
        dummy_input = torch.randn(2, 3, 224, 224)

        for arch in architectures:
            model = build_model(arch, num_classes=2, freeze_backbone=True)
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            self.assertEqual(output.shape, (2, 2), f"Architecture {arch} output shape should be (2, 2)")

    def test_split_tb_dataset_zero_patient_leakage(self):
        """Test cross-site and site-stratified splitting for zero patient overlap."""
        df = parse_tb_dataset(self.tb_dir)

        # Cross-site split
        tr1, va1, te1 = split_tb_dataset(df, strategy="cross_site", val_ratio=0.20, random_state=42)
        p1_tr, p1_va, p1_te = set(tr1["patient_id"]), set(va1["patient_id"]), set(te1["patient_id"])
        self.assertEqual(len(p1_tr.intersection(p1_va)), 0)
        self.assertEqual(len(p1_tr.intersection(p1_te)), 0)
        self.assertEqual(len(p1_va.intersection(p1_te)), 0)

        # Site-stratified split
        tr2, va2, te2 = split_tb_dataset(df, strategy="site_stratified", random_state=42)
        p2_tr, p2_va, p2_te = set(tr2["patient_id"]), set(va2["patient_id"]), set(te2["patient_id"])
        self.assertEqual(len(p2_tr.intersection(p2_va)), 0)
        self.assertEqual(len(p2_tr.intersection(p2_te)), 0)
        self.assertEqual(len(p2_va.intersection(p2_te)), 0)

    def test_checkpoint_and_metadata_integrity(self):
        """Verify saved best_tb_model.pt checkpoint and model_metadata.json."""
        checkpoint_path = os.path.join(self.model_dir, "best_tb_model.pt")
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")

        self.assertTrue(os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}")
        self.assertTrue(os.path.exists(metadata_path), f"Metadata missing: {metadata_path}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        self.assertEqual(meta["task"], "tb")
        self.assertEqual(meta["modality"], "xray")
        self.assertIn("selected_architecture", meta)

        arch = meta["selected_architecture"]
        model = build_model(arch, num_classes=2, freeze_backbone=True)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_input)
        self.assertEqual(logits.shape, (1, 2))

    def test_evaluation_artifacts_exist(self):
        """Verify presence of all required Phase 13 evaluation JSON and CSV artifacts."""
        metrics_path = os.path.join(self.eval_dir, "metrics.json")
        preds_path = os.path.join(self.eval_dir, "external_site_predictions.csv")
        rec_path = os.path.join(self.eval_dir, "reciprocal_cross_site_metrics.json")
        pool_path = os.path.join(self.eval_dir, "pooled_benchmark_metrics.json")

        self.assertTrue(os.path.exists(metrics_path), "metrics.json missing")
        self.assertTrue(os.path.exists(preds_path), "external_site_predictions.csv missing")
        self.assertTrue(os.path.exists(rec_path), "reciprocal_cross_site_metrics.json missing")
        self.assertTrue(os.path.exists(pool_path), "pooled_benchmark_metrics.json missing")

        preds_df = pd.read_csv(preds_path)
        self.assertEqual(len(preds_df), 138, "External predictions should contain 138 Montgomery rows")
        self.assertIn("predicted_label_idx", preds_df.columns)
        self.assertIn("tb_probability", preds_df.columns)


if __name__ == "__main__":
    unittest.main()
