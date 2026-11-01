"""
Unit tests for Phase 16: Skin Lesion (HAM10000) 7-Class Model Development & Artifact Verification.
"""

import unittest
from pathlib import Path
import json
import torch

from preprocessing.images import (
    HAM10000DataSpec,
    parse_ham10000_dataset,
    split_ham10000_by_lesion,
)
from experiments.skin.train_skin_models import (
    build_full_model as build_model,
    get_skin_transforms,
    compute_class_weights,
    DATASET_DIR,
    MODEL_SAVE_DIR,
    EVAL_SAVE_DIR,
)


class TestPhase16SkinModel(unittest.TestCase):
    """Test suite for Phase 16 HAM10000 7-class model pipeline and artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = DATASET_DIR
        cls.model_dir = MODEL_SAVE_DIR
        cls.eval_dir = EVAL_SAVE_DIR

    def test_01_ham10000_class_spec(self):
        """Test HAM10000 7-class specification mapping."""
        self.assertEqual(HAM10000DataSpec.NUM_CLASSES, 7)
        self.assertEqual(len(HAM10000DataSpec.CLASSES), 7)
        expected_classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
        self.assertEqual(HAM10000DataSpec.CLASSES, expected_classes)

        for i, cat in enumerate(expected_classes):
            self.assertEqual(HAM10000DataSpec.CLASS_TO_IDX[cat], i)
            self.assertEqual(HAM10000DataSpec.IDX_TO_CLASS[i], cat)
            self.assertIn(cat, HAM10000DataSpec.CLASS_NAMES)

    def test_02_dataset_parsing_and_class_weights(self):
        """Test parsing HAM10000 dataset and computing training class weights."""
        if not self.dataset_dir.exists():
            self.skipTest("Skin dataset directory not found.")

        df = parse_ham10000_dataset(str(self.dataset_dir))
        self.assertEqual(len(df), 10015)

        train_df, val_df, test_df = split_ham10000_by_lesion(df, random_state=42)
        weights = compute_class_weights(train_df, num_classes=7)

        self.assertEqual(weights.shape[0], 7)
        # Class 0 (nv) should have the smallest weight, minority classes higher
        self.assertLess(weights[0].item(), weights[1].item())
        self.assertLess(weights[0].item(), weights[6].item())  # nv < df

    def test_03_lesion_level_split_zero_overlap(self):
        """Test that split_ham10000_by_lesion prevents lesion-level data leakage."""
        if not self.dataset_dir.exists():
            self.skipTest("Skin dataset directory not found.")

        df = parse_ham10000_dataset(str(self.dataset_dir))
        train_df, val_df, test_df = split_ham10000_by_lesion(df, random_state=42)

        tr_lesions = set(train_df["lesion_id"].unique())
        va_lesions = set(val_df["lesion_id"].unique())
        te_lesions = set(test_df["lesion_id"].unique())

        self.assertEqual(len(tr_lesions.intersection(va_lesions)), 0)
        self.assertEqual(len(tr_lesions.intersection(te_lesions)), 0)
        self.assertEqual(len(va_lesions.intersection(te_lesions)), 0)

    def test_04_model_architectures_output_shape(self):
        """Test candidate CNN backbones produce output shape [batch_size, 7]."""
        dummy_input = torch.randn(2, 3, 224, 224)
        for arch in ["densenet121", "efficientnet_b0", "resnet50"]:
            model = build_model(arch, num_classes=7)
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, (2, 7), f"Architecture {arch} output shape mismatch.")

    def test_05_preprocessing_transforms(self):
        """Test image preprocessing transform pipelines."""
        train_tf = get_skin_transforms(is_training=True)
        eval_tf = get_skin_transforms(is_training=False)

        self.assertIsNotNone(train_tf)
        self.assertIsNotNone(eval_tf)

    def test_06_saved_checkpoint_and_metadata_integrity(self):
        """Test saved best_skin_model.pt checkpoint and model_metadata.json integrity."""
        ckpt_path = self.model_dir / "best_skin_model.pt"
        meta_path = self.model_dir / "model_metadata.json"

        if not ckpt_path.exists() or not meta_path.exists():
            self.skipTest("Phase 16 training artifacts not yet generated.")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("architecture", checkpoint)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["num_classes"], 7)

        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["num_classes"], 7)
        self.assertIn("architecture", meta)
        self.assertIn("val_metrics", meta)

    def test_07_evaluation_artifacts_exist(self):
        """Test presence and validity of required Phase 16 evaluation artifacts."""
        metrics_path = self.eval_dir / "metrics.json"
        pred_path = self.eval_dir / "test_predictions.csv"
        cm_path = self.eval_dir / "confusion_matrix.png"
        comp_path = self.eval_dir / "candidate_comparison.json"

        if not metrics_path.exists():
            self.skipTest("Phase 16 evaluation artifacts not yet generated.")

        self.assertTrue(metrics_path.exists())
        self.assertTrue(pred_path.exists())
        self.assertTrue(cm_path.exists())
        self.assertTrue(comp_path.exists())

        with open(metrics_path, "r") as f:
            eval_data = json.load(f)

        self.assertIn("test_metrics", eval_data)
        self.assertIn("feasibility_criteria", eval_data)
        self.assertEqual(eval_data["num_classes"], 7)

    def test_08_phase16_1_artifacts_integrity(self):
        """Test presence and integrity of Phase 16.1 targeted experiment artifacts."""
        p16_1_model_dir = self.model_dir / "phase16_1"
        p16_1_eval_dir = self.eval_dir / "phase16_1"

        if not (p16_1_model_dir / "best_skin_model_phase16_1.pt").exists():
            self.skipTest("Phase 16.1 artifacts not found.")

        self.assertTrue((p16_1_model_dir / "best_skin_model_phase16_1.pt").exists())
        self.assertTrue((p16_1_model_dir / "model_metadata.json").exists())
        self.assertTrue((p16_1_eval_dir / "experiment_comparison.json").exists())
        self.assertTrue((p16_1_eval_dir / "metrics.json").exists())
        self.assertTrue((p16_1_eval_dir / "confusion_matrix.png").exists())
        self.assertTrue((p16_1_eval_dir / "test_predictions.csv").exists())

        with open(p16_1_eval_dir / "metrics.json", "r") as f:
            eval_data = json.load(f)

        self.assertIn("selected_experiment", eval_data)
        self.assertIn("val_metrics", eval_data)
        self.assertIn("test_metrics", eval_data)
        self.assertIn("feasibility_criteria", eval_data)

    def test_09_phase16_2_binary_artifacts_integrity(self):
        """Test presence and integrity of Phase 16.2 binary skin model artifacts."""
        binary_model_dir = self.model_dir / "binary"
        binary_eval_dir = self.eval_dir / "binary"

        if not (binary_model_dir / "best_skin_binary_model.pt").exists():
            self.skipTest("Phase 16.2 binary artifacts not found.")

        self.assertTrue((binary_model_dir / "best_skin_binary_model.pt").exists())
        self.assertTrue((binary_model_dir / "model_metadata.json").exists())
        self.assertTrue((binary_eval_dir / "candidate_comparison.json").exists())
        self.assertTrue((binary_eval_dir / "metrics.json").exists())
        self.assertTrue((binary_eval_dir / "confusion_matrix.png").exists())
        self.assertTrue((binary_eval_dir / "test_predictions.csv").exists())

        checkpoint = torch.load(binary_model_dir / "best_skin_binary_model.pt", map_location="cpu")
        self.assertEqual(checkpoint["num_classes"], 2)
        self.assertIn("model_state_dict", checkpoint)

        with open(binary_eval_dir / "metrics.json", "r") as f:
            eval_data = json.load(f)

        self.assertIn("selected_experiment", eval_data)
        self.assertIn("val_metrics", eval_data)
        self.assertIn("test_metrics", eval_data)
        self.assertIn("feasibility_criteria", eval_data)

    def test_10_binary_label_mapping_and_split(self):
        """Test binary label mapping (Malignant vs Benign) and lesion-level split."""
        if not self.dataset_dir.exists():
            self.skipTest("Skin dataset directory not found.")

        from experiments.skin.train_skin_binary import map_binary_labels
        df = parse_ham10000_dataset(str(self.dataset_dir))
        df_binary = map_binary_labels(df)

        self.assertIn("binary_label", df_binary.columns)
        malignant_count = (df_binary["binary_label"] == 1).sum()
        benign_count = (df_binary["binary_label"] == 0).sum()

        self.assertEqual(malignant_count + benign_count, 10015)
        self.assertEqual(malignant_count, 1954)
        self.assertEqual(benign_count, 8061)

        train_df, val_df, test_df = split_ham10000_by_lesion(df_binary, random_state=42)
        self.assertEqual(len(train_df), 7009)
        self.assertEqual(len(val_df), 1520)
        self.assertEqual(len(test_df), 1486)

    def test_11_threshold_analysis_artifact_integrity(self):
        """Test presence and integrity of Phase 16.2 threshold calibration artifact."""
        thresh_path = self.eval_dir / "binary" / "threshold_analysis.json"
        if not thresh_path.exists():
            self.skipTest("Threshold analysis artifact not found.")

        with open(thresh_path, "r") as f:
            data = json.load(f)

        self.assertIn("selected_threshold", data)
        self.assertIn("validation_sweep", data)
        self.assertIn("selected_val_metrics", data)
        self.assertIn("selected_test_metrics", data)
        self.assertGreaterEqual(data["selected_threshold"], 0.10)
        self.assertLessEqual(data["selected_threshold"], 0.90)


if __name__ == "__main__":
    unittest.main()



