"""
Unit tests for Phase 14 Tuberculosis (TB) Grad-CAM Explainability Module.
"""

import os
import json
import unittest
import hashlib
import numpy as np
import torch
from PIL import Image

from experiments.tb.train_tb_models import build_model
from xai.gradcam.gradcam import GradCAM, get_default_target_layer
from xai.gradcam.visualization import generate_gradcam_overlay
from preprocessing.images.dataset_specific import TuberculosisDataSpec


class TestPhase14TBGradCAM(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.root_dir, "models", "tb")
        self.eval_dir = os.path.join(self.root_dir, "evaluation", "tb", "gradcam")
        self.checkpoint_path = os.path.join(self.model_dir, "best_tb_model.pt")
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")

    def test_01_efficientnet_b0_target_layer_detection(self):
        """Test auto-detection of EfficientNet-B0 final Conv2d target layer."""
        model = build_model("efficientnet_b0", num_classes=2, freeze_backbone=True)
        target_layer = get_default_target_layer(model)

        self.assertIsInstance(target_layer, torch.nn.Conv2d)
        self.assertEqual(target_layer.in_channels, 320)
        self.assertEqual(target_layer.out_channels, 1280)

        # Check feature map shape
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = model.features(dummy_input)
        self.assertEqual(feat.shape, (1, 1280, 7, 7))

    def test_02_hook_registration_and_activation_capture(self):
        """Test that GradCAM registers hooks and captures activation and gradient tensors."""
        model = build_model("efficientnet_b0", num_classes=2, freeze_backbone=True)
        target_layer = get_default_target_layer(model)
        explainer = GradCAM(model=model, target_layer=target_layer)

        dummy_input = torch.randn(1, 3, 224, 224)
        heatmap, pred_class, prob = explainer.generate_heatmap(dummy_input, target_class=1)

        self.assertIsNotNone(explainer.activations)
        self.assertIsNotNone(explainer.gradients)
        self.assertEqual(explainer.activations.shape, (1, 1280, 7, 7))
        self.assertEqual(explainer.gradients.shape, (1, 1280, 7, 7))

        explainer.remove_hooks()
        self.assertIsNone(explainer.forward_handle)
        self.assertIsNone(explainer.backward_handle)

    def test_03_heatmap_shape_finite_and_normalized(self):
        """Verify heatmap dimensions, finite values, and [0, 1] normalization range."""
        model = build_model("efficientnet_b0", num_classes=2, freeze_backbone=True)
        explainer = GradCAM(model=model)

        dummy_input = torch.randn(1, 3, 224, 224)
        heatmap, pred_class, prob = explainer.generate_heatmap(dummy_input, target_class=0)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(np.all(np.isfinite(heatmap)))
        self.assertGreaterEqual(np.min(heatmap), 0.0)
        self.assertLessEqual(np.max(heatmap), 1.0)
        self.assertGreaterEqual(np.max(heatmap), 0.0, "Heatmap max activation should be non-negative")

    def test_04_overlay_generation_shape_and_dtype(self):
        """Test blending heatmap with RGB image to produce valid uint8 overlay."""
        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_heatmap = np.random.rand(224, 224).astype(np.float32)

        heatmap_rgb, overlay_rgb = generate_gradcam_overlay(dummy_img, dummy_heatmap, alpha=0.4)

        self.assertEqual(heatmap_rgb.shape, (224, 224, 3))
        self.assertEqual(overlay_rgb.shape, (224, 224, 3))
        self.assertEqual(heatmap_rgb.dtype, np.uint8)
        self.assertEqual(overlay_rgb.dtype, np.uint8)

    def test_05_class_mapping_and_finalized_checkpoint(self):
        """Test inference and Grad-CAM using finalized Phase 13 checkpoint."""
        self.assertTrue(os.path.exists(self.checkpoint_path), "Phase 13 best_tb_model.pt missing")
        self.assertTrue(os.path.exists(self.metadata_path), "Phase 13 model_metadata.json missing")

        with open(self.metadata_path, "r") as f:
            meta = json.load(f)

        self.assertEqual(meta["class_to_idx"], TuberculosisDataSpec.CLASS_TO_IDX)
        arch = meta.get("selected_architecture", "densenet121")
        self.assertIn(arch, ["efficientnet_b0", "densenet121"])

        model = build_model(arch, num_classes=2, freeze_backbone=True)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
        model.eval()

        explainer = GradCAM(model=model)
        dummy_input = torch.randn(1, 3, 224, 224)

        heatmap, pred_class, prob = explainer.generate_heatmap(dummy_input, target_class=1)
        self.assertIn(pred_class, [0, 1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_06_model_weight_immutability(self):
        """Verify model parameters remain strictly unchanged after Grad-CAM execution."""
        model = build_model("efficientnet_b0", num_classes=2, freeze_backbone=True)

        def get_hash(m):
            hasher = hashlib.sha256()
            for p in m.state_dict().values():
                hasher.update(p.cpu().numpy().tobytes())
            return hasher.hexdigest()

        hash_before = get_hash(model)

        explainer = GradCAM(model=model)
        dummy_input = torch.randn(1, 3, 224, 224)
        _, _, _ = explainer.generate_heatmap(dummy_input, target_class=0)
        _, _, _ = explainer.generate_heatmap(dummy_input, target_class=1)

        hash_after = get_hash(model)
        self.assertEqual(hash_before, hash_after, "Model parameters were mutated during Grad-CAM generation!")

    def test_07_gradcam_summary_artifact_integrity(self):
        """Verify presence and validity of Phase 14 gradcam_summary.json artifact."""
        summary_path = os.path.join(self.eval_dir, "gradcam_summary.json")
        self.assertTrue(os.path.exists(summary_path), f"gradcam_summary.json missing: {summary_path}")

        with open(summary_path, "r") as f:
            summary = json.load(f)

        self.assertEqual(summary["model_architecture"], "efficientnet_b0")
        self.assertTrue(summary["model_weights_immutable"])
        self.assertEqual(summary["normal_explanations_count"], 3)
        self.assertEqual(summary["tb_explanations_count"], 3)
        self.assertEqual(len(summary["explanation_records"]), 6)


if __name__ == "__main__":
    unittest.main()
