"""
Phase 11 — Chest X-Ray Pneumonia Grad-CAM Explainability Unit Tests.

Tests:
1. GradCAM Initialization & Layer Auto-Detection: Verifies model hook registration and target layer auto-detection.
2. Heatmap Generation Shape & Range: Verifies 2D heatmap output shape (224, 224) and [0, 1] normalization range.
3. Finite Heatmap Value Validation: Verifies absence of NaNs, Infs, or unnormalized values.
4. Overlay Generation Shape & Dtype: Verifies heatmap RGB and blended overlay dimensions (224, 224, 3) and uint8 dtype.
5. Target Class Scoring: Verifies score target evaluation for NORMAL (0) and PNEUMONIA (1).
6. Finalized Model Integration: Verifies Grad-CAM execution on loaded best_pneumonia_model.pt checkpoint.
"""

import unittest
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image

from xai.gradcam import (
    GradCAM,
    get_default_target_layer,
    normalize_heatmap,
    apply_colormap_on_image,
    generate_gradcam_overlay,
)
from experiments.pneumonia.train_pneumonia_models import build_model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pneumonia" / "best_pneumonia_model.pt"


class TestPneumoniaGradCAM(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = build_model("densenet121", num_classes=2, freeze_backbone=True).to(self.device)
        self.model.eval()

    def test_01_gradcam_initialization_and_layer_detection(self):
        """Test GradCAM auto-detecting target layer and registering PyTorch hooks."""
        target_layer = get_default_target_layer(self.model)
        self.assertIsInstance(target_layer, nn.Conv2d)

        # Explicit target layer initialization
        explicit_target = self.model.features.denseblock4.denselayer16.conv2
        gradcam = GradCAM(model=self.model, target_layer=explicit_target, device=self.device)

        self.assertIsNotNone(gradcam.forward_handle)
        self.assertIsNotNone(gradcam.backward_handle)
        gradcam.remove_hooks()
        self.assertIsNone(gradcam.forward_handle)
        self.assertIsNone(gradcam.backward_handle)

    def test_02_heatmap_generation_shape_and_range(self):
        """Test Grad-CAM heatmap output shape (224, 224) and [0, 1] value bounds."""
        gradcam = GradCAM(model=self.model, device=self.device)
        dummy_input = torch.randn(1, 3, 224, 224)

        heatmap, pred_class, class_prob = gradcam.generate_heatmap(dummy_input, target_class=1)

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertIn(pred_class, [0, 1])
        self.assertGreaterEqual(class_prob, 0.0)
        self.assertLessEqual(class_prob, 1.0)
        self.assertGreaterEqual(np.min(heatmap), 0.0)
        self.assertLessEqual(np.max(heatmap), 1.0)

        gradcam.remove_hooks()

    def test_03_finite_heatmap_values(self):
        """Test that generated heatmap contains only finite values (no NaNs or Infs)."""
        gradcam = GradCAM(model=self.model, device=self.device)
        dummy_input = torch.randn(1, 3, 224, 224)

        heatmap, _, _ = gradcam.generate_heatmap(dummy_input, target_class=0)
        self.assertTrue(np.all(np.isfinite(heatmap)), "Heatmap contains non-finite values (NaN or Inf).")

        gradcam.remove_hooks()

    def test_04_overlay_generation_shape_and_dtype(self):
        """Test heatmap RGB colormap mapping and blended overlay array generation."""
        dummy_img_rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
        dummy_heatmap = np.linspace(0, 1, 224 * 224).reshape(224, 224).astype(np.float32)

        heatmap_rgb, overlay_rgb = generate_gradcam_overlay(dummy_img_rgb, dummy_heatmap, alpha=0.5, colormap_name="jet")

        self.assertEqual(heatmap_rgb.shape, (224, 224, 3))
        self.assertEqual(heatmap_rgb.dtype, np.uint8)
        self.assertEqual(overlay_rgb.shape, (224, 224, 3))
        self.assertEqual(overlay_rgb.dtype, np.uint8)

    def test_05_target_class_scoring(self):
        """Test generating Grad-CAM heatmaps for both NORMAL (0) and PNEUMONIA (1) target classes."""
        gradcam = GradCAM(model=self.model, device=self.device)
        dummy_input = torch.randn(1, 3, 224, 224)

        heatmap_normal, cls_0, prob_0 = gradcam.generate_heatmap(dummy_input, target_class=0)
        heatmap_pneumonia, cls_1, prob_1 = gradcam.generate_heatmap(dummy_input, target_class=1)

        self.assertEqual(cls_0, 0)
        self.assertEqual(cls_1, 1)
        self.assertEqual(heatmap_normal.shape, (224, 224))
        self.assertEqual(heatmap_pneumonia.shape, (224, 224))

        gradcam.remove_hooks()

    def test_06_real_pneumonia_model_gradcam(self):
        """Test Grad-CAM execution using the finalized model checkpoint weights."""
        self.assertTrue(MODEL_PATH.exists(), f"Finalized model checkpoint not found at: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        model = build_model("densenet121", num_classes=2, freeze_backbone=True).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        gradcam = GradCAM(model=model, device=self.device)
        dummy_input = torch.randn(1, 3, 224, 224)

        heatmap, pred_cls, prob = gradcam.generate_heatmap(dummy_input, target_class=1)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(np.all(np.isfinite(heatmap)))

        gradcam.remove_hooks()


if __name__ == "__main__":
    unittest.main()
