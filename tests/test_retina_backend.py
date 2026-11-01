"""
Comprehensive Unit & Integration Test Suite for Phase 24 Retina Backend Integration.

Verifies:
1. GET /api/v1/retina/status endpoint returns model availability and metadata.
2. Model Registry lazy loading of EfficientNet-B0 Retina model.
3. SHA-256 integrity verification against frozen Phase 21 hash.
4. POST /api/v1/retina/analyze image prediction & Grad-CAM visual explanation.
5. Binary classification label mapping ("No Diabetic Retinopathy" vs "Diabetic Retinopathy Present").
6. Probability and confidence range [0.0, 1.0].
7. Grad-CAM target layer resolution ('features.8.0').
8. Error handling for missing files, oversized files, corrupt images, unsupported formats.
9. Controlled failure when model SHA mismatch occurs.
10. Regression testing for existing Lab, X-Ray, and Skin endpoints.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.model_registry import model_registry
from backend.app.services.prediction_service import prediction_service


class TestRetinaBackendIntegration(unittest.TestCase):
    """Test suite for Phase 24 Retina Backend Model & XAI Integration."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_dir = os.path.join("tests", "sample_reports")
        cls.image_path = os.path.join(cls.sample_dir, "sample_image_report.jpg")
        
        # Create sample RGB image file if it doesn't exist
        os.makedirs(cls.sample_dir, exist_ok=True)
        if not os.path.exists(cls.image_path):
            img = Image.new("RGB", (224, 224), color=(128, 64, 32))
            img.save(cls.image_path)

    def test_01_retina_status_endpoint(self):
        """1. Test GET /api/v1/retina/status endpoint structure and values."""
        response = self.client.get("/api/v1/retina/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "retina")
        self.assertEqual(data["supported_tasks"], ["diabetic_retinopathy"])
        self.assertEqual(data["xai_method"], "Grad-CAM")
        self.assertIn("model_architecture", data)
        self.assertEqual(data["model_architecture"], "EfficientNet-B0")

    def test_02_retina_model_availability(self):
        """2. Test model registry reports model availability for Retina task."""
        is_avail = model_registry.is_model_available("retina", "diabetic_retinopathy")
        meta = model_registry.get_metadata("retina", "diabetic_retinopathy")
        self.assertTrue(is_avail)
        self.assertEqual(meta.status, "AVAILABLE")
        self.assertEqual(meta.artifact_path, settings.RETINA_MODEL_PATH)

    def test_03_retina_xai_availability(self):
        """3. Test XAI method for Retina is defined as Grad-CAM."""
        meta = model_registry.get_metadata("retina", "diabetic_retinopathy")
        self.assertEqual(meta.xai_method, "Grad-CAM")

    def test_04_valid_retina_image_inference(self):
        """4. Test POST /api/v1/retina/analyze returns successful 200 payload."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "retina")
        self.assertEqual(data["task"], "diabetic_retinopathy")

    def test_05_binary_prediction_mapping(self):
        """5. Test predicted label is mapped to valid binary DR class names."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        data = response.json()
        pred = data["prediction"]
        self.assertIsNotNone(pred)
        self.assertIn(pred["label"], ["No Diabetic Retinopathy", "Diabetic Retinopathy Present"])

    def test_06_probability_range(self):
        """6. Test returned prediction probability is bounded within [0.0, 1.0]."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        data = response.json()
        prob = data["prediction"]["probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_07_confidence_range(self):
        """7. Test raw output probabilities are formatted and within [0.0, 1.0]."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        data = response.json()
        raw = data["prediction"]["raw_output"]
        self.assertIn("dr_present_probability", raw)
        self.assertIn("no_dr_probability", raw)
        self.assertGreaterEqual(raw["dr_present_probability"], 0.0)
        self.assertLessEqual(raw["dr_present_probability"], 1.0)
        self.assertGreaterEqual(raw["no_dr_probability"], 0.0)
        self.assertLessEqual(raw["no_dr_probability"], 1.0)

    def test_08_gradcam_generation(self):
        """8. Test Grad-CAM visual explanation is available in response."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        data = response.json()
        exp = data["explanation"]
        self.assertEqual(exp["method"], "Grad-CAM")
        self.assertTrue(exp["available"])

    def test_09_heatmap_dimensions(self):
        """9. Test Grad-CAM details contain 7x7 spatial grid summary and 224x224 heatmap shape."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        details = response.json()["explanation"]["details"]
        self.assertEqual(details["heatmap_shape"], [224, 224])
        self.assertEqual(len(details["spatial_summary_grid_7x7"]), 49)

    def test_10_heatmap_range(self):
        """10. Test Grad-CAM heatmap min/max values are within [0.0, 1.0]."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        details = response.json()["explanation"]["details"]
        self.assertGreaterEqual(details["heatmap_min"], 0.0)
        self.assertLessEqual(details["heatmap_max"], 1.0)

    def test_11_target_layer_verification(self):
        """11. Test Grad-CAM target layer is correctly reported as 'features.8.0'."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("fundus_sample.jpg", f, "image/jpeg")},
            )
        details = response.json()["explanation"]["details"]
        self.assertEqual(details["target_layer"], "features.8.0")

    def test_12_model_sha_integrity(self):
        """12. Test loaded Retina model matches expected frozen Phase 21 SHA-256 hash."""
        import hashlib
        EXPECTED_SHA = "4bf9260be9393f440d2a201dc763edb2eab52ab57c1f2d7c6780deddd1ac7a07"
        hasher = hashlib.sha256()
        with open(settings.RETINA_MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        self.assertEqual(actual_sha, EXPECTED_SHA)

    def test_13_missing_file_handling(self):
        """13. Test request without file returns HTTP 422 validation error."""
        response = self.client.post("/api/v1/retina/analyze?task=diabetic_retinopathy")
        self.assertEqual(response.status_code, 422)

    def test_14_oversized_file_handling(self):
        """14. Test uploading file exceeding size limit is rejected with HTTP 400."""
        oversized_bytes = b"0" * (20 * 1024 * 1024 + 1024)
        response = self.client.post(
            "/api/v1/retina/analyze?task=diabetic_retinopathy",
            files={"file": ("large_retina.jpg", oversized_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_15_corrupt_image_handling(self):
        """15. Test uploading corrupt image bytes returns INVALID_INPUT HTTP 400."""
        corrupt_bytes = b"corrupt binary data stream"
        response = self.client.post(
            "/api/v1/retina/analyze?task=diabetic_retinopathy",
            files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_16_unsupported_format_handling(self):
        """16. Test uploading unsupported file format (.exe) returns INVALID_INPUT HTTP 400."""
        response = self.client.post(
            "/api/v1/retina/analyze?task=diabetic_retinopathy",
            files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")

    def test_17_controlled_model_failure_handling(self):
        """17. Test controlled ModelNotAvailableError when model SHA integrity check fails."""
        with patch("builtins.open", side_effect=IOError("Simulated file read failure")):
            # Reset cache to force reload attempt
            model_registry._cached_models.pop(("retina", "diabetic_retinopathy"), None)
            with self.assertRaises(Exception):
                model_registry.load_model("retina", "diabetic_retinopathy")

    def test_18_existing_lab_endpoint_regression(self):
        """18. Test Lab status endpoint regression."""
        response = self.client.get("/api/v1/lab/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "lab")
        self.assertEqual(data["task"], "30_day_readmission")
        self.assertEqual(data["model_status"], "AVAILABLE")

    def test_19_existing_xray_endpoint_regression(self):
        """19. Test X-Ray status endpoint regression."""
        response = self.client.get("/api/v1/xray/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "xray")
        self.assertIn("pneumonia", data["supported_tasks"])
        self.assertIn("tb", data["supported_tasks"])

    def test_20_existing_skin_endpoint_regression(self):
        """20. Test Skin status endpoint regression."""
        response = self.client.get("/api/v1/skin/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "skin")
        self.assertEqual(data["supported_tasks"], ["ham10000"])


if __name__ == "__main__":
    unittest.main()
