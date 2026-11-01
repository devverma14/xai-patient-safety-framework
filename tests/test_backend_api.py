"""
Comprehensive Test Suite for Phase 8 Modality-Aware Backend API.
Verifies FastAPI routing, OpenAPI schema generation, Lab XGBoost + SHAP integration,
patient safety prediction blocking for incomplete inputs, and placeholder image endpoints (X-Ray, Skin, Retina).
"""

import os
import sys
import json
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.app.main import app


class TestPhase8BackendAPI(unittest.TestCase):
    """Integration test suite for Phase 8 FastAPI backend."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_dir = os.path.join("tests", "sample_reports")
        cls.image_path = os.path.join(cls.sample_dir, "sample_image_report.jpg")

    def test_01_health_and_root_endpoint(self):
        """Test GET / health check endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("lab", data["supported_modalities"])
        self.assertIn("xray", data["supported_modalities"])
        self.assertIn("skin", data["supported_modalities"])
        self.assertIn("retina", data["supported_modalities"])

    def test_02_openapi_json(self):
        """Test GET /api/v1/openapi.json OpenAPI schema generation."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertIn("paths", schema)
        self.assertIn("/api/v1/lab/analyze", schema["paths"])
        self.assertIn("/api/v1/xray/analyze", schema["paths"])
        self.assertIn("/api/v1/skin/analyze", schema["paths"])
        self.assertIn("/api/v1/retina/analyze", schema["paths"])

    def test_03_lab_status_endpoint(self):
        """Test GET /api/v1/lab/status metadata endpoint."""
        response = self.client.get("/api/v1/lab/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "lab")
        self.assertEqual(data["task"], "30_day_readmission")
        self.assertEqual(data["model_status"], "AVAILABLE")
        self.assertEqual(data["decision_threshold"], 0.35)
        self.assertEqual(data["xai_method"], "SHAP")
        self.assertEqual(data["model_features_count"], 49)

    def test_03b_xray_status_endpoint(self):
        """Test GET /api/v1/xray/status metadata endpoint includes DenseNet121 architecture."""
        response = self.client.get("/api/v1/xray/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "xray")
        self.assertIn("pneumonia", data["supported_tasks"])
        self.assertIn("tb", data["supported_tasks"])
        self.assertEqual(data["model_architecture"], "DenseNet121")
        self.assertIn("task_architectures", data)
        self.assertEqual(data["task_architectures"]["pneumonia"], "DenseNet121")
        self.assertEqual(data["task_architectures"]["tb"], "DenseNet121")

    def test_04_lab_analyze_incomplete_report_blocks_prediction(self):
        """Test that POST /api/v1/lab/analyze with incomplete report blocks prediction for patient safety."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/lab/analyze",
                files={"file": ("sample_image_report.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["review_required", "insufficient_data"])
        self.assertIsNone(data["prediction"])
        self.assertTrue(data["review_required"])
        self.assertFalse(data["explanation"]["available"])

    def test_05_lab_analyze_complete_input_success(self):
        """Test POST /api/v1/lab/analyze with report + complete explicit structured clinical data."""
        complete_explicit_input = {
            "race": "Caucasian", "gender": "Female", "age": "[60-70)",
            "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
            "time_in_hospital": 5, "payer_code": "?", "medical_specialty": "?",
            "num_lab_procedures": 45, "num_procedures": 0, "num_medications": 12,
            "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2,
            "diag_1": "250", "diag_2": "?", "diag_3": "?", "number_diagnoses": 6,
            "max_glu_serum": "None", "A1Cresult": "None", "metformin": "No",
            "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
            "glimepiride": "No", "acetohexamide": "No", "glipizide": "No",
            "glyburide": "No", "tolbutamide": "No", "pioglitazone": "No",
            "rosiglitazone": "No", "acarbose": "No", "miglitol": "No",
            "troglitazone": "No", "tolazamide": "No", "examide": "No",
            "citoglipton": "No", "insulin": "Up", "glyburide-metformin": "No",
            "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No",
            "change": "No", "diabetesMed": "Yes", "readmitted": "NO",
            "encounter_id": 999999, "patient_nbr": 888888,
        }
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/lab/analyze",
                files={"file": ("sample_image_report.jpg", f, "image/jpeg")},
                data={"explicit_structured_input": json.dumps(complete_explicit_input)},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "lab")
        self.assertIsNotNone(data["prediction"])
        self.assertEqual(data["prediction"]["threshold"], 0.35)
        self.assertTrue(data["prediction"]["probability"] > 0.0)
        self.assertEqual(data["explanation"]["method"], "SHAP")
        self.assertTrue(data["explanation"]["available"])

    def test_06_lab_analyze_invalid_extension(self):
        """Test uploading file with unsupported extension triggers INVALID_INPUT error."""
        response = self.client.post(
            "/api/v1/lab/analyze",
            files={"file": ("script.exe", b"binary content", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_07_lab_analyze_invalid_json_form(self):
        """Test uploading invalid JSON string in explicit_structured_input triggers INVALID_INPUT error."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/lab/analyze",
                files={"file": ("sample_image_report.jpg", f, "image/jpeg")},
                data={"explicit_structured_input": "{invalid_json:}"},
            )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_08_xray_pneumonia_inference(self):
        """Test POST /api/v1/xray/analyze?task=pneumonia executes DenseNet121 inference & Grad-CAM."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/xray/analyze?task=pneumonia",
                files={"file": ("xray.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "xray")
        self.assertEqual(data["task"], "pneumonia")
        self.assertIsNotNone(data["prediction"])
        self.assertIn(data["prediction"]["label"], ["PNEUMONIA", "NORMAL"])
        self.assertIsInstance(data["prediction"]["probability"], float)
        self.assertEqual(data["prediction"]["threshold"], 0.50)
        self.assertEqual(data["prediction"]["model_name"], "DenseNet121")
        self.assertEqual(data["prediction"]["model_architecture"], "DenseNet121")
        self.assertEqual(data["explanation"]["method"], "Grad-CAM")
        self.assertTrue(data["explanation"]["available"])
        self.assertEqual(data["explanation"]["details"]["target_layer"], "features.denseblock4.denselayer16.conv2")
        self.assertIn("spatial_summary_grid_7x7", data["explanation"]["details"])
        self.assertTrue(data["review_required"])

    def test_09_xray_tb_inference(self):
        """Test POST /api/v1/xray/analyze?task=tb executes DenseNet121 inference & Grad-CAM."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/xray/analyze?task=tb",
                files={"file": ("xray.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "xray")
        self.assertEqual(data["task"], "tb")
        self.assertIsNotNone(data["prediction"])
        self.assertIn(data["prediction"]["label"], ["Tuberculosis", "Normal"])
        self.assertIsInstance(data["prediction"]["probability"], float)
        self.assertEqual(data["prediction"]["threshold"], 0.50)
        self.assertEqual(data["prediction"]["model_name"], "DenseNet121")
        self.assertEqual(data["prediction"]["model_architecture"], "DenseNet121")
        self.assertEqual(data["explanation"]["method"], "Grad-CAM")
        self.assertTrue(data["explanation"]["available"])
        self.assertEqual(data["explanation"]["details"]["target_layer"], "features.denseblock4.denselayer16.conv2")
        self.assertTrue(data["review_required"])

    def test_10_skin_binary_inference_threshold_0_40(self):
        """Test POST /api/v1/skin/analyze?task=ham10000 uses binary DenseNet121 with frozen threshold t=0.40."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/skin/analyze?task=ham10000",
                files={"file": ("skin.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "skin")
        self.assertEqual(data["task"], "ham10000")
        self.assertIsNotNone(data["prediction"])
        # CRITICAL VERIFICATION: Threshold MUST be exactly 0.40
        self.assertEqual(data["prediction"]["threshold"], 0.40)
        self.assertIn(data["prediction"]["label"], ["HIGHER-RISK", "LOWER-RISK"])
        self.assertIsInstance(data["prediction"]["probability"], float)
        self.assertEqual(data["explanation"]["method"], "Grad-CAM")
        self.assertTrue(data["explanation"]["available"])
        self.assertTrue(data["review_required"])

    def test_11_retina_dr_model_analysis(self):
        """Test POST /api/v1/retina/analyze?task=diabetic_retinopathy returns successful prediction and Grad-CAM."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/retina/analyze?task=diabetic_retinopathy",
                files={"file": ("retina.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["modality"], "retina")
        self.assertEqual(data["task"], "diabetic_retinopathy")
        self.assertIsNotNone(data["prediction"])
        self.assertIn(data["prediction"]["label"], ["Diabetic Retinopathy Present", "No Diabetic Retinopathy"])
        self.assertIsInstance(data["prediction"]["probability"], float)
        self.assertGreaterEqual(data["prediction"]["probability"], 0.0)
        self.assertLessEqual(data["prediction"]["probability"], 1.0)
        self.assertEqual(data["prediction"]["model_name"], "EfficientNet-B0")
        self.assertEqual(data["prediction"]["model_architecture"], "EfficientNet-B0")
        self.assertEqual(data["explanation"]["method"], "Grad-CAM")
        self.assertTrue(data["explanation"]["available"])
        self.assertEqual(data["explanation"]["details"]["target_layer"], "features.8.0")

    def test_12_oversized_image_upload_rejected(self):
        """Test uploading file exceeding MAX_UPLOAD_SIZE_BYTES (> 20 MB) is rejected with HTTP 400."""
        large_bytes = b"0" * (20 * 1024 * 1024 + 1024)  # > 20 MB
        response = self.client.post(
            "/api/v1/xray/analyze?task=pneumonia",
            files={"file": ("large_xray.png", large_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_13_corrupt_image_rejected(self):
        """Test uploading invalid/corrupt image bytes returns INVALID_INPUT error."""
        corrupt_bytes = b"not an image file content string"
        response = self.client.post(
            "/api/v1/skin/analyze?task=ham10000",
            files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_INPUT")

    def test_14_unsupported_task_error(self):
        """Test invalid task query parameter returns HTTP 422 validation error."""
        with open(self.image_path, "rb") as f:
            response = self.client.post(
                "/api/v1/xray/analyze?task=invalid_task",
                files={"file": ("xray.jpg", f, "image/jpeg")},
            )
        self.assertEqual(response.status_code, 422)

    def test_15_no_stack_trace_leakage(self):
        """Test error responses mask raw stack traces and internal machine paths."""
        response = self.client.post(
            "/api/v1/xray/analyze?task=pneumonia",
            files={"file": ("bad_file.txt", b"dummy", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("error", data)
        self.assertNotIn("Traceback", json.dumps(data))
        self.assertNotIn("File \"", json.dumps(data))


if __name__ == "__main__":
    unittest.main()
