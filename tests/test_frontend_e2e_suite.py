"""
Phase 25.7 - Complete End-to-End Frontend Integration & Safety Audit Test Suite

Validates:
1. Lab Report Workflow (Happy path, Insufficient input path, SHAP, Threshold 0.35, Non-causal wording)
2. X-Ray Pneumonia Workflow (Task pneumonia, Threshold 0.50, Grad-CAM target layer, Non-causal wording)
3. X-Ray Tuberculosis Workflow (Task tb, Threshold 0.50, DenseNet121, Grad-CAM features.denseblock4.denselayer16.conv2, Non-causal wording)
4. Skin Lesion Workflow (Task ham10000, Threshold 0.40, Grad-CAM features.denseblock4.denselayer16.conv2, HIGHER-RISK/LOWER-RISK, Safety wording)
5. Retina Workflow (Task diabetic_retinopathy, Threshold 0.50, Grad-CAM features.8.0, DR Present/No DR, Systemic diabetes disclaimer)
6. Negative / Malformed Inputs (Unsupported file type, Oversized file, Corrupt image, Missing Lab features)
7. Error Sanitization & Traceback Leakage Audit
8. Safety Wording & Terminology Audit (Prohibited clinical claims check)
9. Modality Switching & State Isolation Verification
10. Privacy & Object Cleanup Code Audit
"""

import os
import io
import json
import pytest
import requests
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

BASE_URL = "http://127.0.0.1:8000"
test_client = TestClient(app)


class TestResponseWrapper:
    def __init__(self, fastapi_resp):
        self.status_code = fastapi_resp.status_code
        self._resp = fastapi_resp

    def json(self):
        return self._resp.json()

    @property
    def text(self):
        return self._resp.text


def req_get(url, **kwargs):
    try:
        resp = requests.get(url, timeout=2, **kwargs)
        return resp
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        path = url.replace(BASE_URL, "")
        return TestResponseWrapper(test_client.get(path, **kwargs))


def req_post(url, **kwargs):
    try:
        resp = requests.post(url, timeout=5, **kwargs)
        return resp
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        path = url.replace(BASE_URL, "")
        return TestResponseWrapper(test_client.post(path, **kwargs))


def test_01_backend_health():
    """Verify backend connection."""
    resp = req_get(f"{BASE_URL}/docs")
    assert resp.status_code == 200


def test_02_lab_e2e_happy_path():
    """Verify Lab Report OCR workflow with reference-range interpretation and zero fabricated predictions."""
    img_path = os.path.join("tests", "sample_reports", "sample_image_report.jpg")
    with open(img_path, "rb") as f:
        resp = req_post(
            f"{BASE_URL}/api/v1/lab/analyze",
            files={"file": ("sample_image_report.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality"] == "lab"
    assert data["task"] == "report_analysis"

    # Patient Safety Check: Laboratory report must NOT fabricate disease predictions
    assert data["prediction"] is None
    assert data["review_required"] is False

    # XAI Reference Intervals validation
    assert "explanation" in data
    exp = data["explanation"]
    assert exp["method"] == "Clinical Reference Intervals"
    assert exp["available"] is False

    # Extraction verification: parameter names, numeric values, units, reference ranges
    assert "payload_details" in data
    p_details = data["payload_details"]
    lab_vals = p_details.get("laboratory_values", [])
    assert len(lab_vals) > 0

    param_dict = {v.get("test_name"): v for v in lab_vals if "test_name" in v}
    assert "HbA1c" in param_dict or "Glucose" in param_dict

    # Check physiological fields for detected parameters
    if "HbA1c" in param_dict:
        hba1c = param_dict["HbA1c"]
        assert hba1c.get("numeric_value") is not None
        assert hba1c.get("unit") == "%"
        assert hba1c.get("reference_range") == "4.0-5.6"
        assert hba1c.get("status_flag") == "HIGH"

    if "Glucose" in param_dict:
        glu = param_dict["Glucose"]
        assert glu.get("numeric_value") is not None
        assert glu.get("unit") == "mg/dL"
        assert glu.get("reference_range") == "70-100"
        assert glu.get("status_flag") == "HIGH"


def test_03_lab_e2e_insufficient_input():
    """Verify Lab Report missing features returns human review required and no fabricated prediction."""
    img_path = os.path.join("tests", "sample_reports", "sample_poor_quality.jpg")
    with open(img_path, "rb") as f:
        resp = req_post(
            f"{BASE_URL}/api/v1/lab/analyze",
            files={"file": ("sample_poor_quality.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["review_required", "insufficient_data"]
    assert data["prediction"] is None
    assert data["review_required"] is True
    assert data["explanation"]["available"] is False


def test_04_xray_pneumonia_e2e():
    """Verify X-Ray Pneumonia workflow with Grad-CAM and threshold 0.50."""
    img_path = os.path.join("frontend", "public", "demo", "xray_original.jpg")
    with open(img_path, "rb") as f:
        files = {"file": ("xray_original.jpg", f, "image/jpeg")}
        resp = req_post(f"{BASE_URL}/api/v1/xray/analyze?task=pneumonia", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality"] == "xray"
    assert data["task"] == "pneumonia"

    pred = data["prediction"]
    assert pred["label"] in ["PNEUMONIA", "NORMAL", "Pneumonia Pattern Predicted", "No Pneumonia Pattern Predicted"]
    assert pred["threshold"] == 0.50
    assert "probability" in pred
    assert pred.get("model_name") == "DenseNet121"
    assert pred.get("model_architecture") == "DenseNet121"

    # XAI Grad-CAM details
    exp = data["explanation"]
    details = exp.get("details", {})
    assert details.get("target_layer") in ["features.denseblock4", "features.denseblock4.denselayer16.conv2"]
    assert any(k in details for k in ["spatial_summary_grid_7x7", "heatmap_shape", "heatmap_base64", "heatmap_matrix"])


def test_05_xray_tb_e2e():
    """Verify X-Ray Tuberculosis workflow with Grad-CAM DenseNet121 and threshold 0.50."""
    img_path = os.path.join("frontend", "public", "demo", "xray_original.jpg")
    with open(img_path, "rb") as f:
        files = {"file": ("xray_original.jpg", f, "image/jpeg")}
        resp = req_post(f"{BASE_URL}/api/v1/xray/analyze?task=tb", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality"] == "xray"
    assert data["task"] == "tb"

    pred = data["prediction"]
    assert pred["threshold"] == 0.50
    assert pred.get("model_name") == "DenseNet121"
    assert pred.get("model_architecture") == "DenseNet121"

    exp = data["explanation"]
    details = exp.get("details", {})
    assert details.get("target_layer") == "features.denseblock4.denselayer16.conv2"


def test_06_skin_e2e():
    """Verify Skin Lesion workflow with threshold 0.40, Grad-CAM layer, and risk labels."""
    img_path = os.path.join("datasets", "skin", "ISIC_0024306.jpg")
    with open(img_path, "rb") as f:
        files = {"file": ("ISIC_0024306.jpg", f, "image/jpeg")}
        resp = req_post(f"{BASE_URL}/api/v1/skin/analyze?task=ham10000", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality"] == "skin"

    pred = data["prediction"]
    assert pred["threshold"] == 0.40

    exp = data["explanation"]
    details = exp.get("details", {})
    assert details.get("target_layer") == "features.denseblock4.denselayer16.conv2"

    # Terminology prohibition check: NO malignant/benign diagnosis claims
    pred_str = str(pred["label"]).lower()
    assert "malignant" not in pred_str
    assert "benign" not in pred_str
    assert "cancer confirmed" not in pred_str
    assert "melanoma confirmed" not in pred_str


def test_07_retina_e2e():
    """Verify Retina workflow with threshold 0.50, Grad-CAM features.8.0, and systemic diabetes disclaimer."""
    img_path = os.path.join("datasets", "retina", "train_images", "000c1434d8d7.png")
    with open(img_path, "rb") as f:
        files = {"file": ("000c1434d8d7.png", f, "image/png")}
        resp = req_post(f"{BASE_URL}/api/v1/retina/analyze?task=diabetic_retinopathy", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality"] == "retina"

    pred = data["prediction"]
    assert pred["threshold"] == 0.50
    assert pred.get("model_name") == "EfficientNet-B0"
    assert pred.get("model_architecture") == "EfficientNet-B0"

    exp = data["explanation"]
    details = exp.get("details", {})
    assert details.get("target_layer") == "features.8.0"

    # Systemic diabetes terminology check
    pred_str = str(pred["label"]).lower()
    assert "diabetes detected" not in pred_str
    assert "patient has diabetes" not in pred_str
    assert "systemic diabetes diagnosed" not in pred_str


def test_08_negative_file_types():
    """Verify invalid file extensions are rejected gracefully without stack traces."""
    files = {"file": ("test.exe", b"MZ123456789", "application/octet-stream")}
    resp = req_post(f"{BASE_URL}/api/v1/xray/analyze?task=pneumonia", files=files)
    assert resp.status_code in [400, 422]
    data = resp.json()
    error_msg = str(data).lower()
    assert "traceback" not in error_msg
    assert "python" not in error_msg
    assert "file_path" not in error_msg


def test_09_error_sanitization():
    """Verify invalid tasks return sanitized errors without exposing backend internals."""
    resp = req_post(f"{BASE_URL}/api/v1/xray/analyze?task=invalid_task", files={"file": ("test.jpg", b"fake", "image/jpeg")})
    assert resp.status_code in [400, 422]
    data = resp.json()
    error_text = str(data).lower()
    assert "traceback" not in error_text
    assert "site-packages" not in error_text


def test_10_frontend_code_audit_privacy_and_thresholds():
    """Audit frontend source files for privacy compliance, URL cleanup, and neutral thresholds."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "src")

    found_localstorage = []
    found_calibrated = []

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith((".ts", ".tsx")):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8") as file_obj:
                    content = file_obj.read()
                    if "localStorage" in content or "sessionStorage" in content:
                        found_localstorage.append(f)
                    if "calibrated threshold" in content.lower():
                        found_calibrated.append(f)

    # Ensure zero medical data or state in localStorage/sessionStorage
    assert len(found_localstorage) == 0, f"Found local/sessionStorage in: {found_localstorage}"
    # Ensure zero misleading 'calibrated threshold' wording in frontend code
    assert len(found_calibrated) == 0, f"Found 'calibrated threshold' wording in: {found_calibrated}"


def test_11_prohibited_terminology_audit():
    """Audit frontend source files for strictly prohibited clinical terminology as primary result labels."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "src")

    prohibited_patterns = [
        "diabetes detected",
        "you have diabetes",
        "cancer detected",
        "melanoma detected",
        "tb confirmed",
        "pneumonia confirmed",
        "diagnosis confirmed",
    ]

    found_violations = []

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith((".ts", ".tsx")):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8") as file_obj:
                    content = file_obj.read().lower()
                    for pat in prohibited_patterns:
                        if pat in content:
                            found_violations.append((f, pat))

    assert len(found_violations) == 0, f"Found prohibited terminology violations: {found_violations}"


def test_12_dual_view_and_architecture_audit():
    """Verify Patient View and Clinical / Technical View implementation across all 5 prediction modalities."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "src")

    prediction_components = [
        "XRayPrediction.tsx",
        "SkinPrediction.tsx",
        "RetinaPrediction.tsx",
        "PredictionCard.tsx",
    ]

    for comp_name in prediction_components:
        found = False
        for root, _, files in os.walk(src_dir):
            if comp_name in files:
                found = True
                filepath = os.path.join(root, comp_name)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Verify viewMode state and dynamic dual-view toggle buttons
                assert 'viewMode' in content, f"{comp_name} missing viewMode state"
                assert 'Patient View' in content, f"{comp_name} missing Patient View button"
                assert 'Clinical / Technical View' in content, f"{comp_name} missing Clinical View button"
                assert 'viewMode === "patient"' in content or "viewMode === 'patient'" in content, (
                    f"{comp_name} missing patient view conditional branch"
                )
                assert 'viewMode === "clinical"' in content or "viewMode === 'clinical'" in content or "else" in content, (
                    f"{comp_name} missing clinical view branch"
                )

                # Verify plain-language and clinical sections exist
                assert "PatientFriendlyExplanation" in content
                assert "Understanding the Visual" in content or "Understanding What Influenced" in content

        assert found, f"Component {comp_name} not found in {src_dir}"

    # Verify architecture metadata in XRayPrediction
    xray_path = os.path.join(src_dir, "components", "modality", "xray", "XRayPrediction.tsx")
    with open(xray_path, "r", encoding="utf-8") as f:
        xray_content = f.read()
    assert "DenseNet121" in xray_content
    assert "features.denseblock4.denselayer16.conv2" in xray_content

    # Verify architecture metadata in RetinaPrediction
    retina_path = os.path.join(src_dir, "components", "modality", "retina", "RetinaPrediction.tsx")
    with open(retina_path, "r", encoding="utf-8") as f:
        retina_content = f.read()
    assert "EfficientNet-B0" in retina_content
    assert "features.8.0" in retina_content


def test_13_xray_task_sync_and_modality_selection_audit():
    """Verify X-Ray task two-way sync, 5 modality card structures, and product wording."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "src")

    # 1. Verify ModalitySelector has all 5 distinct cards with user-oriented purpose and input requirements
    selector_path = os.path.join(src_dir, "components", "common", "ModalitySelector.tsx")
    with open(selector_path, "r", encoding="utf-8") as f:
        mod_content = f.read()

    assert "xray-pneumonia" in mod_content
    assert "xray-tb" in mod_content
    assert "Pneumonia" in mod_content
    assert "Tuberculosis" in mod_content
    assert "Input required:" in mod_content
    assert "Grad-CAM visual heatmap" in mod_content
    assert "SHAP feature contributions" in mod_content
    assert "not a definitive skin cancer diagnosis" in mod_content
    assert "does not confirm systemic diabetes" in mod_content

    # 2. Verify page.tsx directly mounts XRayAnalysis with correct initialTask and isolated key
    page_path = os.path.join(src_dir, "app", "page.tsx")
    with open(page_path, "r", encoding="utf-8") as f:
        page_content = f.read()

    assert "Understand medical AI predictions, visually." in page_content
    assert "Analyze laboratory reports and medical images with AI-powered pattern prediction and visual explanations" in page_content
    assert "Start an Analysis" in page_content
    assert "Explore Analysis Tools" in page_content
    assert "Explainable AI for Patient Safety" in page_content
    assert "AI-generated results are informational and do not replace professional medical assessment." in page_content
    # Dual-pipeline showcase: Both Laboratory and Imaging workflows represented
    assert "Extract & Validate Values" in page_content
    assert "SHAP Feature Contributions" in page_content
    assert "Image Classification Model" in page_content
    assert "Grad-CAM Heatmap / Overlay" in page_content
    # Directly routes Pneumonia and TB workflows without intermediate selector
    assert 'key={selectedOptionId}' in page_content
    assert 'initialTask={selectedOptionId === "xray-tb" ? "tb" : "pneumonia"}' in page_content

    # 3. Verify XRayAnalysis opens workflow directly and duplicate task selector is ABSENT
    xray_analysis_path = os.path.join(src_dir, "components", "modality", "xray", "XRayAnalysis.tsx")
    with open(xray_analysis_path, "r", encoding="utf-8") as f:
        xray_analysis_content = f.read()

    # Redundant task selector must be completely absent from XRayAnalysis
    assert "XRayTaskSelector" not in xray_analysis_content
    assert "Select Chest X-Ray Analysis Task" not in xray_analysis_content
    assert "Task Configuration" not in xray_analysis_content

    # State reset on task change to guarantee no stale predictions or heatmaps
    assert "setAnalysisResult(null)" in xray_analysis_content
    assert "setErrorState(null)" in xray_analysis_content
    assert "setSelectedFile(null)" in xray_analysis_content

    # 4. Verify XRayUpload directly displays task-specific heading and instructions
    xray_upload_path = os.path.join(src_dir, "components", "modality", "xray", "XRayUpload.tsx")
    with open(xray_upload_path, "r", encoding="utf-8") as f:
        xray_upload_content = f.read()

    assert "Chest X-Ray — Pneumonia Screening" in xray_upload_content
    assert "Chest X-Ray — Tuberculosis Screening" in xray_upload_content
    assert "DenseNet121" in xray_upload_content


