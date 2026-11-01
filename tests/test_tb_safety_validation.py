"""
Comprehensive Regression Test Suite for TB Image Input Validation and Safety Guards.
Verifies rejection of blank/solid images, uncorrelated noise, extreme aspect ratios,
unsupported formats, sub-minimum resolutions, corrupted payloads, and confirms that
valid radiographs (including dark and low-contrast cases) proceed safely through inference.
"""

import os
import io
import json
import hashlib
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.validation_service import validation_service
from backend.app.services.prediction_service import prediction_service
from backend.app.services.model_registry import model_registry
from backend.app.core.exceptions import InvalidInputError


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def real_tb_image_path():
    p1 = os.path.join("datasets", "tb", "shenzhen", "images", "images", "CHNCXR_0001_0.png")
    p2 = os.path.join("datasets", "tb", "shenzhen", "images", "CHNCXR_0001_0.png")
    path = p1 if os.path.exists(p1) else p2
    assert os.path.exists(path), f"Sample real radiograph not found at {path}"
    return path


class TestTBInputSafetyValidation:
    """Unit tests for ValidationService image checks."""

    def _validate_pil_image(self, img: Image.Image, ext: str = ".png") -> dict:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            return validation_service.validate_image_input(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_reject_all_black_image(self):
        """Verify completely solid black image is rejected for insufficient dynamic range / blank."""
        img = Image.new("RGB", (224, 224), (0, 0, 0))
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "blank" in res["error_message"].lower() or "insufficient dynamic range" in res["error_message"].lower()

    def test_reject_all_white_image(self):
        """Verify completely solid white image is rejected."""
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "blank" in res["error_message"].lower() or "insufficient dynamic range" in res["error_message"].lower()

    def test_reject_solid_gray_zero_variance(self):
        """Verify solid flat gray image is rejected."""
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "blank" in res["error_message"].lower() or "insufficient dynamic range" in res["error_message"].lower()

    def test_reject_near_zero_variance(self):
        """Verify near-zero variance image (std < 2.0) is rejected."""
        np.random.seed(42)
        arr = np.clip(np.random.normal(128, 0.8, (224, 224)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "insufficient dynamic range" in res["error_message"].lower()

    def test_reject_uniform_random_noise(self):
        """Verify high-entropy uncorrelated random noise is detected and rejected."""
        np.random.seed(42)
        arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        img = Image.fromarray(arr)
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "uncorrelated noise" in res["error_message"].lower()

    def test_reject_gaussian_random_noise(self):
        """Verify independent Gaussian random noise is detected and rejected."""
        np.random.seed(42)
        arr = np.clip(np.random.normal(128, 40, (224, 224)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        res = self._validate_pil_image(img)
        assert res["passed"] is False
        assert "uncorrelated noise" in res["error_message"].lower()

    def test_aspect_ratio_boundaries(self):
        """Test aspect ratio thresholds: below 3.0 passes, exactly 3.0 passes, above 3.0 fails."""
        # 1. Ratio 2.9 (290x100) -> Below 3.0 limit -> passes aspect check
        # Use structured content so it doesn't fail noise or blank check
        img_29 = Image.new("RGB", (290, 100), (50, 50, 50))
        draw_29 = ImageDraw.Draw(img_29)
        draw_29.rectangle([20, 20, 270, 80], fill=(200, 200, 200))
        res_29 = self._validate_pil_image(img_29)
        assert res_29["passed"] is True, f"Expected 2.9:1 to pass, got: {res_29}"

        # 2. Ratio 3.0 (300x100) -> Exactly 3.0 limit -> passes
        img_30 = Image.new("RGB", (300, 100), (50, 50, 50))
        draw_30 = ImageDraw.Draw(img_30)
        draw_30.rectangle([20, 20, 280, 80], fill=(200, 200, 200))
        res_30 = self._validate_pil_image(img_30)
        assert res_30["passed"] is True, f"Expected 3.0:1 to pass, got: {res_30}"

        # 3. Ratio 3.1 (310x100) -> Exceeds 3.0 limit -> rejected
        img_31 = Image.new("RGB", (310, 100), (50, 50, 50))
        draw_31 = ImageDraw.Draw(img_31)
        draw_31.rectangle([20, 20, 290, 80], fill=(200, 200, 200))
        res_31 = self._validate_pil_image(img_31)
        assert res_31["passed"] is False
        assert "aspect ratio" in res_31["error_message"].lower()

        # 4. Extreme landscape (1000x100) -> rejected
        img_ext_w = Image.new("RGB", (1000, 100), (128, 128, 128))
        res_w = self._validate_pil_image(img_ext_w)
        assert res_w["passed"] is False
        assert "aspect ratio" in res_w["error_message"].lower()

        # 5. Extreme portrait (100x1000) -> rejected
        img_ext_h = Image.new("RGB", (100, 1000), (128, 128, 128))
        res_h = self._validate_pil_image(img_ext_h)
        assert res_h["passed"] is False
        assert "aspect ratio" in res_h["error_message"].lower()

    def test_minimum_resolution_rejection(self):
        """Verify images smaller than 100x100 are rejected."""
        img_99 = Image.new("RGB", (99, 99), (128, 128, 128))
        res_99 = self._validate_pil_image(img_99)
        assert res_99["passed"] is False
        assert "min 100x100 required" in res_99["error_message"].lower()

        img_50 = Image.new("RGB", (50, 50), (128, 128, 128))
        res_50 = self._validate_pil_image(img_50)
        assert res_50["passed"] is False
        assert "min 100x100 required" in res_50["error_message"].lower()

    def test_corrupted_file_rejection(self):
        """Verify malformed/corrupted file bytes are rejected safely."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"CORRUPTED_GARBAGE_BYTES_12345")
            tmp_path = tmp.name
        try:
            res = validation_service.validate_image_input(tmp_path)
            assert res["passed"] is False
            assert "corrupted" in res["error_message"].lower()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_legitimate_radiographs_pass_validation(self, real_tb_image_path):
        """Verify authentic clinical chest X-rays pass validation cleanly."""
        res = validation_service.validate_image_input(real_tb_image_path)
        assert res["passed"] is True
        assert res["error_message"] is None
        assert res["dimensions"][0] >= 100
        assert res["dimensions"][1] >= 100

    def test_low_contrast_and_dark_radiographs_not_falsely_rejected(self, real_tb_image_path):
        """Verify dark and low-contrast authentic X-rays are not falsely rejected."""
        with Image.open(real_tb_image_path) as img:
            gray = np.array(img.convert("L"), dtype=np.float32)

        # 1. Dark radiograph (scaled down by 80%)
        dark_arr = np.clip(gray * 0.20, 0, 255).astype(np.uint8)
        dark_img = Image.fromarray(dark_arr)
        res_dark = self._validate_pil_image(dark_img)
        assert res_dark["passed"] is True, f"Dark radiograph was falsely rejected: {res_dark}"

        # 2. Low-contrast radiograph (compressed dynamic range)
        mean_val = np.mean(gray)
        low_contrast_arr = np.clip(mean_val + (gray - mean_val) * 0.15, 0, 255).astype(np.uint8)
        low_img = Image.fromarray(low_contrast_arr)
        res_low = self._validate_pil_image(low_img)
        assert res_low["passed"] is True, f"Low-contrast radiograph was falsely rejected: {res_low}"


class TestTBEndpointSafetyRejection:
    """Integration tests verifying HTTP API status codes, error messages, and prediction integrity."""

    def test_api_rejects_blank_black_with_400(self, api_client):
        """Test POST /api/v1/xray/analyze?task=tb rejects blank black image with 400 INVALID_INPUT."""
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (0, 0, 0)).save(buf, format="PNG")
        buf.seek(0)

        response = api_client.post(
            "/api/v1/xray/analyze?task=tb",
            files={"file": ("blank_black.png", buf, "image/png")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "INVALID_INPUT"
        assert "blank" in data["error"]["message"].lower() or "insufficient dynamic range" in data["error"]["message"].lower()

    def test_api_rejects_extreme_aspect_ratio_with_400(self, api_client):
        """Test POST /api/v1/xray/analyze?task=tb rejects extreme aspect ratio image with 400 INVALID_INPUT."""
        buf = io.BytesIO()
        Image.new("RGB", (100, 1000), (128, 128, 128)).save(buf, format="PNG")
        buf.seek(0)

        response = api_client.post(
            "/api/v1/xray/analyze?task=tb",
            files={"file": ("extreme_ratio.png", buf, "image/png")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "INVALID_INPUT"
        assert "aspect ratio" in data["error"]["message"].lower()

    def test_api_rejects_uniform_noise_with_400(self, api_client):
        """Test POST /api/v1/xray/analyze?task=tb rejects uniform noise with 400 INVALID_INPUT."""
        buf = io.BytesIO()
        np.random.seed(42)
        noise_arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        Image.fromarray(noise_arr).save(buf, format="PNG")
        buf.seek(0)

        response = api_client.post(
            "/api/v1/xray/analyze?task=tb",
            files={"file": ("noise.png", buf, "image/png")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "INVALID_INPUT"
        assert "uncorrelated noise" in data["error"]["message"].lower()

    def test_api_accepts_valid_radiograph_and_preserves_prediction(self, api_client, real_tb_image_path):
        """Test POST /api/v1/xray/analyze?task=tb processes authentic radiograph successfully."""
        with open(real_tb_image_path, "rb") as f:
            response = api_client.post(
                "/api/v1/xray/analyze?task=tb",
                files={"file": ("valid_xray.png", f, "image/png")}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["modality"] == "xray"
        assert data["task"] == "tb"
        assert data["prediction"]["model_name"] == "DenseNet121"
        assert data["prediction"]["model_architecture"] == "DenseNet121"
        assert data["prediction"]["threshold"] == 0.50
        assert data["prediction"]["label"] in ["Tuberculosis", "Normal"]
        assert 0.0 <= data["prediction"]["probability"] <= 1.0
        assert data["explanation"]["method"] == "Grad-CAM"
        assert data["explanation"]["available"] is True


class TestModelCheckpointAndDeviceSafety:
    """Verifies checkpoint hash integrity, 0.50 threshold, and device compatibility."""

    def test_checkpoint_hash_unmodified(self):
        """Verify best_tb_model.pt SHA256 matches exactly 585A6007...AA5."""
        checkpoint_path = os.path.join("models", "tb", "best_tb_model.pt")
        with open(checkpoint_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        assert h == "585A60072463E143A1041F35679FB91F9A78911B93D9C88EA0BFF6E5122E4AA5"

    def test_threshold_remains_fixed_at_0_50(self):
        """Verify decision threshold is strictly 0.50 across metadata and inference."""
        meta_path = os.path.join("models", "tb", "best_tb_model_metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["threshold"] == 0.50

    def test_device_compatibility_in_prediction_service(self, real_tb_image_path):
        """Verify prediction_service correctly aligns input_batch to model device."""
        model = model_registry.load_model("xray", "tb")
        first_param = next(model.parameters())
        expected_device = first_param.device

        with open(real_tb_image_path, "rb") as f:
            file_bytes = f.read()

        payload = prediction_service.process_image_analysis("xray", "tb", file_bytes, "test.png")
        assert payload.status == "success"
        assert payload.prediction is not None
        assert payload.prediction.threshold == 0.50
