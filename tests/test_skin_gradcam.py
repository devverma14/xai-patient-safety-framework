import hashlib
import json
from pathlib import Path
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "skin" / "binary" / "best_skin_binary_model.pt"
GRADCAM_DIR = PROJECT_ROOT / "evaluation" / "skin" / "gradcam"
SUMMARY_PATH = GRADCAM_DIR / "gradcam_summary.json"
EXPECTED_SHA256 = "f00304d1065147c16d52fee59fc21b3dc38bdf4e601d3630c78a06eec7e3cbda"
EXPECTED_TARGET_LAYER = "features.denseblock4.denselayer16.conv2"
EXPECTED_FROZEN_THRESHOLD = 0.40


def test_skin_binary_model_immutability():
    """Verify binary skin model checkpoint exists and SHA-256 matches expected hash."""
    assert MODEL_PATH.exists(), f"Skin binary model not found at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_SHA256, (
        f"Model hash mismatch! Expected {EXPECTED_SHA256}, got {file_hash}"
    )


def test_skin_gradcam_summary_artifact():
    """Verify gradcam_summary.json exists and contains correct metadata matching Phase 16.2 authoritative test evaluation."""
    assert SUMMARY_PATH.exists(), f"Grad-CAM summary missing at {SUMMARY_PATH}"
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    assert summary.get("model_architecture") == "densenet121"
    assert summary.get("target_layer") == EXPECTED_TARGET_LAYER
    assert summary.get("frozen_threshold") == EXPECTED_FROZEN_THRESHOLD
    assert summary.get("model_file_sha256") == EXPECTED_SHA256
    assert summary.get("model_immutability_verified") is True
    assert summary.get("total_explanations_generated", 0) >= 16

    auth_counts = summary.get("authoritative_test_confusion_counts", {})
    assert auth_counts.get("TP") == 263
    assert auth_counts.get("TN") == 753
    assert auth_counts.get("FP") == 456
    assert auth_counts.get("FN") == 14

    exp_counts = summary.get("explanation_category_counts", {})
    assert exp_counts.get("TP") == 4
    assert exp_counts.get("TN") == 4
    assert exp_counts.get("FP") == 4
    assert exp_counts.get("FN") == 4




def test_skin_gradcam_generated_images_and_stats():
    """Verify generated Grad-CAM heatmaps, overlays, and comparison images exist and meet numerical constraints."""
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    explanations = summary.get("explanations", [])
    assert len(explanations) >= 16

    for exp in explanations:
        artifacts = exp.get("artifacts", {})
        heatmap_path = PROJECT_ROOT / artifacts.get("heatmap_image", "")
        overlay_path = PROJECT_ROOT / artifacts.get("overlay_image", "")
        comp_path = PROJECT_ROOT / artifacts.get("comparison_figure", "")

        assert heatmap_path.exists(), f"Heatmap image missing: {heatmap_path}"
        assert overlay_path.exists(), f"Overlay image missing: {overlay_path}"
        assert comp_path.exists(), f"Comparison figure missing: {comp_path}"

        # Verify image loading & size
        with Image.open(heatmap_path) as img:
            assert img.size == (224, 224)
        with Image.open(overlay_path) as img:
            assert img.size == (224, 224)

        # Check numeric range & finite stats
        stats = exp.get("heatmap_stats", {})
        assert stats.get("is_finite") is True
        assert 0.0 <= stats.get("min") <= 1.0
        assert 0.0 <= stats.get("max") <= 1.0


def test_skin_gradcam_synthetic_forward_backward():
    """Verify live Grad-CAM forward/backward pass on target layer features.denseblock4.denselayer16.conv2."""
    from experiments.skin.train_skin_binary import build_binary_densenet121
    from xai.gradcam import GradCAM

    model = build_binary_densenet121()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model
    for name in EXPECTED_TARGET_LAYER.split("."):
        target_layer = getattr(target_layer, name)

    gradcam = GradCAM(model, target_layer=target_layer)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, target_cls, prob = gradcam.generate_heatmap(dummy_input, target_class=1)

    assert heatmap is not None
    assert heatmap.shape == (224, 224)
    assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all()
    assert 0.0 <= prob <= 1.0
    gradcam.remove_hooks()


