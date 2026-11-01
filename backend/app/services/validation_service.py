"""
Input validation and quality check service for uploaded report and image files.
Enforces file safety, extension whitelisting, size limits, and modality quality checks.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import logging

from backend.app.core.config import settings
from backend.app.core.exceptions import InvalidInputError
from ocr.quality import validate_document_quality

logger = logging.getLogger("backend.services.validation_service")


class ValidationService:
    """Modality input quality validation service."""

    @staticmethod
    def validate_file_safety(filename: str, file_size_bytes: int) -> str:
        """Verify extension and file size limits before processing."""
        if not filename:
            raise InvalidInputError("Uploaded file filename is missing.")

        ext = os.path.splitext(filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise InvalidInputError(
                f"File extension '{ext}' is not supported. Allowed extensions: {settings.ALLOWED_EXTENSIONS}",
                details={"filename": filename, "extension": ext, "allowed": settings.ALLOWED_EXTENSIONS},
            )

        if file_size_bytes == 0:
            raise InvalidInputError("Uploaded file is empty (0 bytes).")

        if file_size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
            size_mb = round(file_size_bytes / (1024 * 1024), 2)
            limit_mb = round(settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024), 2)
            raise InvalidInputError(
                f"File size ({size_mb} MB) exceeds maximum allowed upload limit ({limit_mb} MB).",
                details={"size_mb": size_mb, "limit_mb": limit_mb},
            )

        return ext

    @staticmethod
    def validate_lab_report(file_path: str) -> Dict[str, Any]:
        """Run Phase 7 pre-OCR document quality check for Lab reports."""
        check = validate_document_quality(file_path, max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES)
        return {
            "passed": check.passed,
            "file_type": check.file_type,
            "file_size_mb": check.file_size_mb,
            "error_message": check.error_message,
        }

    @staticmethod
    def validate_image_input(file_path: str) -> Dict[str, Any]:
        """Validate structural integrity and resolution of image uploads (X-Ray, Skin, Retina)."""
        if not os.path.exists(file_path):
            raise InvalidInputError(f"Image file not found at path.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            raise InvalidInputError(f"Image extension '{ext}' is not supported for image modalities.")

        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                w, h = img.size
                if w < 100 or h < 100:
                    return {
                        "passed": False,
                        "status": "failed",
                        "dimensions": (w, h),
                        "error_message": f"Image resolution ({w}x{h}) is too low (min 100x100 required).",
                    }

                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > 3.0:
                    return {
                        "passed": False,
                        "status": "failed",
                        "dimensions": (w, h),
                        "error_message": f"Image aspect ratio ({w}x{h}, ratio {aspect_ratio:.2f}) exceeds maximum allowed limit of 3.0:1.",
                    }

                # Image statistical and information integrity verification
                gray = img.convert("L")
                arr = np.asarray(gray, dtype=np.float32)
                std_val = float(np.std(arr))
                dyn_range = float(np.max(arr) - np.min(arr))

                # Reject blank, solid, or near-zero variance uploads
                if dyn_range < 5.0 or std_val < 2.0:
                    return {
                        "passed": False,
                        "status": "failed",
                        "dimensions": (w, h),
                        "error_message": (
                            f"Image has insufficient dynamic range or is blank (std={std_val:.2f}, range={dyn_range:.1f}). "
                            "Diagnostic images require sufficient contrast and anatomical structure."
                        ),
                    }

                # Reject uncorrelated high-entropy noise (e.g. uniform or Gaussian synthetic noise)
                diff_h = np.abs(np.diff(arr, axis=0))
                diff_w = np.abs(np.diff(arr, axis=1))
                mean_diff = float((np.mean(diff_h) + np.mean(diff_w)) / 2.0)
                diff_ratio = mean_diff / (std_val + 1e-8)

                row_means = arr - np.mean(arr, axis=1, keepdims=True)
                var_val = float(np.var(arr))
                lag1_cov = np.mean(row_means[:, :-1] * row_means[:, 1:])
                lag1_corr = float(lag1_cov / (var_val + 1e-8)) if var_val > 1e-6 else 0.0

                if diff_ratio > 0.85 and lag1_corr < 0.10:
                    return {
                        "passed": False,
                        "status": "failed",
                        "dimensions": (w, h),
                        "error_message": (
                            f"Image exhibits high-entropy uncorrelated noise without diagnostic structural features "
                            f"(diff_ratio={diff_ratio:.2f}, corr={lag1_corr:.3f}). Please provide an authentic diagnostic image."
                        ),
                    }

                return {
                    "passed": True,
                    "status": "passed",
                    "dimensions": (w, h),
                    "error_message": None,
                }
        except Exception as e:
            return {
                "passed": False,
                "status": "failed",
                "dimensions": None,
                "error_message": f"Corrupted image file: {str(e)}",
            }


validation_service = ValidationService()
