"""
Document quality check and pre-OCR upload validation module.
Enforces SRS upload size limits, extension whitelisting, file header validation,
and image/PDF structural integrity checks before text extraction.
"""

import os
from typing import Tuple, Optional
from PIL import Image

from .schemas import DocumentQualityCheck

# SRS configurable maximum upload size limit (Default: 20 MB)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def validate_document_quality(
    file_path: str,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    allowed_extensions: Tuple[str, ...] = ALLOWED_EXTENSIONS,
) -> DocumentQualityCheck:
    """
    Perform pre-OCR document quality check on uploaded lab report files.

    Args:
        file_path: Path to uploaded PDF or image file.
        max_size_bytes: Maximum allowed file size in bytes (default: 20 MB).
        allowed_extensions: Tuple of allowed file extensions.

    Returns:
        DocumentQualityCheck dataclass instance.
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in allowed_extensions:
        return DocumentQualityCheck(
            passed=False,
            file_name=file_name,
            file_type="unsupported",
            file_size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            file_size_mb=0.0,
            error_message=f"Unsupported file extension '{ext}'. Allowed extensions: {allowed_extensions}",
        )

    if not os.path.exists(file_path):
        return DocumentQualityCheck(
            passed=False,
            file_name=file_name,
            file_type="unknown",
            file_size_bytes=0,
            file_size_mb=0.0,
            error_message=f"File not found at path: {file_path}",
        )

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    if file_size_bytes == 0:
        return DocumentQualityCheck(
            passed=False,
            file_name=file_name,
            file_type="unknown",
            file_size_bytes=0,
            file_size_mb=0.0,
            error_message="Uploaded file is empty (0 bytes).",
        )

    if file_size_bytes > max_size_bytes:
        return DocumentQualityCheck(
            passed=False,
            file_name=file_name,
            file_type="unknown",
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            error_message=f"File size ({file_size_mb} MB) exceeds maximum allowed upload limit ({max_size_bytes / (1024*1024):.1f} MB).",
        )

    # PDF File Validation
    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
            if not header.startswith(b"%PDF-"):
                return DocumentQualityCheck(
                    passed=False,
                    file_name=file_name,
                    file_type="pdf",
                    file_size_bytes=file_size_bytes,
                    file_size_mb=file_size_mb,
                    error_message="Invalid PDF file header.",
                )
            return DocumentQualityCheck(
                passed=True,
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_mb,
            )
        except Exception as e:
            return DocumentQualityCheck(
                passed=False,
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_mb,
                error_message=f"Failed to read PDF document: {str(e)}",
            )

    # Image File Validation
    else:
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                width, height = img.size
                if width < 100 or height < 100:
                    return DocumentQualityCheck(
                        passed=False,
                        file_name=file_name,
                        file_type="image",
                        file_size_bytes=file_size_bytes,
                        file_size_mb=file_size_mb,
                        image_dimensions=(width, height),
                        error_message=f"Image resolution ({width}x{height}) is too low for reliable OCR extraction (min 100x100).",
                    )
                return DocumentQualityCheck(
                    passed=True,
                    file_name=file_name,
                    file_type="image",
                    file_size_bytes=file_size_bytes,
                    file_size_mb=file_size_mb,
                    image_dimensions=(width, height),
                )
        except Exception as e:
            return DocumentQualityCheck(
                passed=False,
                file_name=file_name,
                file_type="image",
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_mb,
                error_message=f"Corrupted image file: {str(e)}",
            )
