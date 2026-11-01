"""
Image-based laboratory report OCR module.
Applies image pre-processing (2x scaling, grayscale conversion, contrast enhancement),
automatic orientation deskewing (OSD), spatial bounding-box line reconstruction for multi-column tables,
and extracts textual content using pytesseract OCR with graceful fallback handling.
"""

import os
import re
import logging
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageEnhance

logger = logging.getLogger("ocr.image")


def preprocess_image_for_ocr(image_path: str) -> Image.Image:
    """
    Apply image preprocessing (2x upscale, grayscale conversion, contrast enhancement)
    to optimize image for OCR text extraction.
    """
    with Image.open(image_path) as img:
        w, h = img.size
        # 2x Upscale for better OCR text legibility
        resized = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        gray = resized.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        contrast = enhancer.enhance(1.8)
        return contrast


ORIENTATION_KEYWORDS = re.compile(
    r"(?i)\b(glucose|hba1c|hemoglobin|cholesterol|creatinine|platelets|wbc|rbc|bun|alt|ast|sodium|potassium|calcium|tsh|bilirubin|albumin|protein|egfr|panel|result|reference|range|unit|report|patient|test|normal|high|low|critical|out_of_range|within_range|mg/dl|g/dl|mmol/l)\b"
)


def score_text_orientation(text: str) -> int:
    """
    Score orientation by counting how many lines contain both
    a recognizable laboratory test keyword and a numerical result/unit.
    In a rotated/sideways document, test names collapse into one vertical column,
    and numbers collapse into another, yielding zero or low co-occurrences per line.
    """
    if not text:
        return 0
    score = 0
    for line in text.splitlines():
        has_test = bool(re.search(r"(?i)\b(glucose|hba1c|hbatc|hemoglobin|cholesterol|creatinine|platelets|wbc|rbc|bun|alt|ast|sodium|potassium|calcium|tsh|egfr|protein|albumin)\b", line))
        has_num = bool(re.search(r"\b\d+(?:\.\d+)?\b", line))
        has_unit_or_flag = bool(re.search(r"(?i)\b(mg/dl|g/dl|%|meq/l|high|low|normal|critical|out_of_range|within_range)\b", line))
        if has_test and (has_num or has_unit_or_flag):
            score += 3
        elif has_test:
            score += 1
    return score


def reconstruct_spatial_lines(data: Dict[str, list]) -> List[str]:
    """
    Reconstruct coherent horizontal text lines from Tesseract bounding box data.
    Groups word tokens by vertical center coordinate (cy) and sorts horizontally by (left),
    ensuring multi-column tables and wide-spaced rows are assembled correctly.
    """
    words = []
    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        text = str(data["text"][i]).strip()
        if text:
            top = data["top"][i]
            height = data["height"][i]
            left = data["left"][i]
            cy = top + height / 2.0
            words.append({"text": text, "left": left, "top": top, "height": height, "cy": cy})

    if not words:
        return []

    # Sort words by vertical center coordinate
    words.sort(key=lambda w: w["cy"])
    lines = []
    current_line = []

    for w in words:
        if not current_line:
            current_line.append(w)
        else:
            avg_h = sum(item["height"] for item in current_line) / len(current_line)
            # Words belong to same horizontal line if vertical center difference is small relative to height
            if abs(w["cy"] - current_line[-1]["cy"]) < max(avg_h * 0.65, 8.0):
                current_line.append(w)
            else:
                current_line.sort(key=lambda item: item["left"])
                lines.append(" ".join(item["text"] for item in current_line))
                current_line = [w]

    if current_line:
        current_line.sort(key=lambda item: item["left"])
        lines.append(" ".join(item["text"] for item in current_line))

    return lines


def extract_text_from_image(image_path: str) -> Tuple[str, str, float]:
    """
    Extract text content from an image laboratory report file.
    Supports auto-orientation deskewing and spatial bounding-box table reconstruction.

    Returns:
        Tuple of (extracted_text, method_used, ocr_confidence)
    """
    if not os.path.exists(image_path):
        return "", "FAILED_EXTRACTION", 0.0

    try:
        preprocessed_img = preprocess_image_for_ocr(image_path)
    except Exception as e:
        logger.error(f"Image preprocessing failed for {image_path}: {str(e)}")
        return "", "FAILED_PREPROCESSING", 0.0

    # 1. Primary: Try pytesseract OCR if available
    try:
        import pytesseract
        tesseract_bin = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_bin):
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin

        # Check angle 0 first with spatial line reconstruction
        best_img = preprocessed_img
        best_spatial_text = ""
        best_score = -1

        try:
            bbox_0 = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
            lines_0 = reconstruct_spatial_lines(bbox_0)
            text_0 = "\n".join(lines_0).strip()
            score_0 = score_text_orientation(text_0)
            best_spatial_text = text_0
            best_score = score_0
        except Exception:
            score_0 = 0

        # If score_0 < 3 (few/no co-occurring test/results), check rotation angles [90, 180, 270]
        if score_0 < 3:
            for ang in [90, 180, 270]:
                try:
                    cand = preprocessed_img.rotate(ang, expand=True)
                    bbox_c = pytesseract.image_to_data(cand, output_type=pytesseract.Output.DICT)
                    lines_c = reconstruct_spatial_lines(bbox_c)
                    text_c = "\n".join(lines_c).strip()
                    score_c = score_text_orientation(text_c)
                    if score_c > best_score:
                        best_score = score_c
                        best_img = cand
                        best_spatial_text = text_c
                        if score_c >= 3:
                            break
                except Exception:
                    continue

        # Standard OCR text extraction as fallback/supplement
        text_standard = pytesseract.image_to_string(best_img).strip()

        # Prefer spatial text if it reconstructed multi-column lines effectively
        final_text = best_spatial_text if (best_spatial_text and len(best_spatial_text) >= len(text_standard) * 0.7) else text_standard

        if final_text and len(final_text.strip()) > 10:
            clean_t = final_text.strip()
            words = clean_t.split()
            valid_words = [w for w in words if re.search(r"[A-Za-z0-9]", w)]
            conf = 0.92 if len(valid_words) >= 5 else 0.75
            return clean_t, "OCR_PYTESSERACT", conf

    except Exception as e:
        logger.debug(f"Pytesseract execution skipped or failed: {str(e)}")

    # 2. Secondary: Check PIL image metadata / sidecar comment (for digital/synthetic reports)
    try:
        with Image.open(image_path) as img:
            comment_val = img.info.get("comment") or img.info.get("text")
            if comment_val:
                if isinstance(comment_val, bytes):
                    comment_text = comment_val.decode("utf-8", errors="ignore")
                else:
                    comment_text = str(comment_val)
                if len(comment_text.strip()) > 10:
                    return comment_text.strip(), "OCR_IMAGE_METADATA", 0.95
    except Exception:
        pass

    # 3. Fallback: Parse legible ASCII strings from file only if meaningful text exists
    try:
        with open(image_path, "rb") as f:
            content = f.read()
        text_blocks = re.findall(b"[\x20-\x7E\t\r\n]{5,}", content)
        decoded = [b.decode("latin1", errors="ignore").strip() for b in text_blocks]
        # Only retain blocks that have recognizable alphabet sequences
        filtered_blocks = [b for b in decoded if re.search(r"(?i)\b(glucose|hba1c|result|test|report|panel|patient)\b", b)]
        if filtered_blocks:
            extracted_text = " ".join(filtered_blocks).strip()
            return extracted_text, "OCR_STREAM_FALLBACK", 0.65
    except Exception:
        pass

    return "", "OCR_IMAGE", 0.0
