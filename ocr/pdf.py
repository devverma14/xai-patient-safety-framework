"""
PDF document processing and text extraction module.
Determines whether a PDF contains selectable text or scanned page images,
executes page-by-page text extraction for digital multi-page PDFs,
and executes image extraction with OCR fallback for scanned PDFs.
"""

import os
import io
import re
import logging
from typing import Tuple, List, Optional
from PIL import Image

logger = logging.getLogger("ocr.pdf")


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str, float]:
    """
    Extract text content from a PDF report file, supporting multi-page reports and scanned documents.

    Returns:
        Tuple of (extracted_text, method_used, extraction_confidence)
        where method_used is 'DIRECT_TEXT_PDF' or 'OCR_SCANNED_PDF'.
    """
    if not os.path.exists(pdf_path):
        return "", "FAILED_EXTRACTION", 0.0

    pages_text: List[str] = []
    has_scanned_images = False
    scanned_ocr_text: List[str] = []

    # 1. Page-by-page extraction using pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)

        for p_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(f"--- Page {p_idx + 1} ---\n{page_text.strip()}")

            # If page text is very sparse or empty, inspect page for embedded scanned images
            if len(page_text.strip()) < 20 and hasattr(page, "images") and len(page.images) > 0:
                has_scanned_images = True
                try:
                    import pytesseract
                    from .image import preprocess_image_for_ocr

                    for img_obj in page.images:
                        img_bytes = img_obj.data
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            # Preprocess extracted page image
                            w, h = pil_img.size
                            if w > 100 and h > 100:
                                resized = pil_img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
                                gray = resized.convert("L")
                                text_from_img = pytesseract.image_to_string(gray)
                                if text_from_img and len(text_from_img.strip()) > 10:
                                    scanned_ocr_text.append(f"--- Page {p_idx + 1} (OCR) ---\n{text_from_img.strip()}")
                except Exception as img_err:
                    logger.debug(f"Could not OCR embedded image on page {p_idx + 1}: {str(img_err)}")

    except Exception as e:
        logger.warning(f"pypdf reader encountered error reading {pdf_path}: {str(e)}")

    # 2. Evaluate direct text extraction results
    full_direct_text = "\n\n".join(pages_text).strip()
    if len(full_direct_text) >= 20:
        return full_direct_text, "DIRECT_TEXT_PDF", 0.95

    # 3. If direct text was empty or sparse, evaluate scanned OCR results
    full_scanned_text = "\n\n".join(scanned_ocr_text).strip()
    if len(full_scanned_text) >= 20:
        return full_scanned_text, "OCR_SCANNED_PDF", 0.88

    # 4. Fallback ASCII stream parsing if document is not standard PDF stream
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
        text_blocks = re.findall(b"[\x20-\x7E\t\r\n]{4,}", content)
        raw_str = " ".join([b.decode("latin1", errors="ignore") for b in text_blocks])
        filtered = re.sub(r"/Font|/Type|/Subtype|/Page|/Catalog|/Parent|/Resources|/MediaBox|/Contents", "", raw_str).strip()
        if len(filtered) >= 30 and ("glucose" in filtered.lower() or "hba1c" in filtered.lower() or "result" in filtered.lower()):
            return filtered, "STREAM_PARSED_PDF", 0.70
    except Exception:
        pass

    # Empty or unreadable PDF
    return full_direct_text or full_scanned_text, "OCR_SCANNED_PDF", 0.0
