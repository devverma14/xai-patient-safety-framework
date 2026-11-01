# OCR & Lab Report Input Preparation Subsystem

This package provides an input preparation pipeline connecting real-world laboratory report PDFs and images to the validated Phase 5 XGBoost model and Phase 6 SHAP explainability engine.

## Subsystem Architecture

```
ocr/
├── __init__.py
├── schemas.py           # Dataclass schemas for quality checks, lab values, and payloads
├── quality.py           # Pre-OCR document quality validation (size, extension, header)
├── pdf.py               # PDF document reader (direct text vs scanned page handling)
├── image.py             # Image OCR reader (pytesseract & contrast pre-processing)
├── extraction.py        # Lab value regex extractor & PII redaction engine
├── normalization.py     # Test name synonym dictionary & unit standardization
├── validation.py        # Physiological bounds validator & duplicate conflict resolver
├── human_in_the_loop.py # Clinician manual review & override handler
├── pipeline.py          # End-to-end processing pipeline & ML preprocessor formatter
└── README.md
```

## Key Workflow Principles

1. **Input Preparation Only**: OCR prepares input data; it does NOT determine predictions.
2. **Quality Gating**: Uploads > 20 MB or corrupted files are rejected before OCR.
3. **PII Exclusion**: Patient names, IDs, phone numbers, and DOBs are automatically redacted.
4. **Physiological Bounds**: Values outside physiological ranges trigger `human_review_required = True`.
5. **Confidence Separation**: `ocr_confidence`, `extraction_confidence`, and `validation_status` are kept distinct from ML prediction probabilities.
6. **Integration with Phase 5 & 6**: Validated payloads are formatted into pandas DataFrames and passed to `LabPreprocessor` -> XGBoost (`threshold = 0.35`) -> SHAP `TreeExplainer`.
