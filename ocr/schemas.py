"""
Data schemas and models for OCR extraction, document quality checks,
laboratory value validation, and human-in-the-loop review.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentQualityCheck:
    """Document quality and pre-OCR upload validation status."""

    passed: bool
    file_name: str
    file_type: str  # 'pdf' or 'image'
    file_size_bytes: int
    file_size_mb: float
    error_message: Optional[str] = None
    image_dimensions: Optional[tuple] = None  # (width, height)
    page_count: Optional[int] = None


@dataclass
class ExtractedLabValue:
    """Represents an individual laboratory test parameter extracted from report text."""

    test_name: str  # Raw extracted test name (e.g., 'HbA1c', 'Glucose')
    normalized_name: str  # Mapped feature key (e.g., 'A1Cresult', 'max_glu_serum')
    raw_value_str: str  # Original extracted string (e.g., '7.8', '>200')
    numeric_value: Optional[float]  # Cast float value if applicable
    unit: Optional[str]  # Extracted or verified unit (e.g., '%', 'mg/dL')
    reference_range: Optional[str]  # Extracted or verified reference range (e.g., '4.0 - 5.6')
    status_flag: Optional[str]  # 'NORMAL', 'HIGH', 'LOW', 'CRITICAL'
    extraction_confidence: float  # Extraction confidence (0.0 to 1.0)
    validation_status: str  # 'VALID', 'REVIEW_REQUIRED', 'OUT_OF_BOUNDS'
    validation_message: Optional[str] = None
    source_line: Optional[str] = None  # Text line source
    source_context: Optional[str] = None  # Traceable text context
    reference_range_source: Optional[str] = None  # 'REPORT' | 'STANDARD_REFERENCE' | None
    interpretation_status: str = "UNABLE_TO_DETERMINE"  # 'WITHIN_RANGE' | 'HIGH' | 'LOW' | 'UNABLE_TO_DETERMINE'
    explanation: Optional[Dict[str, Any]] = None  # Patient-friendly explanation payload
    is_supported: bool = True


@dataclass
class HumanReviewCorrection:
    """Represents a human clinician correction applied to an extracted value."""

    normalized_name: str
    corrected_value: Any
    corrected_unit: Optional[str] = None
    corrected_by: str = "Clinician"
    notes: Optional[str] = None


@dataclass
class StructuredLabReportPayload:
    """Structured payload combining document info, extracted lab values, and ML readiness."""

    document_info: Dict[str, Any]
    extraction_method: str  # 'DIRECT_TEXT_PDF', 'OCR_IMAGE', 'OCR_SCANNED_PDF'
    ocr_confidence: float  # OCR text extraction confidence (0.0 to 1.0)
    overall_extraction_confidence: float  # Value extraction confidence (0.0 to 1.0)
    laboratory_values: List[ExtractedLabValue] = field(default_factory=list)
    human_review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    pii_redacted_text: Optional[str] = None
    prediction_status: str = "INSUFFICIENT_INPUT_DATA"  # 'READY', 'INSUFFICIENT_INPUT_DATA', 'REVIEW_REQUIRED'
    feature_readiness: Dict[str, str] = field(default_factory=dict)
    missing_features: List[str] = field(default_factory=list)
    supplied_structured_data: Optional[Dict[str, Any]] = None
    clinical_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to JSON-serializable dictionary."""
        demographic_keys = {"age", "gender", "race", "sex", "patient_age", "patient_gender"}
        pure_lab_values = [
            v for v in self.laboratory_values
            if (v.normalized_name or "").lower() not in demographic_keys
            and (v.test_name or "").lower() not in demographic_keys
        ]
        demographic_items = [
            v for v in self.laboratory_values
            if (v.normalized_name or "").lower() in demographic_keys
            or (v.test_name or "").lower() in demographic_keys
        ]

        return {
            "document_info": self.document_info,
            "extraction_method": self.extraction_method,
            "confidence_metrics": {
                "ocr_confidence": round(self.ocr_confidence, 4),
                "extraction_confidence": round(self.overall_extraction_confidence, 4),
            },
            "prediction_status": self.prediction_status,
            "human_review_required": self.human_review_required,
            "review_reasons": self.review_reasons,
            "feature_availability": {
                "missing_required_features": self.missing_features,
                "readiness_summary": self.feature_readiness,
            },
            "patient_demographics": {
                (v.normalized_name or v.test_name.lower()): {
                    "test_name": v.test_name,
                    "normalized_name": v.normalized_name,
                    "raw_value_str": v.raw_value_str,
                    "numeric_value": v.numeric_value,
                    "validation_status": v.validation_status,
                    "validation_message": v.validation_message,
                }
                for v in demographic_items
            },
            "laboratory_values": [
                {
                    "test_name": v.test_name,
                    "normalized_name": v.normalized_name,
                    "raw_value_str": v.raw_value_str,
                    "numeric_value": v.numeric_value,
                    "unit": v.unit,
                    "reference_range": v.reference_range,
                    "status_flag": v.status_flag,
                    "extraction_confidence": round(v.extraction_confidence, 4),
                    "validation_status": v.validation_status,
                    "validation_message": v.validation_message,
                    "source_context": v.source_context or v.source_line,
                    "reference_range_source": v.reference_range_source,
                    "interpretation_status": v.interpretation_status,
                    "specific_status": (v.explanation.get("specific_status") if v.explanation else None),
                    "explanation": v.explanation,
                    "is_supported": v.is_supported,
                }
                for v in pure_lab_values
            ],
            "clinical_summary": self.clinical_summary,
            "disclaimer": (
                "Educational & Safety Notice: Laboratory findings reflect biochemical measurements at the time of testing. "
                "They do not constitute a standalone medical diagnosis. Consult a licensed physician for clinical synthesis."
            ),
        }
