"""
Human-in-the-Loop review and manual correction module.
Allows medical practitioners to review, edit, or override extracted laboratory parameters
when validation flags uncertain or missing values before passing to ML inference.
"""

from typing import List, Dict, Any, Optional
from .schemas import ExtractedLabValue, HumanReviewCorrection, StructuredLabReportPayload


def apply_human_corrections(
    payload: StructuredLabReportPayload,
    corrections: List[HumanReviewCorrection],
) -> StructuredLabReportPayload:
    """
    Apply human clinician corrections to structured lab report payload.

    Args:
        payload: Original StructuredLabReportPayload instance.
        corrections: List of HumanReviewCorrection objects.

    Returns:
        Updated StructuredLabReportPayload instance.
    """
    updated_values: List[ExtractedLabValue] = []
    correction_map = {c.normalized_name: c for c in corrections}

    for lab_val in payload.laboratory_values:
        norm_key = lab_val.normalized_name
        if norm_key in correction_map:
            c = correction_map[norm_key]
            # Create updated lab value preserving original raw extraction info
            updated_val = ExtractedLabValue(
                test_name=lab_val.test_name,
                normalized_name=norm_key,
                raw_value_str=str(c.corrected_value),
                numeric_value=float(c.corrected_value) if isinstance(c.corrected_value, (int, float)) else (float(c.corrected_value) if str(c.corrected_value).replace(".", "", 1).isdigit() else None),
                unit=c.corrected_unit or lab_val.unit,
                reference_range=lab_val.reference_range,
                status_flag=lab_val.status_flag,
                extraction_confidence=1.0,  # 100% confidence for human-verified input
                validation_status="VALID",
                validation_message=f"Corrected by {c.corrected_by}: {c.notes or 'Manual override'}",
                source_line=lab_val.source_line,
            )
            updated_values.append(updated_val)
        else:
            updated_values.append(lab_val)

    # Check for new manually added parameters in corrections that were missing from OCR
    existing_keys = set(v.normalized_name for v in updated_values)
    for c in corrections:
        if c.normalized_name not in existing_keys:
            new_val = ExtractedLabValue(
                test_name=c.normalized_name,
                normalized_name=c.normalized_name,
                raw_value_str=str(c.corrected_value),
                numeric_value=float(c.corrected_value) if isinstance(c.corrected_value, (int, float)) else (float(c.corrected_value) if str(c.corrected_value).replace(".", "", 1).isdigit() else None),
                unit=c.corrected_unit,
                reference_range=None,
                status_flag="MANUAL_ENTRY",
                extraction_confidence=1.0,
                validation_status="VALID",
                validation_message=f"Manually added by {c.corrected_by}",
            )
            updated_values.append(new_val)

    # Check remaining unvalidated/flagged items
    remaining_flags = [v for v in updated_values if v.validation_status != "VALID"]
    payload.laboratory_values = updated_values
    payload.human_review_required = len(remaining_flags) > 0
    payload.review_reasons = [f"Item '{v.normalized_name}': {v.validation_message}" for v in remaining_flags]

    return payload
