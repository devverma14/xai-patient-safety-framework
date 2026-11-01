"""
Master Phase 7 OCR & Lab Report Input Preparation Pipeline.
Integrates document validation, text extraction, value parsing, PII redaction,
physiological validation, human review flagging, patient-friendly explanations,
and ML preprocessor feature formatting.
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from .quality import validate_document_quality
from .pdf import extract_text_from_pdf
from .image import extract_text_from_image
from .extraction import extract_lab_values_from_text
from .validation import (
    resolve_duplicate_extractions,
    validate_extracted_value,
    validate_and_recalculate_derived_indices,
)
from .schemas import StructuredLabReportPayload, HumanReviewCorrection, ExtractedLabValue
from .human_in_the_loop import apply_human_corrections
from .clinical_summary import generate_clinical_summary


# Full list of 49 required feature columns expected by Phase 3 / Phase 5 LabPreprocessor
REQUIRED_MODEL_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "diag_1",
    "diag_2", "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "readmitted", "encounter_id", "patient_nbr"
]


def check_feature_readiness(
    payload: StructuredLabReportPayload,
    explicit_structured_input: Optional[Dict[str, Any]] = None,
) -> StructuredLabReportPayload:
    """
    Perform explicit feature availability and readiness check for Phase 5 XGBoost prediction.
    Categorizes features into AVAILABLE_FROM_REPORT, EXPLICITLY_SUPPLIED, MISSING_FROM_REPORT, or INVALID.
    Blocks prediction if required clinical features are missing.
    """
    extracted_map = {}
    invalid_map = {}
    for v in payload.laboratory_values:
        if v.normalized_name:
            if v.validation_status == "VALID":
                extracted_map[v.normalized_name] = v
            else:
                invalid_map[v.normalized_name] = v

    explicit_input = explicit_structured_input or payload.supplied_structured_data or {}

    readiness_summary: Dict[str, str] = {}
    missing_list: List[str] = []

    for feat in REQUIRED_MODEL_FEATURES:
        if feat in extracted_map:
            readiness_summary[feat] = "AVAILABLE_FROM_REPORT"
        elif feat in explicit_input:
            readiness_summary[feat] = "EXPLICITLY_SUPPLIED"
        elif feat in invalid_map:
            readiness_summary[feat] = "INVALID_EXTRACTED_VALUE"
            missing_list.append(feat)
        else:
            readiness_summary[feat] = "MISSING_FROM_REPORT"
            missing_list.append(feat)

    payload.feature_readiness = readiness_summary
    payload.missing_features = missing_list
    payload.supplied_structured_data = explicit_input if explicit_input else None

    if len(missing_list) > 0:
        payload.prediction_status = "INSUFFICIENT_INPUT_DATA"
        payload.human_review_required = True
        reason_msg = (
            f"Prediction Blocked: Unextracted/missing clinical features ({len(missing_list)} features missing). "
            "Patient safety policy prohibits population mode/median substitution."
        )
        if reason_msg not in payload.review_reasons:
            payload.review_reasons.append(reason_msg)
    else:
        payload.prediction_status = "READY"

    return payload


def process_lab_report_file(
    file_path: str,
    explicit_structured_input: Optional[Dict[str, Any]] = None,
) -> StructuredLabReportPayload:
    """
    Process an uploaded PDF or image laboratory report through the Phase 7 pipeline.
    Extracts laboratory findings, executes reference-range interpretations and patient-friendly explanations,
    and independently evaluates 30-day hospital readmission feature readiness.

    Args:
        file_path: Path to lab report PDF or image file.
        explicit_structured_input: Optional dictionary of additional clinical/EHR features.

    Returns:
        StructuredLabReportPayload instance.
    """
    # 1. Document Quality Validation
    quality_check = validate_document_quality(file_path)
    if not quality_check.passed:
        payload = StructuredLabReportPayload(
            document_info={
                "file_name": quality_check.file_name,
                "file_type": quality_check.file_type,
                "file_size_mb": quality_check.file_size_mb,
            },
            extraction_method="FAILED_QUALITY_CHECK",
            ocr_confidence=0.0,
            overall_extraction_confidence=0.0,
            human_review_required=True,
            review_reasons=[f"Document Quality Check Failed: {quality_check.error_message}"],
            prediction_status="INSUFFICIENT_INPUT_DATA",
        )
        return check_feature_readiness(payload, explicit_structured_input)

    # 2. Text Extraction (PDF vs Image)
    if quality_check.file_type == "pdf":
        raw_text, method, ocr_conf = extract_text_from_pdf(file_path)
    else:
        raw_text, method, ocr_conf = extract_text_from_image(file_path)

    # 3. OCR Quality Check
    if not raw_text or len(raw_text.strip()) < 15:
        payload = StructuredLabReportPayload(
            document_info={
                "file_name": quality_check.file_name,
                "file_type": quality_check.file_type,
                "file_size_mb": quality_check.file_size_mb,
            },
            extraction_method=method,
            ocr_confidence=ocr_conf,
            overall_extraction_confidence=0.0,
            human_review_required=True,
            review_reasons=["OCR Text Extraction yielded empty or unreadable text."],
            pii_redacted_text="",
            prediction_status="INSUFFICIENT_INPUT_DATA",
        )
        return check_feature_readiness(payload, explicit_structured_input)

    # 4. Lab Value Extraction & PII Redaction
    extracted_values, redacted_text = extract_lab_values_from_text(raw_text)

    # 5. Duplicate & Conflict Resolution
    deduped_values, conflict_reasons = resolve_duplicate_extractions(extracted_values)

    # 6. Derived Index Validation & Recalculation
    validated_values, derived_warnings = validate_and_recalculate_derived_indices(deduped_values)

    # Check if required information was extracted
    review_reasons = list(conflict_reasons)
    for dw in derived_warnings:
        if dw not in review_reasons:
            review_reasons.append(dw)

    if len(validated_values) == 0:
        review_reasons.append("No recognized laboratory test parameters were extracted from the document.")

    # Check validation status of extracted items
    unvalidated_count = 0
    for v in validated_values:
        if v.validation_status != "VALID":
            unvalidated_count += 1
            msg = f"Lab test '{v.test_name}': {v.validation_message}"
            if msg not in review_reasons:
                review_reasons.append(msg)

    overall_conf = (
        float(sum(v.extraction_confidence for v in validated_values) / len(validated_values))
        if validated_values
        else 0.0
    )
    human_review = (len(validated_values) == 0) or (unvalidated_count > 0) or (len(conflict_reasons) > 0)

    payload = StructuredLabReportPayload(
        document_info={
            "file_name": quality_check.file_name,
            "file_type": quality_check.file_type,
            "file_size_mb": quality_check.file_size_mb,
        },
        extraction_method=method,
        ocr_confidence=ocr_conf,
        overall_extraction_confidence=overall_conf,
        laboratory_values=validated_values,
        human_review_required=human_review,
        review_reasons=review_reasons,
        pii_redacted_text=redacted_text,
        clinical_summary=generate_clinical_summary(validated_values),
    )

    # Perform Feature Availability & Readiness Check
    return check_feature_readiness(payload, explicit_structured_input)


def convert_payload_to_ml_dataframe(
    payload: StructuredLabReportPayload,
    explicit_structured_input: Optional[Dict[str, Any]] = None,
    reference_sample_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert validated StructuredLabReportPayload into a single-row pandas DataFrame
    matching the column structure expected by the Phase 3 / Phase 5 LabPreprocessor.

    Patient Safety Rule: Population mode/median default substitution is strictly prohibited.
    If required features are missing from report and not explicitly supplied, raises ValueError.

    Args:
        payload: Validated payload.
        explicit_structured_input: Optional dictionary of additional clinical/EHR features.
        reference_sample_df: Optional reference DataFrame template.

    Returns:
        Single-row pandas DataFrame ready for LabPreprocessor.transform().
    """
    # Re-verify readiness with any new explicit_structured_input
    if explicit_structured_input:
        check_feature_readiness(payload, explicit_structured_input)

    if payload.prediction_status != "READY":
        raise ValueError(
            f"Cannot generate ML prediction for payload with prediction_status='{payload.prediction_status}'. "
            f"Missing required clinical features: {payload.missing_features}. "
            "Patient safety framework prohibits population mode/median default substitution."
        )

    row_dict: Dict[str, Any] = {}

    # 1. Add explicitly supplied structured clinical data
    explicit_data = explicit_structured_input or payload.supplied_structured_data or {}
    for k, v in explicit_data.items():
        row_dict[k] = v

    # 2. Override with validated extracted lab values from report
    for val in payload.laboratory_values:
        if val.normalized_name and val.validation_status == "VALID":
            if val.numeric_value is not None:
                row_dict[val.normalized_name] = val.numeric_value
            else:
                row_dict[val.normalized_name] = val.raw_value_str

    # Construct DataFrame in required column order
    if reference_sample_df is not None:
        cols = list(reference_sample_df.columns)
        final_dict = {c: row_dict[c] for c in cols}
        return pd.DataFrame([final_dict])

    final_dict = {c: row_dict[c] for c in REQUIRED_MODEL_FEATURES}
    return pd.DataFrame([final_dict])
