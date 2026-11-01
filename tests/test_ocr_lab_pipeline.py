"""
Unit and Integration Test Suite for Phase 7 OCR & Lab Report Input Preparation Subsystem.
Verifies quality validation, PDF/image reading, lab value extraction, test-name normalization,
physiological validation, conflict handling, PII redaction, human-in-the-loop overrides,
and schema conversion.
"""

import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from ocr.quality import validate_document_quality
from ocr.normalization import normalize_test_name, normalize_unit
from ocr.extraction import redact_pii_from_text, extract_lab_values_from_text
from ocr.validation import (
    validate_extracted_value,
    resolve_duplicate_extractions,
    validate_and_recalculate_derived_indices,
    parse_reference_range,
    determine_interpretation_status,
)
from ocr.human_in_the_loop import apply_human_corrections
from ocr.schemas import ExtractedLabValue, HumanReviewCorrection, StructuredLabReportPayload
from ocr.pipeline import process_lab_report_file, convert_payload_to_ml_dataframe


class TestPhase7OCRPipeline(unittest.TestCase):
    """Test suite verifying all 17 Phase 7 requirements."""

    @classmethod
    def setUpClass(cls):
        cls.sample_dir = os.path.join("tests", "sample_reports")

    def test_01_valid_image_input(self):
        """Test valid image report processing."""
        fpath = os.path.join(self.sample_dir, "sample_image_report.jpg")
        payload = process_lab_report_file(fpath)
        self.assertIsNotNone(payload.document_info)
        self.assertEqual(payload.document_info["file_type"], "image")
        self.assertTrue(len(payload.laboratory_values) > 0)

    def test_02_invalid_file_type(self):
        """Test rejection of unsupported file types."""
        res = validate_document_quality("invalid_file.exe")
        self.assertFalse(res.passed)
        self.assertEqual(res.file_type, "unsupported")

    def test_03_corrupted_empty_document(self):
        """Test empty/corrupted document handling."""
        fpath = os.path.join(self.sample_dir, "sample_empty.pdf")
        payload = process_lab_report_file(fpath)
        self.assertTrue(payload.human_review_required)

    def test_04_test_name_normalization(self):
        """Test normalization of raw test strings to dataset feature keys."""
        self.assertEqual(normalize_test_name("HbA1c"), "A1Cresult")
        self.assertEqual(normalize_test_name("Glycated Hemoglobin"), "A1Cresult")
        self.assertEqual(normalize_test_name("Fasting Blood Glucose"), "max_glu_serum")
        self.assertEqual(normalize_test_name("Length of Stay"), "time_in_hospital")

    def test_05_unit_extraction_and_standardization(self):
        """Test unit normalization."""
        u1, valid1 = normalize_unit("%", "A1Cresult")
        self.assertEqual(u1, "%")
        self.assertTrue(valid1)

        u2, valid2 = normalize_unit("mg/dL", "max_glu_serum")
        self.assertEqual(u2, "mg/dL")
        self.assertTrue(valid2)

    def test_06_out_of_bounds_detection(self):
        """Test physiological range validation for out-of-bounds values."""
        lab_val = ExtractedLabValue(
            test_name="Glucose",
            normalized_name="max_glu_serum",
            raw_value_str="2500",
            numeric_value=2500.0,
            unit="mg/dL",
            reference_range="70-100",
            status_flag="HIGH",
            extraction_confidence=0.95,
            validation_status="PENDING",
        )
        validated = validate_extracted_value(lab_val)
        self.assertEqual(validated.validation_status, "REVIEW_REQUIRED")
        self.assertIn("exceeds physiological maximum", validated.validation_message)

    def test_07_conflicting_duplicate_resolution(self):
        """Test handling of conflicting duplicate extractions."""
        val1 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="6.5",
            numeric_value=6.5, unit="%", reference_range=None, status_flag=None,
            extraction_confidence=0.90, validation_status="VALID",
        )
        val2 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="12.4",
            numeric_value=12.4, unit="%", reference_range=None, status_flag=None,
            extraction_confidence=0.90, validation_status="VALID",
        )
        resolved, reasons = resolve_duplicate_extractions([val1, val2])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].validation_status, "REVIEW_REQUIRED")
        self.assertTrue(len(reasons) > 0)

    def test_08_pii_redaction(self):
        """Test PII redaction (names, DOB, patient ID, phone)."""
        raw_text = "Patient Name: John Doe DOB: 01/01/1980 Patient ID: MRN12345 Phone: 555-0199 HbA1c: 7.5%"
        redacted = redact_pii_from_text(raw_text)
        self.assertNotIn("John Doe", redacted)
        self.assertNotIn("01/01/1980", redacted)
        self.assertNotIn("MRN12345", redacted)
        self.assertNotIn("555-0199", redacted)
        self.assertIn("[REDACTED_PII]", redacted)

    def test_09_human_in_the_loop_corrections(self):
        """Test applying clinician manual overrides."""
        payload = StructuredLabReportPayload(
            document_info={"file_name": "test.png"},
            extraction_method="OCR_IMAGE",
            ocr_confidence=0.8,
            overall_extraction_confidence=0.8,
            laboratory_values=[
                ExtractedLabValue(
                    test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="12.4",
                    numeric_value=12.4, unit="%", reference_range=None, status_flag=None,
                    extraction_confidence=0.5, validation_status="REVIEW_REQUIRED",
                    validation_message="Uncertain extraction",
                )
            ],
            human_review_required=True,
            review_reasons=["Uncertain extraction"],
        )
        corrections = [
            HumanReviewCorrection(
                normalized_name="A1Cresult", corrected_value=7.2, corrected_unit="%", corrected_by="Dr. Jones"
            )
        ]
        updated = apply_human_corrections(payload, corrections)
        self.assertFalse(updated.human_review_required)
        self.assertEqual(updated.laboratory_values[0].numeric_value, 7.2)
        self.assertEqual(updated.laboratory_values[0].validation_status, "VALID")

    def test_10_missing_features_blocks_prediction(self):
        """Test that incomplete report payload sets INSUFFICIENT_INPUT_DATA and blocks prediction."""
        fpath = os.path.join(self.sample_dir, "sample_image_report.jpg")
        payload = process_lab_report_file(fpath)
        self.assertEqual(payload.prediction_status, "INSUFFICIENT_INPUT_DATA")
        self.assertTrue(payload.human_review_required)
        self.assertTrue(len(payload.missing_features) > 0)
        # Verify ValueError is raised when attempting to convert incomplete payload without defaults
        with self.assertRaises(ValueError):
            convert_payload_to_ml_dataframe(payload)

    def test_11_extracted_values_preserved_despite_missing_ehr(self):
        """Verify extracted lab values remain intact in payload even when prediction is blocked."""
        fpath = os.path.join(self.sample_dir, "sample_image_report.jpg")
        payload = process_lab_report_file(fpath)
        self.assertTrue(len(payload.laboratory_values) > 0)
        self.assertEqual(payload.prediction_status, "INSUFFICIENT_INPUT_DATA")
        # Verify specific extracted lab value is preserved
        a1c_val = next((v for v in payload.laboratory_values if v.normalized_name == "A1Cresult"), None)
        self.assertIsNotNone(a1c_val)
        self.assertEqual(a1c_val.numeric_value, 7.8)

    def test_12_no_silent_population_defaults_substitution(self):
        """Verify pipeline prohibits silent population median/mode default substitution."""
        payload = StructuredLabReportPayload(
            document_info={"file_name": "test.png"},
            extraction_method="OCR_IMAGE",
            ocr_confidence=0.9,
            overall_extraction_confidence=0.9,
            laboratory_values=[
                ExtractedLabValue(
                    test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="8.0",
                    numeric_value=8.0, unit="%", reference_range=None, status_flag=None,
                    extraction_confidence=0.95, validation_status="VALID",
                )
            ],
            prediction_status="INSUFFICIENT_INPUT_DATA",
            missing_features=["age", "gender", "number_inpatient"],
        )
        with self.assertRaises(ValueError) as ctx:
            convert_payload_to_ml_dataframe(payload)
        self.assertIn("prohibits population mode/median default substitution", str(ctx.exception))

    def test_13_complete_explicit_structured_input_allows_ml_dataframe(self):
        """Verify complete explicitly supplied structured input enables prediction readiness."""
        fpath = os.path.join(self.sample_dir, "sample_image_report.jpg")
        # Complete dictionary supplying all required 49 features
        complete_explicit_input = {
            "race": "Caucasian", "gender": "Female", "age": "[60-70)",
            "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
            "time_in_hospital": 5, "payer_code": "?", "medical_specialty": "?",
            "num_lab_procedures": 45, "num_procedures": 0, "num_medications": 12,
            "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2,
            "diag_1": "250", "diag_2": "?", "diag_3": "?", "number_diagnoses": 6,
            "max_glu_serum": "None", "A1Cresult": "None", "metformin": "No",
            "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
            "glimepiride": "No", "acetohexamide": "No", "glipizide": "No",
            "glyburide": "No", "tolbutamide": "No", "pioglitazone": "No",
            "rosiglitazone": "No", "acarbose": "No", "miglitol": "No",
            "troglitazone": "No", "tolazamide": "No", "examide": "No",
            "citoglipton": "No", "insulin": "Up", "glyburide-metformin": "No",
            "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No",
            "change": "No", "diabetesMed": "Yes", "readmitted": "NO",
            "encounter_id": 999999, "patient_nbr": 888888,
        }
        payload = process_lab_report_file(fpath, explicit_structured_input=complete_explicit_input)
        self.assertEqual(payload.prediction_status, "READY")
        df = convert_payload_to_ml_dataframe(payload, explicit_structured_input=complete_explicit_input)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["A1Cresult"], 7.8)

    def test_14_real_world_panels_extraction_and_mapping(self):
        """Test extraction and mapping of CBC, CMP, Lipid, and Thyroid panels."""
        sample_report_text = """
        METABOLIC & HEMATOLOGY PROFILE
        Hemoglobin 14.2 g/dL (12.0 - 17.5) NORMAL
        Platelets 280 10^3/uL (150 - 450) NORMAL
        WBC 7.5 10^3/uL (4.0 - 11.0) NORMAL
        Sodium 140 mEq/L (135 - 145) NORMAL
        Potassium 4.6 mEq/L (3.5 - 5.0) NORMAL
        Creatinine 1.1 mg/dL (0.6 - 1.3) NORMAL
        BUN 15 mg/dL (7 - 20) NORMAL
        Total Cholesterol 185 mg/dL (< 200) NORMAL
        Triglycerides 140 mg/dL (< 150) NORMAL
        TSH 2.1 uIU/mL (0.4 - 4.0) NORMAL
        """
        vals, _ = extract_lab_values_from_text(sample_report_text)
        mapped_keys = {v.normalized_name: v for v in vals}

        self.assertIn("hemoglobin", mapped_keys)
        self.assertEqual(mapped_keys["hemoglobin"].numeric_value, 14.2)
        self.assertEqual(mapped_keys["hemoglobin"].unit, "g/dL")

        self.assertIn("sodium", mapped_keys)
        self.assertEqual(mapped_keys["sodium"].numeric_value, 140.0)

        self.assertIn("creatinine", mapped_keys)
        self.assertEqual(mapped_keys["creatinine"].numeric_value, 1.1)

        self.assertIn("cholesterol_total", mapped_keys)
        self.assertEqual(mapped_keys["cholesterol_total"].numeric_value, 185.0)

        self.assertIn("tsh", mapped_keys)
        self.assertEqual(mapped_keys["tsh"].numeric_value, 2.1)

    def test_15_reference_range_interpretation_high_low_within(self):
        """Test high, low, within-range, and unable-to-determine classifications."""
        test_text = """
        Glucose 245 mg/dL (70 - 99) HIGH
        Hemoglobin 9.5 g/dL (12.0 - 17.5) LOW
        Sodium 138 mEq/L (135 - 145) NORMAL
        """
        vals, _ = extract_lab_values_from_text(test_text)
        status_map = {v.normalized_name: v.interpretation_status for v in vals}

        self.assertEqual(status_map.get("max_glu_serum"), "HIGH")
        self.assertEqual(status_map.get("hemoglobin"), "LOW")
        self.assertEqual(status_map.get("sodium"), "WITHIN_RANGE")

    def test_16_patient_friendly_explanations_populated(self):
        """Test that patient-friendly non-diagnostic explanations and clinical disclaimers are populated."""
        test_text = "Glucose 245 mg/dL (70 - 99) HIGH"
        vals, _ = extract_lab_values_from_text(test_text)
        self.assertEqual(len(vals), 1)
        item = vals[0]

        self.assertIsNotNone(item.explanation)
        self.assertIn("description", item.explanation)
        self.assertIn("possible_causes", item.explanation)
        self.assertTrue(len(item.explanation["possible_causes"]) > 0)
        self.assertIn("clinical_disclaimer", item.explanation)
        # Ensure disease is not claimed as confirmed
        self.assertNotIn("confirmed diabetes", item.explanation["description"].lower())
        self.assertIn("educational", item.explanation["clinical_disclaimer"].lower())

    def test_17_missing_unit_flags_review_without_guessing(self):
        """Test that missing units flag REVIEW_REQUIRED without silent guessing."""
        fpath = os.path.join(self.sample_dir, "sample_missing_unit.png")
        payload = process_lab_report_file(fpath)

        missing_unit_items = [v for v in payload.laboratory_values if v.unit is None]
        self.assertTrue(len(missing_unit_items) > 0)
        for item in missing_unit_items:
            self.assertEqual(item.validation_status, "REVIEW_REQUIRED")
            self.assertEqual(item.interpretation_status, "UNABLE_TO_DETERMINE")
            self.assertIn("Missing unit", item.validation_message)

    def test_18_ambiguous_and_unreadable_values(self):
        """Test that lines with test names but no parseable value are flagged for review."""
        test_text = "Hemoglobin : Pending Laboratory Recalibration"
        vals, _ = extract_lab_values_from_text(test_text)
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0].validation_status, "REVIEW_REQUIRED")
        self.assertIsNone(vals[0].numeric_value)

    def test_19_multi_page_pdf_processing(self):
        """Test multi-page PDF reading preserving page context."""
        fpath = os.path.join(self.sample_dir, "sample_multipage.pdf")
        payload = process_lab_report_file(fpath)
        self.assertEqual(payload.extraction_method, "DIRECT_TEXT_PDF")
        self.assertIn("Page 1", payload.pii_redacted_text)
        self.assertIn("Page 2", payload.pii_redacted_text)
        self.assertTrue(len(payload.laboratory_values) >= 2)

    def test_20_source_context_traceability(self):
        """Test that source_context and source_line are populated and serialized in to_dict."""
        test_text = "Creatinine 1.2 mg/dL (0.6 - 1.3) Normal"
        vals, _ = extract_lab_values_from_text(test_text)
        self.assertEqual(len(vals), 1)
        v = vals[0]
        self.assertIsNotNone(v.source_line)
        self.assertIn("Creatinine", v.source_line)
        self.assertIsNotNone(v.source_context)

        payload = StructuredLabReportPayload(
            document_info={"file_name": "test.pdf"},
            extraction_method="DIRECT_TEXT_PDF",
            ocr_confidence=0.95,
            overall_extraction_confidence=0.95,
            laboratory_values=[v],
        )
        d = payload.to_dict()
        lab_dict = d["laboratory_values"][0]
        self.assertIn("source_context", lab_dict)
        self.assertIn("explanation", lab_dict)
        self.assertIn("interpretation_status", lab_dict)

    def test_21_readmission_prediction_safety_isolation(self):
        """Test that standalone lab panels never bypass readmission model feature readiness gates."""
        test_text = """
        Hemoglobin 14.2 g/dL (12.0 - 17.5) NORMAL
        Platelets 280 10^3/uL (150 - 450) NORMAL
        Sodium 140 mEq/L (135 - 145) NORMAL
        Glucose 105 mg/dL (70 - 99) HIGH
        """
        vals, _ = extract_lab_values_from_text(test_text)
        payload = StructuredLabReportPayload(
            document_info={"file_name": "lab_panel.pdf"},
            extraction_method="DIRECT_TEXT_PDF",
            ocr_confidence=0.95,
            overall_extraction_confidence=0.95,
            laboratory_values=vals,
        )
        from ocr.pipeline import check_feature_readiness
        payload = check_feature_readiness(payload)

        self.assertEqual(payload.prediction_status, "INSUFFICIENT_INPUT_DATA")
        self.assertTrue(payload.human_review_required)
        self.assertTrue(len(payload.missing_features) > 30)
        with self.assertRaises(ValueError):
            convert_payload_to_ml_dataframe(payload)

    def test_22_multiple_tests_on_single_reconstructed_line(self):
        """Test extraction of multiple distinct tests on a single horizontal reconstructed OCR line."""
        single_line = "Glucose 105 mg/dL (70 - 99) HIGH    WBC 6.5 10^3/uL (4.0 - 11.0) NORMAL"
        vals, _ = extract_lab_values_from_text(single_line)
        self.assertEqual(len(vals), 2, "Failed to extract both tests from single reconstructed line")

        # Test Glucose extraction
        v_glu = next((v for v in vals if v.normalized_name == "max_glu_serum"), None)
        self.assertIsNotNone(v_glu, "Glucose not found in extractions")
        self.assertEqual(v_glu.numeric_value, 105.0)
        self.assertEqual(v_glu.unit, "mg/dL")
        self.assertEqual(v_glu.reference_range, "70 - 99")
        self.assertEqual(v_glu.interpretation_status, "HIGH")
        self.assertIn("Glucose", v_glu.source_line)

        # Test WBC extraction
        v_wbc = next((v for v in vals if v.normalized_name == "wbc"), None)
        self.assertIsNotNone(v_wbc, "WBC not found in extractions")
        self.assertEqual(v_wbc.numeric_value, 6.5)
        self.assertEqual(v_wbc.unit, "10^3/uL")
        self.assertEqual(v_wbc.reference_range, "4.0 - 11.0")
        self.assertEqual(v_wbc.interpretation_status, "WITHIN_RANGE")
        self.assertIn("WBC", v_wbc.source_line)

        # Verify side-by-side synthetic image fixture
        sbs_path = os.path.join(self.sample_dir, "synthetic_side_by_side_panel.png")
        if os.path.exists(sbs_path):
            sbs_payload = process_lab_report_file(sbs_path)
            extracted_names = {v.normalized_name for v in sbs_payload.laboratory_values}
            expected_names = {"max_glu_serum", "bun", "albumin", "wbc", "platelets", "creatinine"}
            self.assertTrue(expected_names.issubset(extracted_names), f"Missing tests from side-by-side panel: {expected_names - extracted_names}")

    def test_23_ambiguous_and_duplicate_alias_handling(self):
        """Test negative cases: ambiguous multi-values flag REVIEW_REQUIRED, and duplicate aliases are filtered."""
        # 1. Ambiguous multi-value on single test line
        ambig_line = "Glucose 105 185 mg/dL (70 - 99) NORMAL"
        vals, _ = extract_lab_values_from_text(ambig_line)
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0].validation_status, "REVIEW_REQUIRED")
        self.assertIn("Ambiguous multiple numeric values", vals[0].validation_message)

        # 2. Adjacent test names with single value
        merged_line = "Glucose WBC 105 mg/dL"
        vals_merged, _ = extract_lab_values_from_text(merged_line)
        self.assertTrue(any(v.validation_status == "REVIEW_REQUIRED" for v in vals_merged))

        # 3. Duplicate/overlapping alias avoidance: "Blood Glucose" should not extract "Glucose" as separate test
        dup_alias_line = "Blood Glucose 110 mg/dL (70 - 99) NORMAL"
        vals_dup, _ = extract_lab_values_from_text(dup_alias_line)
        self.assertEqual(len(vals_dup), 1, "Overlapping alias created duplicate extraction")
        self.assertEqual(vals_dup[0].normalized_name, "max_glu_serum")

        # 4. Overlapping alias for HbA1c
        hba1c_line = "Hemoglobin A1c (HbA1c) 7.5 % (4.0 - 5.6) HIGH"
        vals_a1c, _ = extract_lab_values_from_text(hba1c_line)
        self.assertEqual(len(vals_a1c), 1, "Duplicate HbA1c extraction from synonym overlap")
        self.assertEqual(vals_a1c[0].normalized_name, "A1Cresult")

    def test_24_strict_inequality_boundaries(self):
        """Test strict vs inclusive inequality boundaries and closed interval interpretation."""
        # 1. Test parse_reference_range operator preservation and backward-compatible unpacking
        p_strict_upper = parse_reference_range("< 200")
        low, high = p_strict_upper  # Unpacking test
        self.assertIsNone(low)
        self.assertEqual(high, 200.0)
        self.assertEqual(p_strict_upper.op, "<")

        p_incl_lower = parse_reference_range(">= 60")
        low, high = p_incl_lower
        self.assertEqual(low, 60.0)
        self.assertIsNone(high)
        self.assertEqual(p_incl_lower.op, ">=")

        def get_status(val, r):
            status, _, _ = determine_interpretation_status(val, raw_reference_range=r)
            return status

        # 2. Upper-bound strict inequality: < 200
        # Below boundary -> WITHIN_RANGE
        self.assertEqual(get_status(199.9, "< 200"), "WITHIN_RANGE")
        # At boundary -> HIGH (strict < 200 means 200 is not within range)
        self.assertEqual(get_status(200.0, "< 200"), "HIGH")
        # Above boundary -> HIGH
        self.assertEqual(get_status(200.1, "< 200"), "HIGH")

        # 3. Upper-bound inclusive inequality: <= 200
        self.assertEqual(get_status(200.0, "<= 200"), "WITHIN_RANGE")
        self.assertEqual(get_status(200.1, "<= 200"), "HIGH")

        # 4. Lower-bound strict inequality: > 60
        # Below boundary -> LOW
        self.assertEqual(get_status(59.9, "> 60"), "LOW")
        # At boundary -> LOW (strict > 60 means 60 is not within range)
        self.assertEqual(get_status(60.0, "> 60"), "LOW")
        # Above boundary -> WITHIN_RANGE
        self.assertEqual(get_status(60.1, "> 60"), "WITHIN_RANGE")

        # 5. Lower-bound inclusive inequality: >= 60
        self.assertEqual(get_status(59.9, ">= 60"), "LOW")
        self.assertEqual(get_status(60.0, ">= 60"), "WITHIN_RANGE")
        self.assertEqual(get_status(60.1, ">= 60"), "WITHIN_RANGE")

        # 6. Ordinary closed interval: 4.0 - 5.6
        self.assertEqual(get_status(3.9, "4.0 - 5.6"), "LOW")
        self.assertEqual(get_status(4.0, "4.0 - 5.6"), "WITHIN_RANGE")
        self.assertEqual(get_status(5.0, "4.0 - 5.6"), "WITHIN_RANGE")
        self.assertEqual(get_status(5.6, "4.0 - 5.6"), "WITHIN_RANGE")
        self.assertEqual(get_status(5.7, "4.0 - 5.6"), "HIGH")

    def test_25_healthians_multiline_table_and_narrative_rejection(self):
        """
        Regression test for Abhishek Thakur report:
        Verifies multiline table extraction for CRP, HbA1c, Glucose, Cholesterol, Creatinine,
        TSH, Haemoglobin, and Neutrophils, while rejecting narrative guideline thresholds (<6, >19, 126, 200).
        """
        report_snippet = """
        Patient Name : Abhishek Thakur
        Age/Gender : 23Y 7M 25D /Male
        Order Id : 20469105694
        DEPARTMENT OF BIOCHEMISTRY HBA1C
        Test Name Value Unit Bio. Ref Interval

        HbA1c - Glycosylated Hemoglobin
        Hba1c (Glycosylated Hemoglobin)
        Method: HPLC
        Machine: TOSOH G11
        7.80 % 4.2 - 5.7

        Average Estimated Glucose
        Method: Calculated
        177.16 mg/dl

        INTERPRETATION:
        AS PER AMERICAN DIABETES ASSOCIATION (ADA):
        REFERENCE GROUP GLYCOSYLATED HEMOGLOBIN (HBA1c) IN %
        Non diabetic <5.7
        AT Risk (Prediabetes) 5.7 – 6.4
        Diagnosing Diabetes >= 6.5
        Therapeutic goals FOR glycemic control
        Age > 19 Years
        Goals of Therapy: < 7.0
        Actions Suggested: >8.0
        Age < 19 Years
        Goal of therapy: <7.5
        REMARKS
        1. HbA1c IS used FOR monitoring diabetic control.
        7. ANY sample WITH >15% HbA1c should be suspected of HAVING a hemoglobin variant.
        8. HbA1c target IN pregnancy IS TO attain LEVEL <6 % .
        9. HbA1c target IN paediatric age GROUP IS TO attain LEVEL < 7.5 %.

        DEPARTMENT OF BIOCHEMISTRY
        Glucose-Random
        GLUCOSE, RANDOM
        Method: Hexokinase
        Machine: BECKMAN COULTER AU 5800
        169 mg/dl 70 - 140
        Test Interpretation
        According to the American Diabetes Association (ADA), a random blood glucose level of 200 mg/dL or higher.
        Other criteria for diagnosing diabetes include:
        a. Fasting plasma glucose (FPG): A level of 126 mg/dL or higher

        DEPARTMENT OF BIOCHEMISTRY
        C-Reactive Protein (CRP) -Quantitative
        C-REACTIVE PROTEIN (CRP)
        (QUANTITATIVE)
        Method: Latex Particle Immunoturbidimetric
        Machine: BECKMAN COULTER AU 5800
        3.18 mg/L <5
        C-reactive protein (CRP) is one of the most sensitive acute-phase reactants for inflammation.

        DEPARTMENT OF BIOCHEMISTRY
        Lipid Profile
        Total Cholesterol
        Method: CHO-POD (Trac. Abel-Kendall)
        Machine: BECKMAN COULTER AU 5800
        192.1 mg/dL Desirable : <200

        DEPARTMENT OF BIOCHEMISTRY
        Kidney Function Test
        Serum Creatinine
        Method: Modified Jaffe, Kinetic
        Machine: BECKMAN COULTER AU 5800
        0.90 mg/dl 0.2 - 1.2

        DEPARTMENT OF IMMUNOLOGY
        Thyroid Profile (Total T3,T4, TSH)
        Thyroid Stimulating Hormone (TSH)-Ultrasensitive
        Method: CLIA
        Machine: Beckman Coulter UniCel DxI800
        3.638 µIU/ml 0.38-5.33

        DEPARTMENT OF HAEMATOLOGY
        Complete Blood Count
        Haemoglobin (HB)
        Method: Spectrophotometry
        Machine: Yumizen H2500
        14.8 g/dL 13.0-17.0

        Differential Leucocyte Count
        Neutrophils 34.4 % 40 - 80
        """
        extracted, _ = extract_lab_values_from_text(report_snippet)
        deduped, reasons = resolve_duplicate_extractions(extracted)
        mapped = {v.normalized_name: v for v in deduped}

        # 1. Age & Gender from demographic header
        self.assertIn("age", mapped)
        self.assertEqual(mapped["age"].numeric_value, 23.0)
        self.assertEqual(mapped["age"].validation_status, "VALID")
        self.assertIn("gender", mapped)
        self.assertEqual(mapped["gender"].raw_value_str, "Male")

        # 2. HbA1c
        self.assertIn("A1Cresult", mapped)
        self.assertEqual(mapped["A1Cresult"].numeric_value, 7.80)
        self.assertEqual(mapped["A1Cresult"].unit, "%")
        self.assertEqual(mapped["A1Cresult"].reference_range, "4.2 - 5.7")
        self.assertEqual(mapped["A1Cresult"].validation_status, "VALID")

        # 3. Glucose
        self.assertIn("max_glu_serum", mapped)
        self.assertEqual(mapped["max_glu_serum"].numeric_value, 169.0)
        self.assertEqual(mapped["max_glu_serum"].unit, "mg/dL")
        self.assertEqual(mapped["max_glu_serum"].reference_range, "70 - 140")
        self.assertEqual(mapped["max_glu_serum"].validation_status, "VALID")

        # 4. CRP
        self.assertIn("crp", mapped)
        self.assertEqual(mapped["crp"].numeric_value, 3.18)
        self.assertEqual(mapped["crp"].unit, "mg/L")
        self.assertEqual(mapped["crp"].reference_range, "<5")
        self.assertEqual(mapped["crp"].validation_status, "VALID")

        # 5. Total Cholesterol
        self.assertIn("cholesterol_total", mapped)
        self.assertEqual(mapped["cholesterol_total"].numeric_value, 192.1)
        self.assertEqual(mapped["cholesterol_total"].unit, "mg/dL")
        self.assertEqual(mapped["cholesterol_total"].validation_status, "VALID")

        # 6. Creatinine
        self.assertIn("creatinine", mapped)
        self.assertEqual(mapped["creatinine"].numeric_value, 0.90)
        self.assertEqual(mapped["creatinine"].unit, "mg/dL")
        self.assertEqual(mapped["creatinine"].reference_range, "0.2 - 1.2")
        self.assertEqual(mapped["creatinine"].validation_status, "VALID")

        # 7. TSH
        self.assertIn("tsh", mapped)
        self.assertEqual(mapped["tsh"].numeric_value, 3.638)
        self.assertEqual(mapped["tsh"].unit, "uIU/mL")
        self.assertEqual(mapped["tsh"].reference_range, "0.38-5.33")
        self.assertEqual(mapped["tsh"].validation_status, "VALID")

        # 8. Haemoglobin
        self.assertIn("hemoglobin", mapped)
        self.assertEqual(mapped["hemoglobin"].numeric_value, 14.8)
        self.assertEqual(mapped["hemoglobin"].unit, "g/dL")
        self.assertEqual(mapped["hemoglobin"].reference_range, "13.0-17.0")
        self.assertEqual(mapped["hemoglobin"].validation_status, "VALID")

        # 9. Neutrophils
        self.assertIn("neutrophils", mapped)
        self.assertEqual(mapped["neutrophils"].numeric_value, 34.4)
        self.assertEqual(mapped["neutrophils"].unit, "%")
        self.assertEqual(mapped["neutrophils"].reference_range, "40 - 80")
        self.assertEqual(mapped["neutrophils"].validation_status, "VALID")

        # 10. No spurious conflict reasons
        self.assertEqual(len(reasons), 0, f"Spurious conflict reasons found: {reasons}")

    def test_26_empty_candidate_deduplication_and_genuine_conflict_preservation(self):
        """
        Verify that empty candidates do not conflict with valid measurements,
        while genuinely conflicting distinct measured values strictly trigger REVIEW_REQUIRED.
        """
        # Case 1: Empty candidate alongside valid measured candidate -> Resolves safely to measured value
        val_empty = ExtractedLabValue(
            test_name="Glucose", normalized_name="max_glu_serum", raw_value_str="",
            numeric_value=None, unit=None, reference_range=None, status_flag=None,
            extraction_confidence=0.50, validation_status="REVIEW_REQUIRED",
        )
        val_measured = ExtractedLabValue(
            test_name="GLUCOSE, RANDOM", normalized_name="max_glu_serum", raw_value_str="169",
            numeric_value=169.0, unit="mg/dL", reference_range="70 - 140", status_flag=None,
            extraction_confidence=0.95, validation_status="VALID",
        )
        resolved_safe, reasons_safe = resolve_duplicate_extractions([val_empty, val_measured])
        self.assertEqual(len(resolved_safe), 1)
        self.assertEqual(resolved_safe[0].numeric_value, 169.0)
        self.assertEqual(resolved_safe[0].validation_status, "VALID")
        self.assertEqual(len(reasons_safe), 0)

        # Case 2: Genuine conflicting measured values -> Preserves REVIEW_REQUIRED and UNABLE_TO_DETERMINE
        val_conflict1 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="6.5",
            numeric_value=6.5, unit="%", reference_range="4.0 - 5.6", status_flag=None,
            extraction_confidence=0.95, validation_status="VALID",
        )
        val_conflict2 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="12.4",
            numeric_value=12.4, unit="%", reference_range="4.0 - 5.6", status_flag=None,
            extraction_confidence=0.95, validation_status="VALID",
        )
        resolved_conf, reasons_conf = resolve_duplicate_extractions([val_conflict1, val_conflict2])
        self.assertEqual(len(resolved_conf), 1)
        self.assertEqual(resolved_conf[0].validation_status, "REVIEW_REQUIRED")
        self.assertEqual(resolved_conf[0].interpretation_status, "UNABLE_TO_DETERMINE")
        self.assertTrue(len(reasons_conf) > 0)
        self.assertIn("Conflicting values detected", reasons_conf[0])

    def test_27_readmission_gating_preservation(self):
        """
        Verify that readmission ML inference remains strictly blocked (INSUFFICIENT_INPUT_DATA)
        for outpatient lab panels until encounter variables are explicitly provided.
        """
        payload = StructuredLabReportPayload(
            document_info={"file_name": "Abhishek Thakur_report.pdf"},
            extraction_method="DIRECT_TEXT_PDF",
            ocr_confidence=0.95,
            overall_extraction_confidence=0.95,
            laboratory_values=[
                ExtractedLabValue(
                    test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="7.80",
                    numeric_value=7.80, unit="%", reference_range="4.2 - 5.7", status_flag=None,
                    extraction_confidence=0.95, validation_status="VALID",
                ),
                ExtractedLabValue(
                    test_name="Glucose", normalized_name="max_glu_serum", raw_value_str="169",
                    numeric_value=169.0, unit="mg/dL", reference_range="70 - 140", status_flag=None,
                    extraction_confidence=0.95, validation_status="VALID",
                ),
            ],
        )
        from ocr.pipeline import check_feature_readiness
        checked = check_feature_readiness(payload)
        self.assertEqual(checked.prediction_status, "INSUFFICIENT_INPUT_DATA")
        self.assertTrue(checked.human_review_required)
        with self.assertRaises(ValueError) as ctx:
            convert_payload_to_ml_dataframe(checked)
        self.assertIn("prohibits population mode/median default substitution", str(ctx.exception))

    def test_28_clinical_summary_generation(self):
        """
        Verify that generate_clinical_summary groups analytes into diagnostic panels,
        detects multi-analyte clinical patterns (e.g., dual glycemic elevation, atherogenic dyslipidemia),
        compiles abnormal findings with medically supported causes, and attaches clinical disclaimers.
        """
        from ocr.clinical_summary import generate_clinical_summary

        lab_values = [
            ExtractedLabValue(
                test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="7.80",
                numeric_value=7.80, unit="%", reference_range="4.0 - 5.6", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="Glucose", normalized_name="max_glu_serum", raw_value_str="169",
                numeric_value=169.0, unit="mg/dL", reference_range="70 - 140", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="Total Cholesterol", normalized_name="cholesterol", raw_value_str="240",
                numeric_value=240.0, unit="mg/dL", reference_range="<200", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="Triglycerides", normalized_name="triglycerides", raw_value_str="210",
                numeric_value=210.0, unit="mg/dL", reference_range="<150", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="Creatinine", normalized_name="creatinine", raw_value_str="0.90",
                numeric_value=0.90, unit="mg/dL", reference_range="0.70 - 1.30", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
        ]

        summary = generate_clinical_summary(lab_values)
        self.assertIn("overview", summary)
        self.assertEqual(summary["overview"]["total_parameters"], 5)
        self.assertEqual(summary["overview"]["in_range_count"], 1)  # Creatinine
        self.assertEqual(summary["overview"]["abnormal_count"], 4)   # HbA1c, Glu, Chol, Trig

        # Check clinical panels
        self.assertIn("panels", summary)
        panel_ids = [p["panel_id"] for p in summary["panels"]]
        self.assertIn("glycemic", panel_ids)
        self.assertIn("lipid", panel_ids)
        self.assertIn("renal", panel_ids)

        # Check multi-analyte correlated patterns
        self.assertIn("correlated_patterns", summary)
        titles = [p["title"] for p in summary["correlated_patterns"]]
        self.assertTrue(any("Glycemic" in t for t in titles))
        self.assertTrue(any("Lipid" in t or "Dyslipidemia" in t for t in titles))

        # Check non-diagnostic disclaimer and physician guidance
        self.assertIn("clinical_guidance", summary)
        self.assertTrue(len(summary["clinical_guidance"]["actionable_recommendations"]) > 0)
        self.assertIn("physician", summary["clinical_guidance"]["disclaimer"].lower())

    def test_29_standalone_lab_pipeline_summary(self):
        """
        Verify that process_lab_report_file attaches clinical_summary directly to the StructuredLabReportPayload.
        """
        fpath = os.path.join(self.sample_dir, "sample_image_report.jpg")
        payload = process_lab_report_file(fpath)
        self.assertIsNotNone(payload.clinical_summary)
        self.assertIn("overview", payload.clinical_summary)
        self.assertIn("panels", payload.clinical_summary)
        self.assertIn("clinical_guidance", payload.clinical_summary)


    def test_30_toc_and_page_numbers_ignored(self):
        """
        Regression Test: Prevent Table of Contents entries, page reference numbers,
        dotted leader lines, and section numbers from being treated as measured results.
        """
        toc_sample_text = """
Table of Contents / Index of Tests
1.1 Complete Blood Count (CBC) ................................. 14
1.2 C-Reactive Protein (CRP), Quantitative ..................... 6
1.3 Platelet Count ............................................. 15
1.4 Alkaline Phosphatase (ALP) ................................. 10
1.5 Kidney Function Test ....................................... 12
1.6 Total Calcium .............................................. 3
Page 1 of 18
        """
        extracted, _ = extract_lab_values_from_text(toc_sample_text)
        self.assertEqual(
            len(extracted),
            0,
            f"Expected 0 extractions from TOC, but got: {[x.test_name + '=' + str(x.numeric_value) for x in extracted]}"
        )

    def test_31_narrative_prose_sentences_ignored(self):
        """
        Regression Test: Clinical advisory sentences, machine QC details, cutoff descriptions,
        and advisory dates must NOT be extracted as measured results.
        """
        narrative_sample_text = """
Clinical Information & Technical Details:
- The laboratory utilizes an automated analyzer along with 360° homogeneous sample mixing.
- CRP rises rapidly (within 6-12 hours) following acute tissue injury.
- Increased MCV is a known indicator of vitamin B12 deficiency.
- In Mentzer Index, MCV divided by RBC count is less than 13 in microcytic hypochromic anemias.
- For RDWI, a value greater than 220 is suggestive of Iron Deficiency Anemia.
- For Green and King Index, a value greater than 65 is suggestive of iron deficiency.
- Reference Interval revision was adopted (Dated: 11.02.2026).
- Unlike ESR, CRP levels are not influenced by hematologic conditions or plasma proteins.
        """
        extracted, _ = extract_lab_values_from_text(narrative_sample_text)
        self.assertEqual(
            len(extracted),
            0,
            f"Expected 0 extractions from narrative prose, but got: {[x.test_name + '=' + str(x.numeric_value) for x in extracted]}"
        )

    def test_32_abhishek_thakur_exact_fixture_conflict_resolution(self):
        """
        Regression Test: Verifies that all 7 reported conflicts are resolved accurately:
        CRP: 3.18 (not 6)
        Platelets: 157 (not 360)
        RBC: 4.61 (not 13)
        MCV: 97.0 (not 12)
        RDW-CV: 15.9 (not 220)
        ALP: 110.00 (not 11.02)
        Green & King Index: 101 (not 65)
        """
        combined_report_text = """
DEPARTMENT OF BIOCHEMISTRY
C - Reactive Protein (CRP) Quantitative
3.18 mg/L < 5.0
Note: CRP rises rapidly (within 6-12 hours) following tissue injury.

DEPARTMENT OF HAEMATOLOGY
Technical Note: Automated 5-part analyzer with 360° homogeneous sample mixing.

Complete Blood Count
Haemoglobin (HB) : 14.8 g/dL 13.0 - 17.0
RBC Count : 4.61 10^6/µl 4.50 - 5.50
MCV : 97.0 fL 83.0 - 101.0
RDW-CV : 15.9 % 11.6 - 14.0
Mentzer Index : 21.04 Ratio
Red Cell Distribution Width Index (RDWI) : 334.55 Ratio
Green and King Index : 101 Ratio
Platelet Count : 157 10^3/µl 150 - 410

Clinical Interpretation:
- In Mentzer formula, MCV divided by RBC count is less than 13 in thalassemia trait.
- High MCV is an indicator of vitamin B12 deficiency.
- For RDWI, a value greater than 220 is suggestive of Iron Deficiency Anemia.
- For Green and King Index, a value greater than 65 is suggestive of iron deficiency.

LIVER FUNCTION TEST
Alkaline Phosphatase (ALP)
110.00 U/L 43.00 - 115.00
Advisory: Reference range revised (Dated: 11.02.2026).
        """
        extracted, _ = extract_lab_values_from_text(combined_report_text)
        deduped, conflicts = resolve_duplicate_extractions(extracted)
        self.assertEqual(len(conflicts), 0, f"Expected 0 conflicts, got: {conflicts}")

        val_map = {v.normalized_name: v for v in deduped}

        self.assertIn("crp", val_map)
        self.assertEqual(val_map["crp"].numeric_value, 3.18)

        self.assertIn("platelets", val_map)
        self.assertEqual(val_map["platelets"].numeric_value, 157.0)

        self.assertIn("rbc", val_map)
        self.assertEqual(val_map["rbc"].numeric_value, 4.61)

        self.assertIn("mcv", val_map)
        self.assertEqual(val_map["mcv"].numeric_value, 97.0)

        self.assertIn("rdw", val_map)
        self.assertEqual(val_map["rdw"].numeric_value, 15.9)

        self.assertIn("alp", val_map)
        self.assertEqual(val_map["alp"].numeric_value, 110.00)

        self.assertIn("green_king_index", val_map)
        self.assertEqual(val_map["green_king_index"].numeric_value, 101.0)

    def test_33_unperformed_tests_calcium_not_fabricated(self):
        """
        Regression Test: Calcium is captured as 'Test not taken' without fabricating a number.
        ESR mentioned only in prose comparisons is NOT extracted as a lab result.
        """
        sample_text = """
SUMMARY REPORT
Total Calcium
Test not taken

CLINICAL NOTE:
Unlike ESR, CRP levels are not influenced by hematologic conditions.
        """
        extracted, _ = extract_lab_values_from_text(sample_text)
        deduped, _ = resolve_duplicate_extractions(extracted)

        val_map = {v.normalized_name: v for v in deduped}

        # Calcium must be present as unperformed, with NO fabricated numeric value
        self.assertIn("calcium", val_map)
        self.assertIsNone(val_map["calcium"].numeric_value)
        self.assertEqual(val_map["calcium"].raw_value_str, "Test not taken")
        self.assertEqual(val_map["calcium"].validation_status, "REVIEW_REQUIRED")
        self.assertEqual(val_map["calcium"].interpretation_status, "UNABLE_TO_DETERMINE")
        self.assertIn("Measurement unavailable", val_map["calcium"].validation_message)

        # ESR must NOT be extracted
        self.assertNotIn("esr", val_map)

    def test_34_derived_parameter_verification_and_unresolved_sources(self):
        """
        Regression Test: Derived indices (Mentzer, Green & King) are recalculated and validated
        when source values are verified. If required source values are missing or unresolved,
        they must be flagged as REVIEW_REQUIRED / UNABLE_TO_DETERMINE.
        """
        # Case A: Valid sources present -> Green and King verified
        valid_items = [
            ExtractedLabValue(
                test_name="Haemoglobin", normalized_name="hemoglobin", raw_value_str="14.8",
                numeric_value=14.8, unit="g/dL", reference_range="13-17", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="MCV", normalized_name="mcv", raw_value_str="97.0",
                numeric_value=97.0, unit="fL", reference_range="83-101", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="RDW-CV", normalized_name="rdw", raw_value_str="15.9",
                numeric_value=15.9, unit="%", reference_range="11.6-14.0", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            ExtractedLabValue(
                test_name="Green and King Index", normalized_name="green_king_index", raw_value_str="101",
                numeric_value=101.0, unit="Ratio", reference_range=">65", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
        ]
        recalc_items, warnings = validate_and_recalculate_derived_indices(valid_items)
        gk = [x for x in recalc_items if x.normalized_name == "green_king_index"][0]
        self.assertEqual(gk.validation_status, "VALID")
        self.assertIn("Verified by recalculation", gk.validation_message)

        # Case B: Source measurement missing -> Derived index marked REVIEW_REQUIRED
        missing_source_items = [
            ExtractedLabValue(
                test_name="MCV", normalized_name="mcv", raw_value_str="97.0",
                numeric_value=97.0, unit="fL", reference_range="83-101", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
            # Missing RBC!
            ExtractedLabValue(
                test_name="Mentzer Index", normalized_name="mentzer_index", raw_value_str="21.04",
                numeric_value=21.04, unit="Ratio", reference_range="<13", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            ),
        ]
        recalc_missing, warnings_missing = validate_and_recalculate_derived_indices(missing_source_items)
        mentzer = [x for x in recalc_missing if x.normalized_name == "mentzer_index"][0]
        self.assertEqual(mentzer.validation_status, "REVIEW_REQUIRED")
        self.assertEqual(mentzer.interpretation_status, "UNABLE_TO_DETERMINE")
        self.assertIn("required source measurement(s) (rbc) are unresolved or missing", mentzer.validation_message)

    def test_35_genuine_ambiguity_preserves_uncertainty(self):
        """
        Regression Test: Genuinely conflicting duplicate measurements (e.g. HbA1c 6.5% vs 12.4%)
        must NEVER be silently picked; uncertainty must be preserved with REVIEW_REQUIRED
        and UNABLE_TO_DETERMINE.
        """
        val1 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="6.5",
            numeric_value=6.5, unit="%", reference_range="4.0-5.6", status_flag="HIGH",
            extraction_confidence=0.90, validation_status="VALID",
        )
        val2 = ExtractedLabValue(
            test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="12.4",
            numeric_value=12.4, unit="%", reference_range="4.0-5.6", status_flag="HIGH",
            extraction_confidence=0.90, validation_status="VALID",
        )
        resolved, reasons = resolve_duplicate_extractions([val1, val2])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].validation_status, "REVIEW_REQUIRED")
        self.assertEqual(resolved[0].interpretation_status, "UNABLE_TO_DETERMINE")
        self.assertIn("Conflicting duplicate extractions", resolved[0].validation_message)
        self.assertTrue(len(reasons) > 0)

    def test_36_full_pdf_pipeline_on_abhishek_thakur_report(self):
        """
        End-to-End Regression Test on actual E:/Abhishek Thakur_report.pdf if present.
        Verifies all 7 target conflicts are eliminated and true values extracted cleanly.
        """
        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found at path.")

        payload = process_lab_report_file(pdf_path)
        val_map = {v.normalized_name: v for v in payload.laboratory_values}

        # Verify no conflicting duplicate extraction review reasons
        conflict_reasons = [r for r in payload.review_reasons if "Conflicting values detected" in r or "Conflicting duplicate" in r]
        self.assertEqual(len(conflict_reasons), 0, f"Unexpected conflicts: {conflict_reasons}")

        # Verify all target values
        self.assertAlmostEqual(val_map["crp"].numeric_value, 3.18, places=2)
        self.assertEqual(val_map["platelets"].numeric_value, 157.0)
        self.assertAlmostEqual(val_map["rbc"].numeric_value, 4.61, places=2)
        self.assertAlmostEqual(val_map["mcv"].numeric_value, 97.0, places=1)
        self.assertAlmostEqual(val_map["rdw"].numeric_value, 15.9, places=1)
        self.assertAlmostEqual(val_map["alp"].numeric_value, 110.00, places=2)
        self.assertEqual(val_map["green_king_index"].numeric_value, 101.0)

        # Calcium is unperformed, not fabricated
        self.assertIn("calcium", val_map)
        self.assertIsNone(val_map["calcium"].numeric_value)
        self.assertEqual(val_map["calcium"].raw_value_str, "Test not taken")

        # ESR is not falsely extracted
        self.assertNotIn("esr", val_map)


    def test_37_alp_within_range_not_marked_unresolved(self):
        """
        Regression Test: ALP with value 110.00 U/L and reference range 43.00 - 115.00
        must evaluate as WITHIN_RANGE and VALID, and never be marked unresolved.
        """
        alp_val = ExtractedLabValue(
            test_name="Alkaline Phosphatase (ALP)",
            normalized_name="alp",
            raw_value_str="110.00",
            numeric_value=110.00,
            unit="U/L",
            reference_range="43.00 - 115.00",
            status_flag=None,
            extraction_confidence=0.95,
            validation_status="PENDING",
        )
        validated = validate_extracted_value(alp_val)
        self.assertEqual(validated.validation_status, "VALID")
        self.assertEqual(validated.interpretation_status, "WITHIN_RANGE")
        self.assertEqual(validated.reference_range_source, "REPORT")
        self.assertIn("Passed physiological validation", validated.validation_message)

    def test_38_multianalyte_pattern_requires_reliable_sources(self):
        """
        Regression Test: Multi-analyte patterns must only be generated when contributing
        source values are verified and reliable. Unresolved or unperformed candidates
        must NOT trigger clinical patterns.
        """
        from ocr.clinical_summary import evaluate_multianalyte_patterns

        # Case A: Glucose is unperformed/unverified (numeric_value is None, REVIEW_REQUIRED)
        unreliable_map = {
            "max_glu_serum": ExtractedLabValue(
                test_name="Fasting Glucose", normalized_name="max_glu_serum", raw_value_str="Test not taken",
                numeric_value=None, unit=None, reference_range=None, status_flag=None,
                extraction_confidence=0.90, validation_status="REVIEW_REQUIRED", interpretation_status="UNABLE_TO_DETERMINE",
            ),
            "A1Cresult": ExtractedLabValue(
                test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="8.5",
                numeric_value=8.5, unit="%", reference_range="4.0-5.6", status_flag="HIGH",
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="HIGH",
            ),
        }
        patterns_unreliable = evaluate_multianalyte_patterns(unreliable_map)
        # Should NOT trigger concomitant acute and chronic glycemic pattern with unperformed glucose
        concomitant = [p for p in patterns_unreliable if "Concomitant" in p.get("title", "")]
        self.assertEqual(len(concomitant), 0, "Unreliable source must not trigger concomitant pattern")

        # Case B: Both glucose and HbA1c are verified VALID
        reliable_map = {
            "max_glu_serum": ExtractedLabValue(
                test_name="Fasting Glucose", normalized_name="max_glu_serum", raw_value_str="185",
                numeric_value=185.0, unit="mg/dL", reference_range="70-100", status_flag="HIGH",
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="HIGH",
            ),
            "A1Cresult": ExtractedLabValue(
                test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="8.5",
                numeric_value=8.5, unit="%", reference_range="4.0-5.6", status_flag="HIGH",
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="HIGH",
            ),
        }
        patterns_reliable = evaluate_multianalyte_patterns(reliable_map)
        self.assertTrue(any("Concomitant" in p.get("title", "") for p in patterns_reliable))

    def test_39_summary_counts_and_panel_status_consistency(self):
        """
        Regression Test: Summary counts (high, low, in_range, uncertain) must exactly match
        underlying item evaluation statuses, and panel items must reflect the same evaluation states.
        """
        from ocr.clinical_summary import generate_clinical_summary

        items = [
            ExtractedLabValue(
                test_name="Glucose", normalized_name="max_glu_serum", raw_value_str="170",
                numeric_value=170.0, unit="mg/dL", reference_range="70-100", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="HIGH",
            ),
            ExtractedLabValue(
                test_name="HbA1c", normalized_name="A1Cresult", raw_value_str="7.5",
                numeric_value=7.5, unit="%", reference_range="4.0-5.6", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="HIGH",
            ),
            ExtractedLabValue(
                test_name="Hemoglobin", normalized_name="hemoglobin", raw_value_str="10.5",
                numeric_value=10.5, unit="g/dL", reference_range="12.0-17.5", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="LOW",
            ),
            ExtractedLabValue(
                test_name="Platelets", normalized_name="platelets", raw_value_str="250",
                numeric_value=250.0, unit="10^3/uL", reference_range="150-450", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="WITHIN_RANGE",
            ),
            ExtractedLabValue(
                test_name="Creatinine", normalized_name="creatinine", raw_value_str="0.9",
                numeric_value=0.9, unit="mg/dL", reference_range="0.6-1.3", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID", interpretation_status="WITHIN_RANGE",
            ),
            ExtractedLabValue(
                test_name="Total Calcium", normalized_name="calcium", raw_value_str="Test not taken",
                numeric_value=None, unit=None, reference_range=None, status_flag=None,
                extraction_confidence=0.90, validation_status="REVIEW_REQUIRED", interpretation_status="UNABLE_TO_DETERMINE",
            ),
        ]

        summary = generate_clinical_summary(items)
        ov = summary["overview"]

        self.assertEqual(ov["total_parameters"], 6)
        self.assertEqual(ov["high_count"], 2)
        self.assertEqual(ov["low_count"], 1)
        self.assertEqual(ov["abnormal_count"], 3)
        self.assertEqual(ov["in_range_count"], 2)
        self.assertEqual(ov["uncertain_count"], 1)

        # Panel consistency
        glycemic_panel = [p for p in summary["panels"] if p["panel_id"] == "glycemic"][0]
        self.assertEqual(glycemic_panel["total_tests"], 2)
        self.assertEqual(glycemic_panel["abnormal_count"], 2)
        self.assertEqual(glycemic_panel["status"], "ATTENTION")

    def test_40_fastapi_backend_lab_analyze_e2e(self):
        """
        Regression Test: Full backend API integration via FastAPI TestClient on Abhishek Thakur PDF.
        Verifies endpoint returns 200, general lab task, no spurious conflict review reasons,
        and complete clinical summary hierarchy.
        """
        from backend.app.main import app
        from fastapi.testclient import TestClient

        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found at path.")

        client = TestClient(app)
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/v1/lab/analyze",
                files={"file": ("Abhishek_Thakur_report.pdf", f, "application/pdf")},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["modality"], "lab")
        self.assertEqual(data["task"], "report_analysis")
        payload_details = data["payload_details"]
        self.assertIsNotNone(payload_details)
        self.assertTrue(len(payload_details["laboratory_values"]) > 0)
        self.assertIsNotNone(payload_details.get("clinical_summary"))

        # Check that no conflicts were reported for the 7 target analytes
        review_reasons = data.get("review_reasons", [])
        for reason in review_reasons:
            self.assertNotIn("Conflicting values detected for lab test 'crp'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'platelets'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'rbc'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'mcv'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'rdw'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'alp'", reason)
            self.assertNotIn("Conflicting values detected for lab test 'green_king_index'", reason)

    def test_41_patient_demographics_verified_not_in_uncertain_count(self):
        """
        Regression Test: Validated patient demographics (Age, Gender) must be classified as
        verified patient metadata, and strictly excluded from unresolved laboratory measurement counts.
        """
        from ocr.clinical_summary import generate_clinical_summary
        from ocr.validation import validate_extracted_value

        age_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Age", normalized_name="age", raw_value_str="23",
                numeric_value=23.0, unit=None, reference_range=None, status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
                source_context="Patient demographic header",
            )
        )
        gender_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Gender", normalized_name="gender", raw_value_str="Male",
                numeric_value=None, unit=None, reference_range=None, status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
                source_context="Patient demographic header",
            )
        )
        glucose_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Glucose", normalized_name="max_glu_serum", raw_value_str="180",
                numeric_value=180.0, unit="mg/dL", reference_range="70-140", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            )
        )
        hb_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Hemoglobin", normalized_name="hemoglobin", raw_value_str="10.0",
                numeric_value=10.0, unit="g/dL", reference_range="13.0-17.0", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            )
        )
        platelets_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Platelets", normalized_name="platelets", raw_value_str="250",
                numeric_value=250.0, unit="x10^3/uL", reference_range="150-450", status_flag=None,
                extraction_confidence=0.95, validation_status="VALID",
            )
        )
        calcium_val = validate_extracted_value(
            ExtractedLabValue(
                test_name="Total Calcium", normalized_name="calcium", raw_value_str="Test not taken",
                numeric_value=None, unit=None, reference_range=None, status_flag=None,
                extraction_confidence=0.90, validation_status="REVIEW_REQUIRED",
            )
        )

        # 1. Demographics validation checks
        self.assertEqual(age_val.validation_status, "VALID")
        self.assertEqual(age_val.interpretation_status, "VERIFIED")
        self.assertEqual(age_val.reference_range_source, "PATIENT_METADATA")

        self.assertEqual(gender_val.validation_status, "VALID")
        self.assertEqual(gender_val.interpretation_status, "VERIFIED")
        self.assertEqual(gender_val.reference_range_source, "PATIENT_METADATA")

        # 2. Clinical summary aggregation
        items = [age_val, gender_val, glucose_val, hb_val, platelets_val, calcium_val]
        summary = generate_clinical_summary(items)
        ov = summary["overview"]

        # Only 4 laboratory test parameters (Glucose, Hemoglobin, Platelets, Calcium)
        self.assertEqual(ov["total_parameters"], 4)
        self.assertEqual(ov["high_count"], 1)       # Glucose
        self.assertEqual(ov["low_count"], 1)        # Hemoglobin
        self.assertEqual(ov["abnormal_count"], 2)   # Glucose + Hemoglobin
        self.assertEqual(ov["in_range_count"], 1)   # Platelets
        self.assertEqual(ov["uncertain_count"], 1)  # Only Calcium ("Test not taken")

        # Patient demographics dictionary is populated with verified status
        self.assertIn("patient_demographics", summary)
        p_demo = summary["patient_demographics"]
        self.assertIn("age", p_demo)
        self.assertIn("gender", p_demo)
        self.assertEqual(p_demo["age"]["status"], "VERIFIED")
        self.assertEqual(p_demo["gender"]["status"], "VERIFIED")

        # Uncertain findings list does NOT contain age or gender
        uncertain_names = [u["test_name"] for u in summary["uncertain_findings"]]
        self.assertNotIn("Age", uncertain_names)
        self.assertNotIn("Gender", uncertain_names)
        self.assertEqual(uncertain_names, ["Total Calcium"])

    def test_42_abhishek_thakur_demographics_and_counts_consistency(self):
        """
        Regression Test: Abhishek Thakur 18-page PDF report has verified Age: 23 and Gender: Male.
        Verifies:
        1. Demographics have interpretation_status == 'VERIFIED' and validation_status == 'VALID'.
        2. overview['uncertain_count'] equals exactly the number of unresolved items.
        3. Demographics are not counted as abnormal or uncertain lab measurements.
        4. Math consistency: total_parameters == in_range + high + low + uncertain.
        """
        from ocr.pipeline import process_lab_report_file

        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found.")

        result = process_lab_report_file(pdf_path)
        summary = result.clinical_summary
        self.assertIsNotNone(summary)

        # Check demographics
        p_demo = summary.get("patient_demographics", {})
        self.assertIn("age", p_demo)
        self.assertIn("gender", p_demo)
        self.assertEqual(p_demo["age"]["value"], "23")
        self.assertEqual(p_demo["age"]["status"], "VERIFIED")
        self.assertEqual(p_demo["age"]["validation_status"], "VALID")
        self.assertEqual(p_demo["gender"]["value"], "Male")
        self.assertEqual(p_demo["gender"]["status"], "VERIFIED")
        self.assertEqual(p_demo["gender"]["validation_status"], "VALID")

        # Check overview counts
        ov = summary["overview"]
        self.assertEqual(ov["uncertain_count"], len(summary["uncertain_findings"]))
        self.assertEqual(ov["high_count"] + ov["low_count"], ov["abnormal_count"])
        self.assertEqual(
            ov["total_parameters"],
            ov["in_range_count"] + ov["abnormal_count"] + ov["uncertain_count"],
        )

        # Ensure demographics are not in uncertain findings
        uncertain_test_names = [item["test_name"] for item in summary["uncertain_findings"]]
        self.assertNotIn("Age", uncertain_test_names)
        self.assertNotIn("Gender", uncertain_test_names)

    def test_43_result_specific_explanations_and_unresolved_audits(self):
        """
        Regression Test: Result-specific explanations, non-generic text, individually audited
        unresolved items, and unit deduplication.
        """
        from ocr.pipeline import process_lab_report_file
        from ocr.explanations import format_clean_reference_range

        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found.")

        result = process_lab_report_file(pdf_path)
        summary = result.clinical_summary
        self.assertIsNotNone(summary)

        # 1. No generic fallback sentence anywhere in abnormal findings or uncertain items
        GENERIC_FALLBACK = "Measured concentration falls outside established reference bounds for this parameter."
        all_abnormal = summary["abnormal_findings"]["high"] + summary["abnormal_findings"]["low"]
        self.assertGreater(len(all_abnormal), 0)

        for item in all_abnormal:
            guidance = item.get("guidance", "")
            what_it_means = item.get("what_it_means", "")
            causes_text = item.get("possible_causes_text", "")
            next_steps = item.get("next_steps", "")

            self.assertNotEqual(guidance, GENERIC_FALLBACK, f"Item {item['test_name']} has generic fallback guidance")
            self.assertNotEqual(what_it_means, GENERIC_FALLBACK, f"Item {item['test_name']} has generic fallback meaning")
            self.assertTrue(len(what_it_means) > 20, f"Item {item['test_name']} has empty/short what_it_means")
            self.assertTrue(len(causes_text) > 15, f"Item {item['test_name']} has empty/short causes_text")
            self.assertTrue(len(next_steps) > 15, f"Item {item['test_name']} has empty/short next_steps")

        # 2. Check representative findings (HbA1c, Triglycerides, ALT)
        hba1c = next(x for x in summary["abnormal_findings"]["high"] if x["normalized_name"] == "A1Cresult")
        self.assertIn("7.8", hba1c["what_it_means"])
        self.assertIn("HbA1c", hba1c["what_it_means"])
        self.assertIn("average blood glucose", hba1c["what_it_means"])
        self.assertIn("prediabetes", hba1c["possible_causes_text"])

        trig = next(x for x in summary["abnormal_findings"]["high"] if x["normalized_name"] == "triglycerides")
        self.assertIn("triglyceride", trig["what_it_means"].lower())
        self.assertIn("279.1", trig["what_it_means"])
        self.assertIn("fasting", trig["possible_causes_text"].lower())

        alt = next(x for x in summary["abnormal_findings"]["high"] if x["normalized_name"] == "alt")
        self.assertIn("alt", alt["what_it_means"].lower())
        self.assertIn("liver", alt["what_it_means"].lower())
        self.assertIn("mildly above", alt["what_it_means"])

        # 3. Check differential count relative percentage vs absolute count
        neut = next(x for x in summary["abnormal_findings"]["low"] if x["normalized_name"] == "neutrophils")
        self.assertIn("percentage", neut["what_it_means"].lower())
        self.assertIn("relative", neut["what_it_means"].lower())

        # 4. Audit all 7 unresolved items individually
        uncertain_map = {x["test_name"]: x for x in summary["uncertain_findings"]}
        self.assertEqual(len(summary["uncertain_findings"]), 7)

        # Calcium Total: Test not performed
        self.assertIn("Calcium Total", uncertain_map)
        calc = uncertain_map["Calcium Total"]
        self.assertEqual(calc["specific_status"], "Test not performed")
        self.assertEqual(calc["value"], "Test not taken")
        self.assertIn("no measurement was reported", calc["what_it_means"].lower())

        # Average Estimated Glucose: preserved 177.16, eAG derived from HbA1c
        self.assertIn("Average Estimated Glucose", uncertain_map)
        eag = uncertain_map["Average Estimated Glucose"]
        self.assertEqual(eag["value"], "177.16")
        self.assertEqual(eag["specific_status"], "Reference interval unavailable")
        self.assertIn("hba1c", eag["what_it_means"].lower())

        # Ratios and indices
        expected_indices = [
            "Bun/Creatinine Ratio",
            "Urea/Creatinine Ratio",
            "Mentzer Index",
            "RDWI",
            "Green and king index",
        ]
        for name in expected_indices:
            self.assertIn(name, uncertain_map)
            item = uncertain_map[name]
            self.assertIsNotNone(item["value"])
            self.assertEqual(item["specific_status"], "Reference interval unavailable")

        # 5. Distinguish 51 detected parameters from 49 laboratory analytes
        self.assertEqual(summary["overview"]["total_parameters"], 49)
        self.assertEqual(summary["overview"]["total_detected_count"], 51)
        self.assertEqual(len(result.laboratory_values), 51)

        # 6. Unit deduplication test
        clean_hb = format_clean_reference_range("12.0 - 17.5 g/dL", "g/dL")
        self.assertEqual(clean_hb, "12.0 - 17.5 g/dL")
        self.assertNotIn("g/dL g/dL", clean_hb)

        clean_chol = format_clean_reference_range("<200 mg/dL", "mg/dL")
        self.assertEqual(clean_chol, "<200 mg/dL")
        self.assertNotIn("mg/dL mg/dL", clean_chol)

        clean_egfr = format_clean_reference_range(">60 mL/min/1.73m2", "mL/min/1.73m2")
        self.assertEqual(clean_egfr, ">60 mL/min/1.73m2")
        self.assertNotIn("mL/min/1.73m2 mL/min/1.73m2", clean_egfr)

    def test_44_derived_index_verification_separation_and_unperformed_rules(self):
        """
        Regression Test:
        1. Absent analytes are not automatically labelled 'Test not taken'.
        2. Explicitly unperformed tests ('Test not taken') are represented accurately as unperformed.
        3. Demographics do not appear as laboratory analytes or inflate lab counts.
        4. Derived-index extraction, calculation, and clinical interpretation states are distinct.
        5. No item is simultaneously described as verified by recalculation and needing extraction verification.
        """
        from ocr.pipeline import process_lab_report_file
        from ocr.extraction import extract_lab_values_from_text
        from ocr.validation import validate_extracted_value

        # 1. Snippet with absent analyte (e.g. no Calcium, no Sodium) does NOT invent them
        snippet = """
        DEPARTMENT OF BIOCHEMISTRY
        Fasting Blood Sugar: 95 mg/dL (70 - 100)
        Serum Creatinine: 0.9 mg/dL (0.6 - 1.2)
        """
        extracted, _ = extract_lab_values_from_text(snippet)
        extracted_names = [v.test_name.lower() for v in extracted]
        self.assertNotIn("calcium", extracted_names)
        self.assertNotIn("sodium", extracted_names)
        self.assertNotIn("test not taken", [v.raw_value_str.lower() for v in extracted])

        # 2. Complete report verification on Abhishek Thakur PDF
        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found.")

        result = process_lab_report_file(pdf_path)
        summary = result.clinical_summary
        self.assertIsNotNone(summary)

        # 3. Explicit unperformed test (Calcium Total)
        calc_item = next((v for v in result.laboratory_values if "calcium" in v.test_name.lower()), None)
        self.assertIsNotNone(calc_item)
        self.assertEqual(calc_item.raw_value_str, "Test not taken")
        self.assertEqual(calc_item.explanation["specific_status"], "Test not performed")
        self.assertEqual(calc_item.validation_status, "REVIEW_REQUIRED")
        self.assertIsNone(calc_item.numeric_value)

        # 4. Derived indices have distinct extraction, calculation, and interpretation states
        derived_keys = ["mentzer_index", "rdwi", "green_king_index", "eag", "bun_cr_ratio", "urea_cr_ratio"]
        for key in derived_keys:
            val = next((v for v in result.laboratory_values if v.normalized_name == key), None)
            self.assertIsNotNone(val, f"Derived key '{key}' not found in laboratory values")
            # Extraction is Valid (number was read from report)
            self.assertEqual(val.validation_status, "VALID")
            self.assertIsNotNone(val.numeric_value)
            # Calculation is verified by formula (matches source analytes)
            self.assertIn("recalculation", val.validation_message.lower())
            # Interpretation state is distinct: Reference interval unavailable
            self.assertEqual(val.explanation["specific_status"], "Reference interval unavailable")
            # Not falsely flagged as conflicting extraction or generic extraction error
            self.assertNotIn("conflict", val.validation_message.lower())

        # 5. Demographics (Age & Gender) are excluded from laboratory evaluation count (49)
        self.assertEqual(summary["overview"]["total_parameters"], 49)
        lab_items_only = [v for v in result.laboratory_values if v.normalized_name not in ["age", "gender", "race"]]
        self.assertEqual(len(lab_items_only), 49)

    def test_45_final_clinical_hierarchy_and_demographic_exclusion(self):
        """
        Regression Test:
        1. Demographics do not appear in serialized laboratory_values (exactly 49 items).
        2. Demographic items are serialized into patient_demographics.
        3. Filter partitions add up cleanly: 14 abnormal + 28 in-range + 7 limitations = 49.
        4. What this means, Possible reasons, and What to do next are non-repetitive.
        5. Neutrophils 34.4% and ANC are interpreted separately without false 'safe' claims.
        6. Lymphocytes and ALC are interpreted separately without duplicated text.
        7. Lipids: Triglycerides and VLDL are explained as shared pathway; Total Cholesterol & LDL not flagged.
        8. Executive synthesis accurately reflects both normal (renal, electrolytes, hemoglobin) and abnormal (differentials, RDW) findings.
        """
        from ocr.pipeline import process_lab_report_file
        pdf_path = "E:/Abhishek Thakur_report.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Abhishek Thakur PDF not found.")

        result = process_lab_report_file(pdf_path)
        d = result.to_dict()

        # 1. Demographics strictly excluded from laboratory_values
        lab_vals = d["laboratory_values"]
        self.assertEqual(len(lab_vals), 49)
        demo_keys = {"age", "gender", "race", "sex"}
        for item in lab_vals:
            self.assertNotIn(item["normalized_name"], demo_keys)
            self.assertNotIn(item["test_name"].lower(), demo_keys)

        # 2. Patient demographics preserved in metadata dictionary
        self.assertIn("patient_demographics", d)
        self.assertIn("age", d["patient_demographics"])
        self.assertIn("gender", d["patient_demographics"])
        self.assertEqual(d["patient_demographics"]["age"]["raw_value_str"], "23")
        self.assertEqual(d["patient_demographics"]["gender"]["raw_value_str"], "Male")

        # 3. Filter counts and partition consistency
        abnormal = [v for v in lab_vals if v["interpretation_status"] in ["HIGH", "LOW"]]
        in_range = [v for v in lab_vals if v["interpretation_status"] == "WITHIN_RANGE" and v["validation_status"] == "VALID"]
        limitations = [v for v in lab_vals if v["validation_status"] != "VALID" or v["interpretation_status"] == "UNABLE_TO_DETERMINE" or v["numeric_value"] is None]

        self.assertEqual(len(abnormal), 14)
        self.assertEqual(len(in_range), 28)
        self.assertEqual(len(limitations), 7)
        self.assertEqual(len(abnormal) + len(in_range) + len(limitations), 49)

        # 4. Clinical wording & non-repetition across abnormal findings
        abnormal_items = result.clinical_summary["abnormal_findings"]["high"] + result.clinical_summary["abnormal_findings"]["low"]
        for item in abnormal_items:
            meaning = item.get("what_it_means") or ""
            causes = item.get("possible_causes_text") or ""
            next_steps = item.get("next_steps") or ""

            # No generic fallback text
            self.assertNotIn("Result-specific physiological review is indicated", meaning)
            self.assertNotIn("falls outside established reference bounds", meaning)
            self.assertTrue(len(meaning) > 20)
            self.assertTrue(len(causes) > 20)
            self.assertTrue(len(next_steps) > 15)

            # Meaning and Causes must not repeat the exact same sentence
            self.assertNotEqual(meaning, causes)
            if "Elevation can occur for several reasons" in causes:
                self.assertNotIn("Elevation can occur for several reasons", meaning)

        # 5. Neutrophils 34.4% vs ANC
        neut_item = next((i for i in abnormal_items if i["normalized_name"] == "neutrophils"), None)
        self.assertIsNotNone(neut_item)
        neut_meaning = neut_item["what_it_means"]
        self.assertIn("relative differential percentage", neut_meaning)
        self.assertIn("Absolute Neutrophil Count (ANC)", neut_meaning)
        self.assertNotIn("safe range", neut_meaning)

        # 6. Lymphocytes (47.2%) vs ALC (3.74)
        lym_item = next((i for i in abnormal_items if i["normalized_name"] == "lymphocytes"), None)
        alc_item = next((i for i in abnormal_items if i["normalized_name"] == "alc"), None)
        self.assertIsNotNone(lym_item)
        self.assertIsNotNone(alc_item)
        self.assertNotEqual(lym_item["what_it_means"], alc_item["what_it_means"])
        self.assertIn("relative percentage", lym_item["what_it_means"])
        self.assertIn("actual number of lymphocytes", alc_item["what_it_means"])

        # 7. Lipids: Shared pathway, Total Cholesterol & LDL not described as elevated
        trig_item = next((i for i in abnormal_items if i["normalized_name"] == "triglycerides"), None)
        vldl_item = next((i for i in abnormal_items if i["normalized_name"] == "vldl"), None)
        self.assertIsNotNone(trig_item)
        self.assertIsNotNone(vldl_item)
        self.assertIn("remain within normal reference limits", trig_item["possible_causes_text"])
        self.assertIn("single shared lipid particle pathway", vldl_item["possible_causes_text"])

        # 8. Executive synthesis accuracy
        overview = result.clinical_summary["overview"]
        narrative = overview["narrative_summary"]
        self.assertNotIn("core blood counts remain stable", narrative)
        self.assertIn("total hemoglobin is preserved alongside mild shifts in white cell differential percentages", narrative)
        self.assertIn("Renal function markers and electrolytes are within normal reference limits", narrative)


if __name__ == "__main__":
    unittest.main()



