"""
Physiological range validator, reference-range interpreter, and duplicate/conflict resolution module.
Validates extracted laboratory values against physiological bounds, verifies unit compatibility,
interprets reference ranges without guessing, and populates patient-friendly explanations.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from .schemas import ExtractedLabValue
from .explanations import get_parameter_explanation


# Physiological absolute boundary limits (values beyond these indicate unphysiological corruption / extreme risk)
PHYSIOLOGICAL_BOUNDS = {
    "A1Cresult": {"min": 2.0, "max": 25.0, "valid_categorical": ["Norm", ">7", ">8", "None"]},
    "max_glu_serum": {"min": 10.0, "max": 2000.0, "valid_categorical": ["Norm", ">200", ">300", "None"]},
    "hemoglobin": {"min": 2.0, "max": 26.0},
    "hematocrit": {"min": 5.0, "max": 75.0},
    "rbc": {"min": 0.5, "max": 12.0},
    "wbc": {"min": 0.1, "max": 250.0},
    "platelets": {"min": 5.0, "max": 2500.0},
    "mcv": {"min": 40.0, "max": 150.0},
    "mch": {"min": 10.0, "max": 50.0},
    "mchc": {"min": 15.0, "max": 50.0},
    "rdw": {"min": 5.0, "max": 40.0},
    "neutrophils": {"min": 0.0, "max": 100.0},
    "lymphocytes": {"min": 0.0, "max": 100.0},
    "sodium": {"min": 90.0, "max": 190.0},
    "potassium": {"min": 1.0, "max": 10.0},
    "chloride": {"min": 60.0, "max": 140.0},
    "bicarbonate": {"min": 2.0, "max": 60.0},
    "bun": {"min": 1.0, "max": 250.0},
    "blood_urea": {"min": 1.0, "max": 300.0},
    "creatinine": {"min": 0.1, "max": 30.0},
    "egfr": {"min": 1.0, "max": 250.0},
    "calcium": {"min": 3.0, "max": 20.0},
    "total_protein": {"min": 1.0, "max": 16.0},
    "albumin": {"min": 0.5, "max": 9.0},
    "globulin": {"min": 0.5, "max": 10.0},
    "bilirubin_total": {"min": 0.0, "max": 40.0},
    "bilirubin_direct": {"min": 0.0, "max": 25.0},
    "alp": {"min": 5.0, "max": 2000.0},
    "alt": {"min": 1.0, "max": 2000.0},
    "ast": {"min": 1.0, "max": 2000.0},
    "cholesterol_total": {"min": 20.0, "max": 1000.0},
    "triglycerides": {"min": 10.0, "max": 4000.0},
    "hdl": {"min": 5.0, "max": 200.0},
    "ldl": {"min": 5.0, "max": 600.0},
    "vldl": {"min": 1.0, "max": 300.0},
    "crp": {"min": 0.0, "max": 500.0},
    "esr": {"min": 0.0, "max": 200.0},
    "tsh": {"min": 0.01, "max": 200.0},
    "free_t4": {"min": 0.01, "max": 20.0},
    "free_t3": {"min": 0.01, "max": 20.0},
    "mentzer_index": {"min": 1.0, "max": 100.0},
    "rdwi": {"min": 10.0, "max": 2000.0},
    "green_king_index": {"min": 5.0, "max": 1500.0},
    "eag": {"min": 20.0, "max": 1000.0},
    "bun_cr_ratio": {"min": 1.0, "max": 100.0},
    "urea_cr_ratio": {"min": 1.0, "max": 150.0},
    "ag_ratio": {"min": 0.1, "max": 10.0},
    "ast_alt_ratio": {"min": 0.05, "max": 20.0},
    "chol_hdl_ratio": {"min": 0.5, "max": 30.0},
    "ldl_hdl_ratio": {"min": 0.1, "max": 30.0},
    "hdl_ldl_ratio": {"min": 0.01, "max": 10.0},
    "non_hdl_cholesterol": {"min": 10.0, "max": 1000.0},
    "time_in_hospital": {"min": 1, "max": 14},
    "num_lab_procedures": {"min": 1, "max": 150},
    "num_procedures": {"min": 0, "max": 20},
    "num_medications": {"min": 1, "max": 100},
    "number_inpatient": {"min": 0, "max": 30},
    "number_emergency": {"min": 0, "max": 50},
    "number_outpatient": {"min": 0, "max": 50},
    "number_diagnoses": {"min": 1, "max": 16},
    "age": {"min": 0, "max": 110},
}

# Standard adult reference intervals used only when verified report unit is compatible
STANDARD_REFERENCE_INTERVALS: Dict[str, Dict[str, Any]] = {
    "A1Cresult": {"low": 4.0, "high": 5.6, "unit": "%", "display": "4.0 - 5.6 %"},
    "max_glu_serum": {"low": 70.0, "high": 99.0, "unit": "mg/dL", "display": "70 - 99 mg/dL"},
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "display": "12.0 - 17.5 g/dL"},
    "hematocrit": {"low": 36.0, "high": 50.0, "unit": "%", "display": "36.0 - 50.0 %"},
    "rbc": {"low": 4.2, "high": 5.9, "unit": "10^6/uL", "display": "4.2 - 5.9 10^6/uL"},
    "wbc": {"low": 4.0, "high": 11.0, "unit": "10^3/uL", "display": "4.0 - 11.0 10^3/uL"},
    "platelets": {"low": 150.0, "high": 450.0, "unit": "10^3/uL", "display": "150 - 450 10^3/uL"},
    "mcv": {"low": 80.0, "high": 100.0, "unit": "fL", "display": "80 - 100 fL"},
    "mch": {"low": 27.0, "high": 33.0, "unit": "pg", "display": "27 - 33 pg"},
    "mchc": {"low": 32.0, "high": 36.0, "unit": "g/dL", "display": "32 - 36 g/dL"},
    "rdw": {"low": 11.5, "high": 14.5, "unit": "%", "display": "11.5 - 14.5 %"},
    "neutrophils": {"low": 40.0, "high": 70.0, "unit": "%", "display": "40.0 - 70.0 %"},
    "lymphocytes": {"low": 20.0, "high": 40.0, "unit": "%", "display": "20.0 - 40.0 %"},
    "sodium": {"low": 135.0, "high": 145.0, "unit": "mEq/L", "display": "135 - 145 mEq/L"},
    "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L", "display": "3.5 - 5.0 mEq/L"},
    "chloride": {"low": 96.0, "high": 106.0, "unit": "mEq/L", "display": "96 - 106 mEq/L"},
    "bicarbonate": {"low": 22.0, "high": 29.0, "unit": "mEq/L", "display": "22 - 29 mEq/L"},
    "bun": {"low": 7.0, "high": 20.0, "unit": "mg/dL", "display": "7 - 20 mg/dL"},
    "creatinine": {"low": 0.6, "high": 1.3, "unit": "mg/dL", "display": "0.6 - 1.3 mg/dL"},
    "egfr": {"low": 60.0, "high": None, "unit": "mL/min/1.73m2", "display": ">= 60 mL/min/1.73m2"},
    "calcium": {"low": 8.5, "high": 10.2, "unit": "mg/dL", "display": "8.5 - 10.2 mg/dL"},
    "total_protein": {"low": 6.0, "high": 8.3, "unit": "g/dL", "display": "6.0 - 8.3 g/dL"},
    "albumin": {"low": 3.5, "high": 5.0, "unit": "g/dL", "display": "3.5 - 5.0 g/dL"},
    "bilirubin_total": {"low": 0.1, "high": 1.2, "unit": "mg/dL", "display": "0.1 - 1.2 mg/dL"},
    "alp": {"low": 44.0, "high": 147.0, "unit": "U/L", "display": "44 - 147 U/L"},
    "alt": {"low": 7.0, "high": 56.0, "unit": "U/L", "display": "7 - 56 U/L"},
    "ast": {"low": 10.0, "high": 40.0, "unit": "U/L", "display": "10 - 40 U/L"},
    "cholesterol_total": {"low": None, "high": 200.0, "unit": "mg/dL", "display": "< 200 mg/dL"},
    "triglycerides": {"low": None, "high": 150.0, "unit": "mg/dL", "display": "< 150 mg/dL"},
    "hdl": {"low": 40.0, "high": None, "unit": "mg/dL", "display": ">= 40 mg/dL"},
    "ldl": {"low": None, "high": 100.0, "unit": "mg/dL", "display": "< 100 mg/dL"},
    "crp": {"low": None, "high": 3.0, "unit": "mg/L", "display": "< 3.0 mg/L"},
    "esr": {"low": 0.0, "high": 20.0, "unit": "mm/hr", "display": "0 - 20 mm/hr"},
    "tsh": {"low": 0.4, "high": 4.0, "unit": "uIU/mL", "display": "0.4 - 4.0 uIU/mL"},
}


UNPERFORMED_STATUS_LABELS = {
    "Test not taken", "Not Tested", "Not Done", "Not Performed",
    "Pending", "Cancelled", "Sample Clotted", "Sample Hemolyzed",
    "Quantity Not Sufficient", "QNS"
}

DERIVED_PARAMETER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "mentzer_index": {
        "name": "Mentzer Index",
        "sources": ["mcv", "rbc"],
        "calculate": lambda s: s["mcv"] / s["rbc"] if s["rbc"] > 0 else None,
        "tolerance": 0.15,
    },
    "rdwi": {
        "name": "Red Cell Distribution Width Index (RDWI)",
        "sources": ["mcv", "rdw", "rbc"],
        "calculate": lambda s: (s["mcv"] * s["rdw"]) / s["rbc"] if s["rbc"] > 0 else None,
        "tolerance": 0.15,
    },
    "green_king_index": {
        "name": "Green and King Index",
        "sources": ["mcv", "rdw", "hemoglobin"],
        "calculate": lambda s: (s["mcv"] ** 2 * s["rdw"]) / (s["hemoglobin"] * 100.0) if s["hemoglobin"] > 0 else None,
        "tolerance": 0.15,
    },
    "eag": {
        "name": "Estimated Average Glucose (eAG)",
        "sources": ["A1Cresult"],
        "calculate": lambda s: (28.7 * s["A1Cresult"]) - 46.7,
        "tolerance": 0.15,
    },
    "bun_cr_ratio": {
        "name": "BUN / Creatinine Ratio",
        "sources": ["bun", "creatinine"],
        "calculate": lambda s: s["bun"] / s["creatinine"] if s["creatinine"] > 0 else None,
        "tolerance": 0.15,
    },
    "urea_cr_ratio": {
        "name": "Urea / Creatinine Ratio",
        "sources": ["blood_urea", "creatinine"],
        "calculate": lambda s: s["blood_urea"] / s["creatinine"] if s["creatinine"] > 0 else None,
        "tolerance": 0.15,
    },
    "ag_ratio": {
        "name": "Albumin / Globulin Ratio",
        "sources": ["albumin", "globulin"],
        "calculate": lambda s: s["albumin"] / s["globulin"] if s["globulin"] > 0 else None,
        "tolerance": 0.15,
    },
    "ast_alt_ratio": {
        "name": "AST / ALT Ratio",
        "sources": ["ast", "alt"],
        "calculate": lambda s: s["ast"] / s["alt"] if s["alt"] > 0 else None,
        "tolerance": 0.15,
    },
    "chol_hdl_ratio": {
        "name": "Total Cholesterol / HDL Ratio",
        "sources": ["cholesterol_total", "hdl"],
        "calculate": lambda s: s["cholesterol_total"] / s["hdl"] if s["hdl"] > 0 else None,
        "tolerance": 0.15,
    },
    "ldl_hdl_ratio": {
        "name": "LDL / HDL Ratio",
        "sources": ["ldl", "hdl"],
        "calculate": lambda s: s["ldl"] / s["hdl"] if s["hdl"] > 0 else None,
        "tolerance": 0.15,
    },
    "hdl_ldl_ratio": {
        "name": "HDL / LDL Ratio",
        "sources": ["hdl", "ldl"],
        "calculate": lambda s: s["hdl"] / s["ldl"] if s["ldl"] > 0 else None,
        "tolerance": 0.15,
    },
    "non_hdl_cholesterol": {
        "name": "Non-HDL Cholesterol",
        "sources": ["cholesterol_total", "hdl"],
        "calculate": lambda s: s["cholesterol_total"] - s["hdl"],
        "tolerance": 0.15,
    },
}


class ParsedReferenceRange:
    """
    Structured representation of a parsed laboratory reference range.
    Preserves numerical bounds and the exact comparison operator ('<', '<=', '>', '>=', 'range').
    Supports backward-compatible 2-element tuple unpacking: `low, high = parse_reference_range(...)`.
    """
    def __init__(self, low: Optional[float], high: Optional[float], op: Optional[str] = None):
        self.low = low
        self.high = high
        self.op = op

    def __iter__(self):
        return iter((self.low, self.high))

    def __getitem__(self, index):
        return (self.low, self.high)[index]

    def __len__(self):
        return 2

    def __repr__(self):
        return f"ParsedReferenceRange(low={self.low}, high={self.high}, op={repr(self.op)})"


def parse_reference_range(range_str: Optional[str]) -> ParsedReferenceRange:
    """
    Parse a numerical lower and upper bound from a reference range string,
    preserving comparison operators ('<', '<=', '>', '>=', 'range').
    Supports formats like '4.0 - 5.6', '(70-100)', '< 200', '<= 200', '>= 60', '> 60'.
    Never silently guesses bounds.
    """
    if not range_str:
        return ParsedReferenceRange(None, None, None)

    clean = range_str.strip()
    clean = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", clean).strip()

    # Pattern: '<=' or 'less than or equal to'
    match_lte = re.match(r"^(?:<=|less than or equal to)\s*([0-9\.]+)", clean, re.IGNORECASE)
    if match_lte:
        try:
            return ParsedReferenceRange(None, float(match_lte.group(1)), "<=")
        except ValueError:
            return ParsedReferenceRange(None, None, None)

    # Pattern: '<' or 'less than'
    match_lt = re.match(r"^(?:<|less than)\s*([0-9\.]+)", clean, re.IGNORECASE)
    if match_lt:
        try:
            return ParsedReferenceRange(None, float(match_lt.group(1)), "<")
        except ValueError:
            return ParsedReferenceRange(None, None, None)

    # Pattern: '>=' or 'greater than or equal to'
    match_gte = re.match(r"^(?:>=|greater than or equal to)\s*([0-9\.]+)", clean, re.IGNORECASE)
    if match_gte:
        try:
            return ParsedReferenceRange(float(match_gte.group(1)), None, ">=")
        except ValueError:
            return ParsedReferenceRange(None, None, None)

    # Pattern: '>' or 'greater than'
    match_gt = re.match(r"^(?:>|greater than)\s*([0-9\.]+)", clean, re.IGNORECASE)
    if match_gt:
        try:
            return ParsedReferenceRange(float(match_gt.group(1)), None, ">")
        except ValueError:
            return ParsedReferenceRange(None, None, None)

    # Pattern: 'X - Y' or 'X to Y'
    match_range = re.search(r"([0-9\.]+)\s*(?:-|–|—|to)\s*([0-9\.]+)", clean, re.IGNORECASE)
    if match_range:
        try:
            low = float(match_range.group(1))
            high = float(match_range.group(2))
            return ParsedReferenceRange(low, high, "range")
        except ValueError:
            return ParsedReferenceRange(None, None, None)

    return ParsedReferenceRange(None, None, None)


def determine_interpretation_status(
    numeric_val: Optional[float],
    unit: Optional[str] = None,
    raw_reference_range: Optional[str] = None,
    norm_key: Optional[str] = None,
    status_flag: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Determine interpretation status ('WITHIN_RANGE', 'HIGH', 'LOW', 'UNABLE_TO_DETERMINE')
    without silently guessing values, units, or reference ranges.
    Preserves strict inequality semantics at boundaries.

    Returns:
        Tuple of (interpretation_status, resolved_reference_range_display, range_source)
    """
    # Demographics (Age, Gender, Race) are patient metadata, not diagnostic lab analytes
    if norm_key in ["age", "gender", "race"]:
        return "VERIFIED", None, "PATIENT_METADATA"

    # 1. Non-numeric or missing value
    if numeric_val is None:
        if status_flag in ["HIGH", "LOW", "NORMAL"]:
            status_map = {"HIGH": "HIGH", "LOW": "LOW", "NORMAL": "WITHIN_RANGE"}
            return status_map[status_flag], raw_reference_range, "REPORT" if raw_reference_range else None
        return "UNABLE_TO_DETERMINE", raw_reference_range, None

    # 2. Check if report provided a parseable reference range
    if raw_reference_range:
        parsed_ref = parse_reference_range(raw_reference_range)
        low, high = parsed_ref.low, parsed_ref.high
        op = parsed_ref.op

        if op == "<":
            if numeric_val < high:
                return "WITHIN_RANGE", raw_reference_range, "REPORT"
            else:
                # Equality or greater violates strict '<' boundary
                return "HIGH", raw_reference_range, "REPORT"
        elif op == "<=":
            if numeric_val <= high:
                return "WITHIN_RANGE", raw_reference_range, "REPORT"
            else:
                return "HIGH", raw_reference_range, "REPORT"
        elif op == ">":
            if numeric_val > low:
                return "WITHIN_RANGE", raw_reference_range, "REPORT"
            else:
                # Equality or less violates strict '>' boundary
                return "LOW", raw_reference_range, "REPORT"
        elif op == ">=":
            if numeric_val >= low:
                return "WITHIN_RANGE", raw_reference_range, "REPORT"
            else:
                return "LOW", raw_reference_range, "REPORT"
        elif op == "range":
            if low is not None and numeric_val < low:
                return "LOW", raw_reference_range, "REPORT"
            elif high is not None and numeric_val > high:
                return "HIGH", raw_reference_range, "REPORT"
            else:
                return "WITHIN_RANGE", raw_reference_range, "REPORT"

    # 3. Check if standard reference range applies AND unit is compatible
    if norm_key in STANDARD_REFERENCE_INTERVALS:
        std_info = STANDARD_REFERENCE_INTERVALS[norm_key]
        std_unit = std_info["unit"]

        # Only compare if verified unit matches standard unit
        if unit and unit.lower() == std_unit.lower():
            parsed_std = parse_reference_range(std_info["display"])
            s_low, s_high = parsed_std.low, parsed_std.high
            s_op = parsed_std.op

            if s_op == "<":
                if numeric_val < s_high:
                    return "WITHIN_RANGE", std_info["display"], "STANDARD_REFERENCE"
                else:
                    return "HIGH", std_info["display"], "STANDARD_REFERENCE"
            elif s_op == "<=":
                if numeric_val <= s_high:
                    return "WITHIN_RANGE", std_info["display"], "STANDARD_REFERENCE"
                else:
                    return "HIGH", std_info["display"], "STANDARD_REFERENCE"
            elif s_op == ">":
                if numeric_val > s_low:
                    return "WITHIN_RANGE", std_info["display"], "STANDARD_REFERENCE"
                else:
                    return "LOW", std_info["display"], "STANDARD_REFERENCE"
            elif s_op == ">=":
                if numeric_val >= s_low:
                    return "WITHIN_RANGE", std_info["display"], "STANDARD_REFERENCE"
                else:
                    return "LOW", std_info["display"], "STANDARD_REFERENCE"
            elif s_op == "range":
                if s_low is not None and numeric_val < s_low:
                    return "LOW", std_info["display"], "STANDARD_REFERENCE"
                elif s_high is not None and numeric_val > s_high:
                    return "HIGH", std_info["display"], "STANDARD_REFERENCE"
                else:
                    return "WITHIN_RANGE", std_info["display"], "STANDARD_REFERENCE"

    # 4. If report provided a clear status flag (e.g. H / HIGH, L / LOW, NORMAL)
    if status_flag:
        flag_upper = status_flag.upper()
        if flag_upper in ["HIGH", "H", "CRITICAL"]:
            return "HIGH", raw_reference_range, "REPORT" if raw_reference_range else None
        elif flag_upper in ["LOW", "L"]:
            return "LOW", raw_reference_range, "REPORT" if raw_reference_range else None
        elif flag_upper in ["NORMAL", "NORM", "WITHIN_RANGE"]:
            return "WITHIN_RANGE", raw_reference_range, "REPORT" if raw_reference_range else None

    # 5. Default when unit or range is unverified
    return "UNABLE_TO_DETERMINE", raw_reference_range, None


def validate_extracted_value(lab_val: ExtractedLabValue) -> ExtractedLabValue:
    """
    Validate a single extracted laboratory value against physiological bounds,
    interpret reference range alignment without guessing, and attach patient explanation.

    Args:
        lab_val: ExtractedLabValue instance.

    Returns:
        Updated ExtractedLabValue instance with validation_status, validation_message,
        interpretation_status, reference_range_source, and explanation attached.
    """
    norm_key = lab_val.normalized_name

    # Check unmapped test name
    if not norm_key:
        lab_val.validation_status = "REVIEW_REQUIRED"
        lab_val.validation_message = f"Unmapped lab test name: '{lab_val.test_name}'"
        lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
        lab_val.is_supported = False
        lab_val.explanation = get_parameter_explanation(
            normalized_name="",
            raw_test_name=lab_val.test_name,
            numeric_value=lab_val.numeric_value,
            unit=lab_val.unit,
            interpretation_status="UNABLE_TO_DETERMINE",
            reference_range=lab_val.reference_range,
        )
        return lab_val

    # Handle patient demographic metadata (Age, Gender, Race)
    if norm_key in ["age", "gender", "race"]:
        if norm_key == "age":
            val = lab_val.numeric_value
            if val is not None and 0 <= val <= 110:
                lab_val.validation_status = "VALID"
                lab_val.validation_message = f"Verified patient age: {int(val)} years."
                lab_val.interpretation_status = "VERIFIED"
                lab_val.reference_range = None
                lab_val.reference_range_source = "PATIENT_METADATA"
                lab_val.explanation = get_parameter_explanation(
                    normalized_name=norm_key,
                    raw_test_name=lab_val.test_name,
                    numeric_value=val,
                    unit=lab_val.unit or "years",
                    interpretation_status="VERIFIED",
                    reference_range=None,
                    range_source="PATIENT_METADATA",
                )
                return lab_val
            else:
                lab_val.validation_status = "REVIEW_REQUIRED"
                lab_val.validation_message = (
                    f"Patient age '{lab_val.raw_value_str}' outside physiological bounds (0-110)."
                    if val is not None
                    else f"Invalid non-numeric patient age: '{lab_val.raw_value_str}'."
                )
                lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
                return lab_val
        elif norm_key == "gender":
            gen_val = (lab_val.raw_value_str or "").strip().capitalize()
            if gen_val in ["Male", "Female", "Other"]:
                lab_val.raw_value_str = gen_val
                lab_val.validation_status = "VALID"
                lab_val.validation_message = f"Verified patient gender: {gen_val}."
                lab_val.interpretation_status = "VERIFIED"
                lab_val.reference_range = None
                lab_val.reference_range_source = "PATIENT_METADATA"
                lab_val.explanation = get_parameter_explanation(
                    normalized_name=norm_key,
                    raw_test_name=lab_val.test_name,
                    numeric_value=None,
                    unit=None,
                    interpretation_status="VERIFIED",
                    reference_range=None,
                    range_source="PATIENT_METADATA",
                )
                return lab_val
            else:
                lab_val.validation_status = "REVIEW_REQUIRED"
                lab_val.validation_message = f"Unrecognized patient gender string: '{lab_val.raw_value_str}'."
                lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
                return lab_val
        elif norm_key == "race":
            lab_val.validation_status = "VALID"
            lab_val.validation_message = f"Recorded patient demographic race/ethnicity: {lab_val.raw_value_str}."
            lab_val.interpretation_status = "VERIFIED"
            lab_val.reference_range = None
            lab_val.reference_range_source = "PATIENT_METADATA"
            return lab_val

    # Check unperformed status (e.g. "Test not taken", "Pending", "Not Tested")
    if lab_val.raw_value_str in UNPERFORMED_STATUS_LABELS or (
        lab_val.raw_value_str and any(lab_val.raw_value_str.lower() == u.lower() for u in UNPERFORMED_STATUS_LABELS)
    ):
        lab_val.validation_status = "REVIEW_REQUIRED"
        lab_val.validation_message = f"Measurement unavailable: {lab_val.raw_value_str}."
        lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
        lab_val.explanation = get_parameter_explanation(
            normalized_name=norm_key,
            raw_test_name=lab_val.test_name,
            numeric_value=None,
            unit=None,
            interpretation_status="UNABLE_TO_DETERMINE",
            reference_range=lab_val.reference_range,
            raw_value_str=lab_val.raw_value_str,
            validation_status=lab_val.validation_status,
            validation_message=lab_val.validation_message,
        )
        return lab_val

    # Check missing numeric value or invalid format
    valid_categoricals = [
        "Norm", ">7", ">8", ">200", ">300", "None", "No", "Steady", "Up", "Down",
        "Yes", "Ch", "Male", "Female", "AfricanAmerican", "Caucasian", "Hispanic", "Other"
    ]
    if lab_val.numeric_value is None and lab_val.raw_value_str not in valid_categoricals:
        lab_val.validation_status = "REVIEW_REQUIRED"
        lab_val.validation_message = f"Invalid or non-numeric value string: '{lab_val.raw_value_str}'"
        lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
        lab_val.explanation = get_parameter_explanation(
            normalized_name=norm_key,
            raw_test_name=lab_val.test_name,
            numeric_value=None,
            unit=lab_val.unit,
            interpretation_status="UNABLE_TO_DETERMINE",
            reference_range=lab_val.reference_range,
        )
        return lab_val

    # Check physiological bounds
    if norm_key in PHYSIOLOGICAL_BOUNDS:
        bounds = PHYSIOLOGICAL_BOUNDS[norm_key]
        val = lab_val.numeric_value

        if val is not None:
            if "min" in bounds and val < bounds["min"]:
                lab_val.validation_status = "REVIEW_REQUIRED"
                lab_val.validation_message = f"Value {val} below physiological minimum ({bounds['min']})."
                lab_val.interpretation_status = "LOW"
                lab_val.explanation = get_parameter_explanation(
                    normalized_name=norm_key,
                    raw_test_name=lab_val.test_name,
                    numeric_value=val,
                    unit=lab_val.unit,
                    interpretation_status="LOW",
                    reference_range=lab_val.reference_range,
                )
                return lab_val

            if "max" in bounds and val > bounds["max"]:
                lab_val.validation_status = "REVIEW_REQUIRED"
                lab_val.validation_message = f"Value {val} exceeds physiological maximum ({bounds['max']})."
                lab_val.interpretation_status = "HIGH"
                lab_val.explanation = get_parameter_explanation(
                    normalized_name=norm_key,
                    raw_test_name=lab_val.test_name,
                    numeric_value=val,
                    unit=lab_val.unit,
                    interpretation_status="HIGH",
                    reference_range=lab_val.reference_range,
                )
                return lab_val

    # Check low extraction confidence (< 0.70)
    if lab_val.extraction_confidence < 0.70:
        lab_val.validation_status = "REVIEW_REQUIRED"
        lab_val.validation_message = f"Low extraction confidence ({lab_val.extraction_confidence:.2f} < 0.70)."
        status_interp, resolved_range, r_source = determine_interpretation_status(
            lab_val.numeric_value, lab_val.unit, lab_val.reference_range, norm_key, lab_val.status_flag
        )
        lab_val.interpretation_status = status_interp
        lab_val.reference_range = resolved_range or lab_val.reference_range
        lab_val.reference_range_source = r_source
        lab_val.explanation = get_parameter_explanation(
            normalized_name=norm_key,
            raw_test_name=lab_val.test_name,
            numeric_value=lab_val.numeric_value,
            unit=lab_val.unit,
            interpretation_status=status_interp,
            reference_range=lab_val.reference_range,
            range_source=r_source,
        )
        return lab_val

    # Check missing unit for blood/metabolic laboratory tests
    lab_tests_requiring_unit = {
        "A1Cresult", "max_glu_serum", "hemoglobin", "hematocrit", "rbc", "wbc", "platelets",
        "sodium", "potassium", "chloride", "bicarbonate", "bun", "creatinine", "calcium",
        "total_protein", "albumin", "bilirubin_total", "alp", "alt", "ast", "cholesterol_total",
        "triglycerides", "hdl", "ldl", "crp", "esr", "tsh"
    }
    if norm_key in lab_tests_requiring_unit and not lab_val.unit and lab_val.raw_value_str not in valid_categoricals:
        lab_val.validation_status = "REVIEW_REQUIRED"
        lab_val.validation_message = f"Missing unit for laboratory parameter '{lab_val.test_name}'."
        lab_val.interpretation_status = "UNABLE_TO_DETERMINE"
        lab_val.explanation = get_parameter_explanation(
            normalized_name=norm_key,
            raw_test_name=lab_val.test_name,
            numeric_value=lab_val.numeric_value,
            unit=None,
            interpretation_status="UNABLE_TO_DETERMINE",
            reference_range=lab_val.reference_range,
        )
        return lab_val

    # Determine reference range alignment and populate patient-friendly explanation
    status_interp, resolved_range, r_source = determine_interpretation_status(
        lab_val.numeric_value, lab_val.unit, lab_val.reference_range, norm_key, lab_val.status_flag
    )

    lab_val.interpretation_status = status_interp
    if not lab_val.reference_range and resolved_range:
        lab_val.reference_range = resolved_range
    lab_val.reference_range_source = r_source

    if lab_val.validation_status != "REVIEW_REQUIRED":
        lab_val.validation_status = "VALID"
        lab_val.validation_message = "Passed physiological validation."

    lab_val.explanation = get_parameter_explanation(
        normalized_name=norm_key,
        raw_test_name=lab_val.test_name,
        numeric_value=lab_val.numeric_value,
        unit=lab_val.unit,
        interpretation_status=status_interp,
        reference_range=lab_val.reference_range,
        range_source=r_source,
        raw_value_str=lab_val.raw_value_str,
        validation_status=lab_val.validation_status,
        validation_message=lab_val.validation_message,
    )

    return lab_val


def resolve_duplicate_extractions(extracted_values: List[ExtractedLabValue]) -> Tuple[List[ExtractedLabValue], List[str]]:
    """
    Resolve duplicate extractions for the same laboratory feature key.
    Deduplicates identical or rounding-equivalent extractions deterministically while safely
    ignoring empty candidates when valid measurements exist, and flagging genuinely conflicting
    measured values for human review.

    Returns:
        Tuple of (deduplicated_values_list, list_of_review_reasons).
    """
    grouped: Dict[str, List[ExtractedLabValue]] = {}
    for v in extracted_values:
        key = v.normalized_name or v.test_name
        grouped.setdefault(key, []).append(v)

    resolved: List[ExtractedLabValue] = []
    review_reasons: List[str] = []

    valid_cats = {
        "Norm", ">7", ">8", ">200", ">300", "None", "No", "Steady", "Up", "Down",
        "Yes", "Ch", "Male", "Female", "AfricanAmerican", "Caucasian", "Hispanic", "Other"
    }

    for key, val_list in grouped.items():
        if len(val_list) == 1:
            resolved.append(val_list[0])
            continue

        # Filter out empty/unmeasured candidates when valid measured/categorical candidates exist
        non_empty = [
            v for v in val_list
            if v.raw_value_str and v.raw_value_str.strip() and v.numeric_value is not None
        ]
        valid_categoricals = [
            v for v in val_list
            if v.raw_value_str and v.raw_value_str.strip() and v.numeric_value is None and v.raw_value_str in valid_cats
        ]

        candidates = non_empty if non_empty else valid_categoricals if valid_categoricals else val_list

        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue

        # If all candidates are unmeasured/empty strings (e.g. [''])
        if all(not v.raw_value_str or not v.raw_value_str.strip() for v in candidates):
            best_val = max(candidates, key=lambda x: x.extraction_confidence)
            resolved.append(best_val)
            continue

        # Check for identical raw value strings
        raw_vals = set(v.raw_value_str for v in candidates)
        if len(raw_vals) == 1:
            best_val = max(
                candidates,
                key=lambda x: (
                    1 if x.reference_range else 0,
                    1 if x.unit else 0,
                    x.extraction_confidence
                )
            )
            resolved.append(best_val)
            continue

        # Check numeric equivalence for numeric candidates (e.g. 7.8 vs 7.80, or minor summary vs detail rounding)
        if all(v.numeric_value is not None for v in candidates):
            num_vals = [v.numeric_value for v in candidates]
            min_v, max_v = min(num_vals), max(num_vals)
            is_consistent = (
                (min_v == max_v)
                or (round(min_v, 2) == round(max_v, 2))
                or (round(min_v, 1) == round(max_v, 1))
                or (abs(max_v - min_v) / max(abs(max_v), 1e-6) <= 0.02)
            )

            if is_consistent:
                # Same measured value reported across summary card and detail table
                best_val = max(
                    candidates,
                    key=lambda x: (
                        1 if x.reference_range else 0,
                        1 if x.unit else 0,
                        len(x.raw_value_str or ""),
                        x.extraction_confidence
                    )
                )
                resolved.append(best_val)
                continue

        # Check if this is a derived parameter and we can verify against calculated value from source measurements
        if key in DERIVED_PARAMETER_DEFINITIONS:
            defn = DERIVED_PARAMETER_DEFINITIONS[key]
            sources = {}
            sources_valid = True
            for src_key in defn["sources"]:
                src_candidates = grouped.get(src_key, [])
                valid_src = [sc for sc in src_candidates if sc.numeric_value is not None and sc.validation_status != "REVIEW_REQUIRED"]
                if len(valid_src) == 1:
                    sources[src_key] = valid_src[0].numeric_value
                else:
                    sources_valid = False
                    break

            if sources_valid:
                try:
                    expected_val = defn["calculate"](sources)
                    if expected_val is not None:
                        tol = defn.get("tolerance", 0.15)
                        matching_candidates = [
                            c for c in candidates
                            if c.numeric_value is not None and abs(c.numeric_value - expected_val) <= max(tol * abs(expected_val), 0.5)
                        ]
                        if len(matching_candidates) == 1:
                            matching_candidates[0].validation_status = "VALID"
                            matching_candidates[0].validation_message = f"Verified by recalculation from source measurements ({expected_val:.2f})."
                            resolved.append(matching_candidates[0])
                            continue
                except Exception:
                    pass

        # Genuinely conflicting duplicate extractions -> Flag for human review
        sorted_raw_vals = sorted(list(set(v.raw_value_str for v in candidates)))
        best_val = candidates[0]
        best_val.validation_status = "REVIEW_REQUIRED"
        best_val.validation_message = f"Conflicting duplicate extractions for '{key}': {sorted_raw_vals}"
        best_val.interpretation_status = "UNABLE_TO_DETERMINE"
        resolved.append(best_val)
        review_reasons.append(f"Conflicting values detected for lab test '{key}': {sorted_raw_vals}")

    return resolved, review_reasons


def validate_and_recalculate_derived_indices(
    laboratory_values: List[ExtractedLabValue]
) -> Tuple[List[ExtractedLabValue], List[str]]:
    """
    Validate and recalculate derived indices (Mentzer Index, RDWI, Green and King Index, eAG, ratios).
    Derived indices are only marked VALID when their required source measurements are verified and compatible.
    Otherwise, they are marked REVIEW_REQUIRED / UNABLE_TO_DETERMINE.

    Args:
        laboratory_values: List of ExtractedLabValue instances.

    Returns:
        Tuple of (updated_values, list_of_derived_warnings).
    """
    valid_sources: Dict[str, float] = {}
    for v in laboratory_values:
        if v.normalized_name and v.validation_status == "VALID" and v.numeric_value is not None:
            valid_sources[v.normalized_name] = v.numeric_value

    derived_warnings: List[str] = []

    for v in laboratory_values:
        norm_key = v.normalized_name
        if not norm_key or norm_key not in DERIVED_PARAMETER_DEFINITIONS:
            continue

        defn = DERIVED_PARAMETER_DEFINITIONS[norm_key]
        required_sources = defn["sources"]
        missing_or_unresolved = [s for s in required_sources if s not in valid_sources]

        if missing_or_unresolved:
            v.validation_status = "REVIEW_REQUIRED"
            v.interpretation_status = "UNABLE_TO_DETERMINE"
            v.validation_message = (
                f"Derived index '{v.test_name}' cannot be verified: required source measurement(s) "
                f"({', '.join(missing_or_unresolved)}) are unresolved or missing from report."
            )
            derived_warnings.append(v.validation_message)
            continue

        try:
            calc_val = defn["calculate"](valid_sources)
            if calc_val is None:
                v.validation_status = "REVIEW_REQUIRED"
                v.interpretation_status = "UNABLE_TO_DETERMINE"
                v.validation_message = f"Derived index '{v.test_name}' calculation failed (division by zero or invalid source)."
                derived_warnings.append(v.validation_message)
                continue

            if v.numeric_value is not None:
                tol = defn.get("tolerance", 0.15)
                diff = abs(v.numeric_value - calc_val)
                allowed_tol = max(tol * abs(calc_val), 0.5)
                if diff <= allowed_tol:
                    v.validation_status = "VALID"
                    v.validation_message = f"Verified by recalculation from source values ({calc_val:.2f} vs reported {v.numeric_value})."
                else:
                    v.validation_status = "REVIEW_REQUIRED"
                    v.interpretation_status = "UNABLE_TO_DETERMINE"
                    v.validation_message = (
                        f"Reported value ({v.numeric_value}) is incompatible with recalculated value "
                        f"({calc_val:.2f}) from verified sources."
                    )
                    derived_warnings.append(
                        f"Derived index mismatch for '{v.test_name}': reported {v.numeric_value} vs recalculated {calc_val:.2f}"
                    )
            else:
                v.validation_status = "REVIEW_REQUIRED"
                v.interpretation_status = "UNABLE_TO_DETERMINE"
                v.validation_message = f"Derived index '{v.test_name}' was not reported with a measured value."
        except Exception as e:
            v.validation_status = "REVIEW_REQUIRED"
            v.interpretation_status = "UNABLE_TO_DETERMINE"
            v.validation_message = f"Derived calculation error for '{v.test_name}': {str(e)}"
            derived_warnings.append(v.validation_message)

        # Refresh patient-friendly explanation with updated recalculation and validation status
        v.explanation = get_parameter_explanation(
            normalized_name=norm_key,
            raw_test_name=v.test_name,
            numeric_value=v.numeric_value,
            unit=v.unit,
            interpretation_status=v.interpretation_status,
            reference_range=v.reference_range,
            range_source=v.reference_range_source,
            raw_value_str=v.raw_value_str,
            validation_status=v.validation_status,
            validation_message=v.validation_message,
        )

    return laboratory_values, derived_warnings
