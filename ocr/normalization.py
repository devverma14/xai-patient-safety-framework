"""
Laboratory test-name normalization and unit standardization module.
Maps diverse laboratory report test descriptions (e.g., 'HbA1c', 'Glycated Hemoglobin', 'Fasting Blood Sugar')
to standardized feature keys, and validates units without guessing.
"""

import re
from typing import Dict, Optional, Tuple


# Comprehensive normalization dictionary for real-world laboratory panels
TEST_NAME_SYNONYM_MAP: Dict[str, str] = {
    # Glycemic Assessment
    "hba1c": "A1Cresult",
    "hb a1c": "A1Cresult",
    "hb-a1c": "A1Cresult",
    "hbalc": "A1Cresult",  # OCR 'l' for '1'
    "hb alc": "A1Cresult",
    "hbatc": "A1Cresult",  # OCR 't' for '1'
    "hb atc": "A1Cresult",
    "a1c": "A1Cresult",
    "a1c result": "A1Cresult",
    "glycated hemoglobin": "A1Cresult",
    "glycohemoglobin": "A1Cresult",
    "hemoglobin a1c": "A1Cresult",
    "average estimated glucose": "eag",
    "estimated average glucose": "eag",

    # Glucose
    "glucose": "max_glu_serum",
    "blood glucose": "max_glu_serum",
    "serum glucose": "max_glu_serum",
    "plasma glucose": "max_glu_serum",
    "max glu serum": "max_glu_serum",
    "fasting glucose": "max_glu_serum",
    "fasting blood glucose": "max_glu_serum",
    "fasting blood sugar": "max_glu_serum",
    "random blood sugar": "max_glu_serum",
    "random glucose": "max_glu_serum",
    "blood sugar": "max_glu_serum",
    "rbs": "max_glu_serum",
    "fbs": "max_glu_serum",
    "ppbs": "max_glu_serum",

    # Complete Blood Count (CBC)
    "hemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "blood hemoglobin": "hemoglobin",
    "hematocrit": "hematocrit",
    "hct": "hematocrit",
    "pcv": "hematocrit",
    "packed cell volume": "hematocrit",
    "rbc": "rbc",
    "rbc count": "rbc",
    "total rbc count": "rbc",
    "total rbc": "rbc",
    "red blood cells": "rbc",
    "red blood cell count": "rbc",
    "erythrocyte count": "rbc",
    "erythrocytes": "rbc",
    "wbc": "wbc",
    "wbc count": "wbc",
    "total wbc count": "wbc",
    "total wbc": "wbc",
    "white blood cells": "wbc",
    "white blood cell count": "wbc",
    "leukocyte count": "wbc",
    "total leukocyte count": "wbc",
    "tlc": "wbc",
    "platelets": "platelets",
    "platelet count": "platelets",
    "plt": "platelets",
    "thrombocyte count": "platelets",
    "mcv": "mcv",
    "mean corpuscular volume": "mcv",
    "mean corp volume": "mcv",
    "mch": "mch",
    "mean corpuscular hemoglobin": "mch",
    "mean corp hb": "mch",
    "mchc": "mchc",
    "mean corpuscular hemoglobin concentration": "mchc",
    "mean corp hb conc": "mchc",
    "rdw": "rdw",
    "rdw-cv": "rdw",
    "rdw - cv": "rdw",
    "rdw-sd": "rdw_sd",
    "rdw - sd": "rdw_sd",
    "red cell distribution width": "rdw",
    "neutrophils": "neutrophils",
    "neutrophil count": "neutrophils",
    "segmented neutrophils": "neutrophils",
    "segs": "neutrophils",
    "lymphocytes": "lymphocytes",
    "lymphocyte count": "lymphocytes",
    "lymphs": "lymphocytes",
    "absolute neutrophil count": "anc",
    "anc": "anc",
    "absolute lymphocyte count": "alc",
    "alc": "alc",
    "absolute monocyte count": "amc",
    "amc": "amc",
    "absolute eosinophil count": "aec",
    "aec": "aec",
    "absolute basophil count": "abc",
    "abc": "abc",
    "mentzer index": "mentzer_index",
    "rdwi": "rdwi",
    "green and king index": "green_king_index",

    # Basic & Comprehensive Metabolic Panels (BMP / CMP)
    "sodium": "sodium",
    "serum sodium": "sodium",
    "na": "sodium",
    "potassium": "potassium",
    "serum potassium": "potassium",
    "k": "potassium",
    "chloride": "chloride",
    "serum chloride": "chloride",
    "cl": "chloride",
    "bicarbonate": "bicarbonate",
    "carbon dioxide": "bicarbonate",
    "co2": "bicarbonate",
    "total co2": "bicarbonate",
    "hco3": "bicarbonate",
    "blood urea nitrogen": "bun",
    "urea nitrogen": "bun",
    "bun": "bun",
    "blood urea": "blood_urea",
    "serum urea": "blood_urea",
    "urea": "blood_urea",
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "creat": "creatinine",
    "cr": "creatinine",
    "egfr": "egfr",
    "estimated gfr": "egfr",
    "gfr": "egfr",
    "glomerular filtration rate": "egfr",
    "calcium": "calcium",
    "serum calcium": "calcium",
    "calcium total": "calcium",
    "total calcium": "calcium",
    "ca": "calcium",
    "total protein": "total_protein",
    "protein total": "total_protein",
    "serum total protein": "total_protein",
    "tp": "total_protein",
    "albumin": "albumin",
    "serum albumin": "albumin",
    "alb": "albumin",
    "serum globulin": "globulin",
    "globulin": "globulin",
    "total bilirubin": "bilirubin_total",
    "bilirubin total": "bilirubin_total",
    "t bili": "bilirubin_total",
    "t. bili": "bilirubin_total",
    "direct bilirubin": "bilirubin_direct",
    "bilirubin direct": "bilirubin_direct",
    "d bili": "bilirubin_direct",
    "alkaline phosphatase": "alp",
    "alk phos": "alp",
    "alp": "alp",
    "alanine aminotransferase": "alt",
    "serum glutamic pyruvic transaminase": "alt",
    "sgpt": "alt",
    "alt": "alt",
    "aspartate aminotransferase": "ast",
    "serum glutamic oxaloacetic transaminase": "ast",
    "sgot": "ast",
    "ast": "ast",

    # Lipid Profile
    "total cholesterol": "cholesterol_total",
    "cholesterol total": "cholesterol_total",
    "cholesterol": "cholesterol_total",
    "chol": "cholesterol_total",
    "triglycerides": "triglycerides",
    "serum triglycerides": "triglycerides",
    "trig": "triglycerides",
    "tg": "triglycerides",
    "hdl cholesterol": "hdl",
    "hdl": "hdl",
    "high density lipoprotein": "hdl",
    "hdl-c": "hdl",
    "ldl cholesterol": "ldl",
    "ldl": "ldl",
    "low density lipoprotein": "ldl",
    "ldl-c": "ldl",
    "vldl": "vldl",
    "vldl cholesterol": "vldl",
    "total chol / hdl cholesterol ratio": "chol_hdl_ratio",
    "total chol/hdl cholesterol ratio": "chol_hdl_ratio",
    "ldl / hdl cholesterol ratio": "ldl_hdl_ratio",
    "ldl/hdl cholesterol ratio": "ldl_hdl_ratio",
    "hdl / ldl cholesterol ratio": "hdl_ldl_ratio",
    "hdl/ldl cholesterol ratio": "hdl_ldl_ratio",
    "non-hdl cholesterol": "non_hdl_cholesterol",
    "albumin/globulin ratio": "ag_ratio",
    "sgot/sgpt ratio": "ast_alt_ratio",
    "bun/creatinine ratio": "bun_cr_ratio",
    "urea/creatinine ratio": "urea_cr_ratio",

    # Inflammatory & Thyroid Markers
    "c-reactive protein": "crp",
    "c reactive protein": "crp",
    "crp": "crp",
    "hs-crp": "crp",
    "erythrocyte sedimentation rate": "esr",
    "esr": "esr",
    "sed rate": "esr",
    "thyroid stimulating hormone": "tsh",
    "tsh": "tsh",
    "s. tsh": "tsh",
    "thyrotropin": "tsh",
    "free t4": "free_t4",
    "ft4": "free_t4",
    "free thyroxine": "free_t4",
    "free t3": "free_t3",
    "ft3": "free_t3",

    # Clinical Encounter & Model Context Features
    "inpatient visits": "number_inpatient",
    "prior inpatient visits": "number_inpatient",
    "inpatient admissions": "number_inpatient",
    "prior inpatient": "number_inpatient",
    "emergency visits": "number_emergency",
    "emergency room visits": "number_emergency",
    "er visits": "number_emergency",
    "outpatient visits": "number_outpatient",
    "prior outpatient": "number_outpatient",
    "time in hospital": "time_in_hospital",
    "hospital stay duration": "time_in_hospital",
    "length of stay": "time_in_hospital",
    "hospital stay": "time_in_hospital",
    "number of lab procedures": "num_lab_procedures",
    "lab procedures count": "num_lab_procedures",
    "lab count": "num_lab_procedures",
    "number of medications": "num_medications",
    "medications count": "num_medications",
    "total diagnoses": "number_diagnoses",
    "diagnoses count": "number_diagnoses",

    # Demographics & Medications
    "age": "age",
    "patient age": "age",
    "gender": "gender",
    "sex": "gender",
    "race": "race",
    "ethnicity": "race",
    "insulin": "insulin",
    "insulin dosage": "insulin",
    "metformin": "metformin",
    "glipizide": "glipizide",
    "glyburide": "glyburide",
    "diabetes med": "diabetesMed",
    "diabetes medication": "diabetesMed",
    "medication change": "change",
}


def clean_test_string(raw: str) -> str:
    """Clean raw test string for robust matching."""
    s = raw.strip().lower()
    # Strip smart quotes, apostrophes, and accents
    s = re.sub(r"[\u2018\u2019\u201c\u201d\u00b4\'\`\"\ufffd\?\*]", "", s)
    # Strip common trailing and leading punctuation
    s = re.sub(r"^[\s\:\-\,\(\)\[\]]+|[\s\:\-\,\(\)\[\]]+$", "", s)
    # Normalize common OCR typos (e.g. 'alc' or 'atc' for 'a1c')
    if ("alc" in s or "atc" in s) and "calcium" not in s and "alk" not in s:
        s = s.replace("alc", "a1c").replace("atc", "a1c")
    return s.strip()


def normalize_test_name(raw_test_name: str) -> Optional[str]:
    """
    Map a raw laboratory test string to a standardized feature key.

    Args:
        raw_test_name: Extracted string (e.g., 'HbA1c Level', 'Fasting Blood Glucose', 'Hgb').

    Returns:
        Standardized feature key string or None if unmapped.
    """
    if not raw_test_name:
        return None

    clean_name = clean_test_string(raw_test_name)

    # 1. Direct dictionary match
    if clean_name in TEST_NAME_SYNONYM_MAP:
        return TEST_NAME_SYNONYM_MAP[clean_name]

    # 2. Match without inner special characters
    alphanumeric_clean = re.sub(r"[^a-z0-9\s]", "", clean_name).strip()
    if alphanumeric_clean in TEST_NAME_SYNONYM_MAP:
        return TEST_NAME_SYNONYM_MAP[alphanumeric_clean]

    # 3. Substring matching ordered by synonym length (longest match first)
    sorted_synonyms = sorted(TEST_NAME_SYNONYM_MAP.keys(), key=len, reverse=True)
    for syn in sorted_synonyms:
        # Match as whole word or phrase
        pattern = rf"\b{re.escape(syn)}\b"
        if re.search(pattern, clean_name):
            return TEST_NAME_SYNONYM_MAP[syn]

    return None


def normalize_unit(unit_str: Optional[str], target_feature: str) -> Tuple[Optional[str], bool]:
    """
    Validate and normalize unit strings for a given laboratory parameter.
    Never silently guesses a missing unit.

    Returns:
        Tuple of (standardized_unit, is_valid_unit).
    """
    if not unit_str or not str(unit_str).strip():
        return None, False

    clean_u = str(unit_str).strip().lower().replace("µ", "u").replace("²", "2")
    # Remove surrounding brackets or parenthesis
    clean_u = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", clean_u).strip()

    if target_feature.endswith("_ratio") or target_feature.endswith("_index") or clean_u == "ratio":
        if clean_u == "ratio":
            return "Ratio", True
        return clean_u, True

    # Glycemic Assessment
    if target_feature == "A1Cresult":
        if clean_u in ["%", "percent", "% of total hgb"]:
            return "%", True
        if clean_u in ["mmol/mol"]:
            return "mmol/mol", True
        return clean_u, False

    # Blood Glucose
    elif target_feature == "max_glu_serum":
        if clean_u in ["mg/dl", "mg/100ml", "mg%"]:
            return "mg/dL", True
        if clean_u in ["mmol/l"]:
            return "mmol/L", True
        return clean_u, False

    # Complete Blood Count (CBC)
    elif target_feature in ["hemoglobin", "mchc"]:
        if clean_u in ["g/dl", "gm/dl", "g/100ml"]:
            return "g/dL", True
        if clean_u in ["g/l"]:
            return "g/L", True
        return clean_u, False

    elif target_feature in ["hematocrit", "rdw", "neutrophils", "lymphocytes"]:
        if clean_u in ["%", "percent"]:
            return "%", True
        return clean_u, False

    elif target_feature in ["rbc"]:
        if clean_u in ["10^6/ul", "m/ul", "mil/ul", "x10^6/ul", "10*6/ul", "x10e6/ul", "10^6/mm3", "m/cumm"]:
            return "10^6/uL", True
        return clean_u, False

    elif target_feature in ["wbc", "platelets"]:
        if clean_u in ["10^3/ul", "k/ul", "k/ull", "thous/ul", "x10^3/ul", "10*3/ul", "x10e3/ul", "/ul", "/ull", "cells/ul", "cumm", "k/cumm"]:
            return "10^3/uL", True
        return clean_u, False

    elif target_feature == "mcv":
        if clean_u in ["fl", "femtoliter", "um^3"]:
            return "fL", True
        return clean_u, False

    elif target_feature == "mch":
        if clean_u in ["pg", "picogram"]:
            return "pg", True
        return clean_u, False

    # Electrolytes
    elif target_feature in ["sodium", "potassium", "chloride", "bicarbonate"]:
        if clean_u in ["meq/l", "meq/liter", "mmol/l"]:
            return "mEq/L", True
        return clean_u, False

    # Metabolic Chemistries
    elif target_feature in ["bun", "blood_urea", "creatinine", "calcium", "bilirubin_total", "bilirubin_direct", "cholesterol_total", "triglycerides", "hdl", "ldl", "vldl"]:
        if clean_u in ["mg/dl", "mg/100ml", "mg%"]:
            return "mg/dL", True
        if clean_u in ["umol/l", "mmol/l"]:
            return clean_u, True
        return clean_u, False

    elif target_feature in ["total_protein", "albumin", "globulin"]:
        if clean_u in ["g/dl", "gm/dl"]:
            return "g/dL", True
        if clean_u in ["g/l"]:
            return "g/L", True
        return clean_u, False

    elif target_feature in ["alp", "alt", "ast"]:
        if clean_u in ["u/l", "iu/l", "units/l"]:
            return "U/L", True
        return clean_u, False

    elif target_feature == "egfr":
        if clean_u in ["ml/min/1.73m2", "ml/min", "ml/min/1.73m^2", "ml/min/1.73"]:
            return "mL/min/1.73m2", True
        return clean_u, False

    elif target_feature == "crp":
        if clean_u in ["mg/l"]:
            return "mg/L", True
        if clean_u in ["mg/dl"]:
            return "mg/dL", True
        return clean_u, False

    elif target_feature == "esr":
        if clean_u in ["mm/hr", "mm/h", "mm/1hr"]:
            return "mm/hr", True
        return clean_u, False

    elif target_feature == "tsh":
        if clean_u in ["uiu/ml", "miu/l", "uiu/l", "microu/ml"]:
            return "uIU/mL", True
        return clean_u, False

    # Derived indices and ratios
    elif target_feature in [
        "mentzer_index", "rdwi", "green_king_index", "ag_ratio", "ast_alt_ratio",
        "bun_cr_ratio", "urea_cr_ratio", "chol_hdl_ratio", "ldl_hdl_ratio", "hdl_ldl_ratio"
    ]:
        if clean_u in ["ratio", "index"]:
            return "Ratio", True
        return clean_u, False

    # Clinical encounter / utilization features
    elif target_feature in ["time_in_hospital"]:
        if clean_u in ["days", "day", "d"]:
            return "days", True
        return clean_u, False

    elif target_feature in ["number_inpatient", "number_emergency", "number_outpatient"]:
        if clean_u in ["visits", "visit", "admissions", "wisits"]:  # wisits handles OCR typo
            return "visits", True
        return clean_u, False

    elif target_feature in ["num_lab_procedures", "num_medications", "number_diagnoses"]:
        if clean_u in ["count", "tests", "diagnoses", "meds"]:
            return "count", True
        return clean_u, False

    # Unmapped parameter default behavior
    return clean_u, True
