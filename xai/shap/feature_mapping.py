"""
Feature name mapping and clinical display formatter module for tabular SHAP attributions.
Maps one-hot encoded feature names (e.g., 'number_inpatient', 'medical_specialty_InternalMedicine',
'diag_1_Circulatory') to clear, human-readable clinical feature labels.
Strictly excludes all PII fields (patient_nbr, encounter_id).
"""

from typing import List, Dict, Any
import re


# Clinical dictionary for raw feature names
FEATURE_LABEL_MAP = {
    "time_in_hospital": "Time in Hospital (days)",
    "num_lab_procedures": "Number of Lab Procedures",
    "num_procedures": "Number of Non-Lab Procedures",
    "num_medications": "Number of Medications",
    "number_outpatient": "Outpatient Visits in Prior Year",
    "number_emergency": "Emergency Visits in Prior Year",
    "number_inpatient": "Inpatient Visits in Prior Year",
    "num_outpatient": "Outpatient Visits in Prior Year",
    "num_emergency": "Emergency Visits in Prior Year",
    "num_inpatient": "Inpatient Visits in Prior Year",
    "number_diagnoses": "Total Number of Diagnoses",
    "admission_type_id": "Admission Type ID",
    "discharge_disposition_id": "Discharge Disposition ID",
    "admission_source_id": "Admission Source ID",
    "age": "Patient Age Bracket",
    "gender": "Patient Gender",
    "race": "Patient Race/Ethnicity",
    "payer_code": "Insurance Payer Code",
    "medical_specialty": "Admitting Medical Specialty",
    "diag_1": "Primary Diagnosis Code/Group",
    "diag_2": "Secondary Diagnosis Code/Group",
    "diag_3": "Tertiary Diagnosis Code/Group",
    "max_glu_serum": "Glucose Serum Test Result",
    "A1Cresult": "HbA1c Test Result",
    "metformin": "Metformin Dosage",
    "repaglinide": "Repaglinide Dosage",
    "nateglinide": "Nateglinide Dosage",
    "chlorpropamide": "Chlorpropamide Dosage",
    "glimepiride": "Glimepiride Dosage",
    "acetohexamide": "Acetohexamide Dosage",
    "glipizide": "Glipizide Dosage",
    "glyburide": "Glyburide Dosage",
    "tolbutamide": "Tolbutamide Dosage",
    "pioglitazone": "Pioglitazone Dosage",
    "rosiglitazone": "Rosiglitazone Dosage",
    "acarbose": "Acarbose Dosage",
    "miglitol": "Miglitol Dosage",
    "troglitazone": "Troglitazone Dosage",
    "tolazamide": "Tolazamide Dosage",
    "insulin": "Insulin Dosage",
    "change": "Diabetes Med Change Status",
    "diabetesMed": "Prescribed Diabetes Medication",
}


def format_human_readable_feature_name(feature_name: str) -> str:
    """
    Convert a preprocessed feature string (numerical or OHE categorical)
    into a human-readable clinical descriptor string.

    Examples:
        - 'time_in_hospital' -> 'Time in Hospital (days)'
        - 'medical_specialty_Cardiology' -> 'Admitting Medical Specialty: Cardiology'
        - 'age_[70-80)' -> 'Patient Age Bracket: [70-80)'
        - 'insulin_Up' -> 'Insulin Dosage: Up'

    Args:
        feature_name: Transformed feature column name.

    Returns:
        Formatted human-readable string.
    """
    # Safety Check: PII exclusion
    if feature_name in ["encounter_id", "patient_nbr"]:
        raise ValueError(f"PII attribute '{feature_name}' must not be present in feature space!")

    # Check if numerical column direct match
    if feature_name in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[feature_name]

    # Check if One-Hot Encoded categorical feature (contains underscore)
    for raw_prefix, display_label in FEATURE_LABEL_MAP.items():
        if feature_name.startswith(raw_prefix + "_"):
            category_value = feature_name[len(raw_prefix) + 1 :]
            return f"{display_label}: {category_value}"

    # Fallback formatting: replace underscores with spaces and title case
    formatted = feature_name.replace("_", " ").title()
    return formatted


def build_feature_mapping(transformed_feature_names: List[str]) -> Dict[str, str]:
    """
    Build a mapping dictionary from transformed feature column names to human-readable labels.

    Args:
        transformed_feature_names: List of feature names from preprocessor.

    Returns:
        Dictionary of {transformed_name: human_readable_label}.
    """
    mapping = {}
    for feat in transformed_feature_names:
        mapping[feat] = format_human_readable_feature_name(feat)
    return mapping
