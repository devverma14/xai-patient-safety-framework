"""
Lab Data Preprocessing Package.
Contains modules for data auditing, patient-level leakage prevention, tabular feature preprocessing,
and ICD-9 diagnostic feature engineering.
"""

from .audit import audit_lab_dataset
from .split import patient_level_split, verify_split_integrity
from .preprocess import LabPreprocessor, prepare_lab_splits
from .feature_engineering import map_icd9_code_to_category, apply_icd9_grouping

__all__ = [
    "audit_lab_dataset",
    "patient_level_split",
    "verify_split_integrity",
    "LabPreprocessor",
    "prepare_lab_splits",
    "map_icd9_code_to_category",
    "apply_icd9_grouping",
]
