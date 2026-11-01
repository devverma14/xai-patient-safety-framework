"""
Audit module for UCI Diabetes 130-US Hospitals laboratory tabular dataset.
Performs empirical data inspection, missing value analysis, zero-variance feature identification,
discharge disposition audit, patient encounter overlap check, and target candidate analysis.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def audit_lab_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Perform empirical audit of the laboratory report dataset.

    Args:
        csv_path: Path to diabetic_data.csv

    Returns:
        Dictionary containing audit metrics and statistics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Lab dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    total_rows = len(df)
    total_cols = len(df.columns)

    # Missing values analysis ('?' or NaN)
    missing_analysis: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        q_count = int((df[col] == '?').sum())
        null_count = int(df[col].isnull().sum())
        total_missing = q_count + null_count
        if total_missing > 0:
            missing_analysis[col] = {
                "missing_count": total_missing,
                "missing_pct": round((total_missing / total_rows) * 100, 2),
            }

    # Patient identifier & repeat encounter analysis
    unique_patients = int(df["patient_nbr"].nunique())
    total_encounters = len(df)
    repeat_encounters = total_encounters - unique_patients

    patient_encounter_counts = df["patient_nbr"].value_counts()
    max_encounters_per_patient = int(patient_encounter_counts.max())
    patients_with_multiple_visits = int((patient_encounter_counts > 1).sum())

    # Zero-variance / constant feature detection
    constant_features = [col for col in df.columns if df[col].nunique() <= 1]

    # Identifier columns
    identifier_cols = ["encounter_id", "patient_nbr"]

    # Target options analysis
    target_distribution = {}
    if "readmitted" in df.columns:
        raw_counts = df["readmitted"].value_counts().to_dict()
        raw_pcts = (df["readmitted"].value_counts(normalize=True) * 100).round(2).to_dict()

        target_distribution["raw"] = {
            "counts": raw_counts,
            "percentages": raw_pcts,
        }

        # Candidate Option 1: 30-Day Readmission (<30 vs NO or >30)
        df["target_30day"] = (df["readmitted"] == "<30").astype(int)
        target_distribution["option_1_30day"] = {
            "counts": df["target_30day"].value_counts().to_dict(),
            "percentages": (df["target_30day"].value_counts(normalize=True) * 100).round(2).to_dict(),
        }

        # Candidate Option 2: Generic Readmission (readmitted != NO vs NO)
        df["target_any"] = (df["readmitted"] != "NO").astype(int)
        target_distribution["option_2_any"] = {
            "counts": df["target_any"].value_counts().to_dict(),
            "percentages": (df["target_any"].value_counts(normalize=True) * 100).round(2).to_dict(),
        }

    # Discharge disposition audit (Check for expired / hospice codes: 11, 19, 20, 21)
    expired_disposition_ids = [11, 19, 20, 21]
    expired_count = int(df["discharge_disposition_id"].isin(expired_disposition_ids).sum())

    audit_results = {
        "file_path": csv_path,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "unique_patients": unique_patients,
        "repeat_encounters": repeat_encounters,
        "patients_with_multiple_visits": patients_with_multiple_visits,
        "max_encounters_per_patient": max_encounters_per_patient,
        "missing_analysis": missing_analysis,
        "constant_features": constant_features,
        "identifier_columns": identifier_cols,
        "target_distribution": target_distribution,
        "expired_encounters": {
            "disposition_ids": expired_disposition_ids,
            "count": expired_count,
            "pct": round((expired_count / total_rows) * 100, 2),
        },
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }

    return audit_results


if __name__ == "__main__":
    csv_file = os.path.join("datasets", "lab", "diabetic_data.csv")
    if os.path.exists(csv_file):
        results = audit_lab_dataset(csv_file)
        print("=== LAB DATASET AUDIT RESULTS ===")
        print(f"Total Rows: {results['total_rows']}, Total Cols: {results['total_columns']}")
        print(f"Unique Patients: {results['unique_patients']}, Repeat Visits: {results['repeat_encounters']}")
        print(f"Constant Features: {results['constant_features']}")
        print(f"Expired Disposition Encounters: {results['expired_encounters']}")
        print(f"Target Distribution: {results['target_distribution']}")
        print("\nMissing Values (> 10%):")
        for k, v in results["missing_analysis"].items():
            if v["missing_pct"] > 10.0:
                print(f"  {k}: {v['missing_count']} ({v['missing_pct']}%)")
