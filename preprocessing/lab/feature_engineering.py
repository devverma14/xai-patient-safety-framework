"""
Feature engineering and ICD-9 diagnosis grouping module for UCI Diabetes laboratory dataset.
Maps raw, high-cardinality ICD-9 diagnosis codes (diag_1, diag_2, diag_3) to standard
major clinical diagnostic categories to reduce feature noise and dimensionality.
"""

import re
import pandas as pd


def map_icd9_code_to_category(code: str) -> str:
    """
    Map an individual ICD-9 string code to one of 9 major clinical diagnostic categories.

    Categories:
        - Diabetes (ICD-9 250.xx)
        - Circulatory (390–459, 785)
        - Respiratory (460–519, 786)
        - Digestive (520–579, 787)
        - Genitourinary (580–629, 788)
        - Musculoskeletal (710–739)
        - Neoplasms (140–239)
        - Injury (800–999)
        - Other (all other numeric or V/E codes)
    """
    if pd.isna(code) or code == "?" or code == "None" or code == "":
        return "Missing"

    code_str = str(code).strip()

    # Diabetes check
    if code_str.startswith("250"):
        return "Diabetes"

    # External causes (E) or Supplemental (V)
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Other"

    # Extract leading numeric part
    match = re.match(r"^(\d+)", code_str)
    if not match:
        return "Other"

    num = int(match.group(1))

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 800 <= num <= 999:
        return "Injury"
    else:
        return "Other"


def apply_icd9_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ICD-9 category mapping to diag_1, diag_2, and diag_3 columns in DataFrame.

    Args:
        df: Input raw lab DataFrame

    Returns:
        DataFrame with diag_1, diag_2, diag_3 transformed to grouped categories.
    """
    df_grouped = df.copy()
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df_grouped.columns:
            df_grouped[diag_col] = df_grouped[diag_col].astype(str).apply(map_icd9_code_to_category)
    return df_grouped
