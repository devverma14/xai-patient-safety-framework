"""
Patient-level leakage prevention and dataset partitioning module for tabular laboratory data.
Ensures that all encounters for any given patient (identified by `patient_nbr`) are restricted
to a single partition (train, validation, or test), preventing cross-partition data leakage.
"""

from typing import Tuple, Dict, Any, Set
import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    group_col: str = "patient_nbr",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split tabular dataframe into train, validation, and test subsets grouped by patient identifier.

    Args:
        df: Input pandas DataFrame
        group_col: Column name identifying patients (default: 'patient_nbr')
        train_ratio: Proportion of unique patients for training
        val_ratio: Proportion of unique patients for validation
        test_ratio: Proportion of unique patients for testing
        random_state: Seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0. Got: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
        )

    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in dataframe columns.")

    unique_patients = df[group_col].unique()

    rng = np.random.RandomState(random_state)
    shuffled_patients = rng.permutation(unique_patients)

    n_patients = len(shuffled_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train : n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val :])

    train_df = df[df[group_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[group_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_patients)].copy().reset_index(drop=True)

    # Sanity check for patient leakage
    integrity = verify_split_integrity(train_df, val_df, test_df, group_col=group_col)
    if not integrity["passed"]:
        raise ValueError(f"Patient-level split failed integrity check: {integrity['overlap_details']}")

    return train_df, val_df, test_df


def verify_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "patient_nbr",
) -> Dict[str, Any]:
    """
    Verify that there is zero patient overlap between train, validation, and test sets.

    Args:
        train_df: Training partition
        val_df: Validation partition
        test_df: Test partition
        group_col: Column name identifying patients

    Returns:
        Dictionary with boolean 'passed' and overlap counts.
    """
    train_ids: Set[Any] = set(train_df[group_col].unique())
    val_ids: Set[Any] = set(val_df[group_col].unique())
    test_ids: Set[Any] = set(test_df[group_col].unique())

    train_val_overlap = len(train_ids.intersection(val_ids))
    train_test_overlap = len(train_ids.intersection(test_ids))
    val_test_overlap = len(val_ids.intersection(test_ids))

    total_overlap = train_val_overlap + train_test_overlap + val_test_overlap

    return {
        "passed": total_overlap == 0,
        "train_patients": len(train_ids),
        "val_patients": len(val_ids),
        "test_patients": len(test_ids),
        "train_encounters": len(train_df),
        "val_encounters": len(val_df),
        "test_encounters": len(test_df),
        "overlap_details": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
        },
    }
