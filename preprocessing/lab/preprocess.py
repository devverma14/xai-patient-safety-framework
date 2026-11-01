"""
Feature preprocessing pipeline for laboratory report tabular datasets.
Provides a scikit-learn compatible preprocessor supporting missing value imputation,
scaling, categorical encoding with unknown handling, zero-variance feature removal,
and feature name preservation for downstream XGBoost & SHAP attribution modules.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LabPreprocessor(BaseEstimator, TransformerMixin):
    """
    Reusable feature preprocessing pipeline for UCI Diabetes laboratory tabular data.
    Ensures zero data leakage by fitting transformers strictly on training data partitions
    and automatically excluding target columns ('readmitted', 'target_*') and identifiers.
    """

    def __init__(
        self,
        drop_identifiers: bool = True,
        drop_high_missing_threshold: float = 0.80,
        drop_constant_features: bool = True,
        target_col: Optional[str] = None,
    ):
        """
        Args:
            drop_identifiers: Whether to drop identifier columns ('encounter_id', 'patient_nbr').
            drop_high_missing_threshold: Proportion threshold for dropping columns with missing values (default 0.80 = 80%).
            drop_constant_features: Whether to drop constant/zero-variance features.
            target_col: Target column name if extracting y along with X.
        """
        self.drop_identifiers = drop_identifiers
        self.drop_high_missing_threshold = drop_high_missing_threshold
        self.drop_constant_features = drop_constant_features
        self.target_col = target_col

        self.dropped_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.transformed_feature_names_: List[str] = []

    def _clean_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace '?' string markers and empty spaces with np.nan."""
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object or df_clean[col].dtype == "string":
                df_clean[col] = df_clean[col].replace(["?", " ", "None"], np.nan)
        return df_clean

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "LabPreprocessor":
        """
        Fit preprocessing pipeline components on training DataFrame.

        Args:
            X: Input training DataFrame
            y: Ignored

        Returns:
            self
        """
        df = self._clean_missing(X)

        self.dropped_cols_ = []

        # Identifier columns
        if self.drop_identifiers:
            for id_col in ["encounter_id", "patient_nbr"]:
                if id_col in df.columns:
                    self.dropped_cols_.append(id_col)

        # Target columns (raw 'readmitted' AND any derived 'target_*' columns)
        target_candidates = ["readmitted", "target_30day", "target_any", "target_multi"]
        if self.target_col and self.target_col not in target_candidates:
            target_candidates.append(self.target_col)

        for t_col in target_candidates:
            if t_col in df.columns and t_col not in self.dropped_cols_:
                self.dropped_cols_.append(t_col)

        # High missing value columns (> threshold)
        for col in df.columns:
            if col not in self.dropped_cols_:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio >= self.drop_high_missing_threshold:
                    self.dropped_cols_.append(col)

        # Constant features (zero-variance)
        if self.drop_constant_features:
            for col in df.columns:
                if col not in self.dropped_cols_:
                    if df[col].nunique(dropna=True) <= 1:
                        self.dropped_cols_.append(col)

        # Remaining active columns
        remaining_cols = [c for c in df.columns if c not in self.dropped_cols_]

        # Classify numerical vs categorical
        self.num_cols_ = []
        self.cat_cols_ = []
        for col in remaining_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.num_cols_.append(col)
            else:
                self.cat_cols_.append(col)

        # Construct ColumnTransformer
        transformers = []
        if self.num_cols_:
            from sklearn.pipeline import Pipeline
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.num_cols_))

        if self.cat_cols_:
            from sklearn.pipeline import Pipeline
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, self.cat_cols_))

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.column_transformer_.fit(df)

        # Store feature names
        feature_names = []
        if self.num_cols_:
            feature_names.extend(self.num_cols_)
        if self.cat_cols_:
            cat_ohe = self.column_transformer_.named_transformers_["cat"].named_steps["ohe"]
            cat_names = cat_ohe.get_feature_names_out(self.cat_cols_).tolist()
            feature_names.extend(cat_names)

        self.transformed_feature_names_ = feature_names

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input DataFrame using fitted transformers.

        Args:
            X: DataFrame to transform

        Returns:
            Numpy array of preprocessed numeric features.
        """
        if self.column_transformer_ is None:
            raise RuntimeError("LabPreprocessor instance is not fitted yet. Call fit() first.")

        df = self._clean_missing(X)
        transformed_arr = self.column_transformer_.transform(df)
        return np.asarray(transformed_arr, dtype=np.float32)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return names of transformed features for SHAP attribution mapping."""
        return self.transformed_feature_names_


def prepare_lab_splits(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function performing patient-level splitting and fitting LabPreprocessor.

    Args:
        df: Input raw lab DataFrame
        target_col: Name of target column (e.g., 'target_30day' or 'readmitted')
        train_ratio: Train split ratio
        val_ratio: Val split ratio
        test_ratio: Test split ratio
        random_state: Seed

    Returns:
        Dictionary containing preprocessed X_train, y_train, X_val, y_val, X_test, y_test,
        preprocessor instance, and feature names.
    """
    from .split import patient_level_split

    train_df, val_df, test_df = patient_level_split(
        df,
        group_col="patient_nbr",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    preprocessor = LabPreprocessor(target_col=target_col)
    X_train_proc = preprocessor.fit_transform(train_df)
    X_val_proc = preprocessor.transform(val_df)
    X_test_proc = preprocessor.transform(test_df)

    return {
        "X_train": X_train_proc,
        "y_train": y_train,
        "X_val": X_val_proc,
        "y_val": y_val,
        "X_test": X_test_proc,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out(),
        "train_df_raw": train_df,
        "val_df_raw": val_df,
        "test_df_raw": test_df,
    }
