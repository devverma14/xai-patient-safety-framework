"""
Dataset-specific specifications, label mappings, and lesion-level grouping modules
for Chest X-Ray Pneumonia, Tuberculosis (Montgomery + Shenzhen), HAM10000 Skin Lesion, and APTOS 2019 Retina.
"""

from typing import Dict, Tuple, Any, Set, List, Optional
import pandas as pd
import numpy as np


class PneumoniaDataSpec:
    """Specification for Chest X-Ray Images (Pneumonia) dataset."""

    CLASSES = ["NORMAL", "PNEUMONIA"]
    CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
    IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
    NUM_CLASSES = 2
    EXPECTED_SPLITS = ["train", "val", "test"]


class TuberculosisDataSpec:
    """Specification for Combined Montgomery + Shenzhen Tuberculosis datasets."""

    CLASSES = ["Normal", "Tuberculosis"]
    CLASS_TO_IDX = {"Normal": 0, "Tuberculosis": 1, "TB": 1}
    IDX_TO_CLASS = {0: "Normal", 1: "Tuberculosis"}
    NUM_CLASSES = 2
    SUBSETS = ["montgomery", "shenzhen"]


class HAM10000DataSpec:
    """Specification for HAM10000 7-Class Skin Lesion dataset."""

    CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    CLASS_NAMES = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses & intraepithelial carcinoma",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }
    CLASS_TO_IDX = {cat: i for i, cat in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: cat for i, cat in enumerate(CLASSES)}
    NUM_CLASSES = 7


class APTOS2019DataSpec:
    """Specification for APTOS 2019 Diabetic Retinopathy dataset."""

    RAW_CLASSES = [0, 1, 2, 3, 4]
    RAW_CLASS_NAMES = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR",
    }
    BINARY_CLASSES = ["No Diabetic Retinopathy", "Diabetic Retinopathy Present"]
    BINARY_CLASS_TO_IDX = {"No Diabetic Retinopathy": 0, "Diabetic Retinopathy Present": 1}
    NUM_CLASSES_V1 = 2


def map_aptos_binary_labels(df: pd.DataFrame, diagnosis_col: str = "diagnosis") -> pd.DataFrame:
    """
    Map APTOS 2019 5-grade severity labels (0-4) to V1 binary DR labels.

    Mapping:
        Grade 0 -> 0 (No Diabetic Retinopathy)
        Grades 1, 2, 3, 4 -> 1 (Diabetic Retinopathy Present)

    Args:
        df: Metadata DataFrame containing 'diagnosis' column.
        diagnosis_col: Name of column containing 0-4 diagnosis codes.

    Returns:
        DataFrame with new column 'binary_label' (0 or 1) and 'binary_class_name'.
    """
    if diagnosis_col not in df.columns:
        raise KeyError(f"Diagnosis column '{diagnosis_col}' not found in DataFrame.")

    df_mapped = df.copy()
    df_mapped["binary_label"] = (df_mapped[diagnosis_col] > 0).astype(int)
    df_mapped["binary_class_name"] = df_mapped["binary_label"].map({
        0: "No Diabetic Retinopathy",
        1: "Diabetic Retinopathy Present",
    })
    return df_mapped


def parse_aptos_dataset(dataset_dir: str = "datasets/retina") -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Parse local APTOS 2019 Blindness Detection dataset directory.

    Args:
        dataset_dir: Path to directory containing APTOS 2019 files (e.g. train.csv and train_images/).

    Returns:
        Tuple of (DataFrame if dataset exists else None, audit metadata dictionary).
    """
    import os
    import glob

    path = os.path.abspath(dataset_dir)
    csv_path = os.path.join(path, "train.csv")
    if not os.path.exists(csv_path):
        csvs = glob.glob(os.path.join(path, "*.csv"))
        csv_path = csvs[0] if csvs else None

    has_metadata = csv_path is not None and os.path.exists(csv_path)

    if not has_metadata:
        audit_info = {
            "dataset_name": "APTOS 2019 Blindness Detection",
            "local_directory": path,
            "status": "REQUIRES_MANUAL_ACQUISITION",
            "exists_locally": False,
            "images_found": 0,
            "readable_images": 0,
            "corrupted_images": 0,
            "acquisition_blocker": "HTTP 403 Forbidden via Kaggle CLI due to unaccepted competition rules/terms.",
            "legitimate_acquisition_requirements": [
                "1. User must log into Kaggle.com.",
                "2. User must navigate to https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules.",
                "3. User must click 'I Understand and Accept' to accept the APTOS 2019 competition terms.",
                "4. User must configure ~/.kaggle/kaggle.json API key token or manually download train.csv and train_images/ into datasets/retina/.",
            ],
            "benchmark_reference_counts": {
                "published_total_train_images": 3662,
                "grade_0_no_dr": 1805,
                "grade_1_mild": 370,
                "grade_2_moderate": 999,
                "grade_3_severe": 193,
                "grade_4_proliferative": 295,
                "v1_binary_no_dr": 1805,
                "v1_binary_dr_present": 1857,
            },
            "grouping_limitation": "APTOS 2019 metadata does not provide patient_id or eye_id; split is image-level stratified.",
        }
        return None, audit_info

    df_meta = pd.read_csv(csv_path)
    df_mapped = map_aptos_binary_labels(df_meta)

    # Check for image paths
    img_dir = os.path.join(path, "train_images")
    if not os.path.exists(img_dir):
        img_dir = path

    image_paths = []
    for idx, row in df_mapped.iterrows():
        id_code = str(row["id_code"])
        # Check png, jpg, jpeg extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            fp = os.path.join(img_dir, id_code + ext)
            if os.path.exists(fp):
                image_paths.append(fp)
                break
        else:
            image_paths.append(None)

    df_mapped["file_path"] = image_paths
    readable = df_mapped["file_path"].notna().sum()

    audit_info = {
        "dataset_name": "APTOS 2019 Blindness Detection",
        "local_directory": path,
        "status": "VERIFIED_READY" if readable > 0 else "MISSING_IMAGES",
        "exists_locally": True,
        "metadata_rows": len(df_mapped),
        "images_found": readable,
        "readable_images": readable,
        "raw_class_distribution": df_mapped["diagnosis"].value_counts().to_dict(),
        "binary_class_distribution": df_mapped["binary_class_name"].value_counts().to_dict(),
    }

    return df_mapped, audit_info


def split_aptos_dataset(
    df: pd.DataFrame,
    diagnosis_col: str = "diagnosis",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform duplicate-hash isolated stratified split (70/15/15) for APTOS 2019 Retina dataset.

    Exact Duplicate Handling:
        Compute SHA-256 hash for each image. Group unique SHA-256 hashes and partition the hash
        groups into train/validation/test stratified by 5-grade diagnosis. This guarantees that
        no exact duplicate image content crosses train/validation/test partitions while preserving
        approximate 5-grade diagnosis stratification.

    Note on Data Leakage Limitation:
        APTOS 2019 metadata provides `id_code` per image but does NOT provide `patient_id` or `eye_id`.
        Therefore, grouping by patient is impossible without additional external metadata.
        Splitting is performed on unique image hashes stratified on 5-grade `diagnosis` (or `binary_label`).
        This limitation is explicitly documented: if non-identical bilateral images exist for the same subject,
        subtle patient-level leakage across splits cannot be ruled out.

    Args:
        df: APTOS 2019 DataFrame containing `diagnosis` and `binary_label`.
        diagnosis_col: Column for stratification.
        train_ratio: Training proportion (default 0.70).
        val_ratio: Validation proportion (default 0.15).
        test_ratio: Testing proportion (default 0.15).
        random_state: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    import os
    import hashlib

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    df_copy = df.copy()

    # Compute img_hash if file_path is available and img_hash is missing
    if "img_hash" not in df_copy.columns and "file_path" in df_copy.columns:
        hashes = []
        for fp in df_copy["file_path"]:
            if fp and os.path.exists(fp):
                with open(fp, "rb") as f:
                    hashes.append(hashlib.sha256(f.read()).hexdigest())
            else:
                hashes.append(None)
        df_copy["img_hash"] = hashes

    if "img_hash" in df_copy.columns and df_copy["img_hash"].notna().all():
        hash_df = df_copy.groupby("img_hash").agg({
            diagnosis_col: "first"
        }).reset_index()

        groups = hash_df.groupby(diagnosis_col)
        rng = np.random.RandomState(random_state)

        tr_hashes, va_hashes, te_hashes = [], [], []

        for _, group_df in groups:
            indices = group_df.index.values.copy()
            rng.shuffle(indices)
            n = len(indices)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)

            tr_idx = indices[:n_tr]
            va_idx = indices[n_tr : n_tr + n_va]
            te_idx = indices[n_tr + n_va :]

            tr_hashes.extend(group_df.loc[tr_idx, "img_hash"])
            va_hashes.extend(group_df.loc[va_idx, "img_hash"])
            te_hashes.extend(group_df.loc[te_idx, "img_hash"])

        tr_set, va_set, te_set = set(tr_hashes), set(va_hashes), set(te_hashes)

        train_df = df_copy[df_copy["img_hash"].isin(tr_set)].reset_index(drop=True)
        val_df = df_copy[df_copy["img_hash"].isin(va_set)].reset_index(drop=True)
        test_df = df_copy[df_copy["img_hash"].isin(te_set)].reset_index(drop=True)

        # Verify zero cross-partition duplicate hash overlap
        tr_h, va_h, te_h = set(train_df["img_hash"]), set(val_df["img_hash"]), set(test_df["img_hash"])
        overlap = len(tr_h.intersection(va_h)) + len(tr_h.intersection(te_h)) + len(va_h.intersection(te_h))
        if overlap > 0:
            raise ValueError(f"Duplicate-hash split failed: found {overlap} overlapping image hashes across splits.")

        return train_df, val_df, test_df
    else:
        # Fallback to row-level stratification if img_hash is unavailable
        groups = df_copy.groupby(diagnosis_col)
        train_dfs, val_dfs, test_dfs = [], [], []
        rng = np.random.RandomState(random_state)

        for _, group_df in groups:
            indices = group_df.index.values.copy()
            rng.shuffle(indices)
            n = len(indices)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)

            tr_idx = indices[:n_tr]
            va_idx = indices[n_tr : n_tr + n_va]
            te_idx = indices[n_tr + n_va :]

            train_dfs.append(df_copy.loc[tr_idx])
            val_dfs.append(df_copy.loc[va_idx])
            test_dfs.append(df_copy.loc[te_idx])

        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_df = pd.concat(val_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)

        return train_df, val_df, test_df



def split_ham10000_by_lesion(
    metadata_df: pd.DataFrame,
    lesion_id_col: str = "lesion_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split HAM10000 metadata DataFrame by `lesion_id` to prevent lesion-level data leakage.
    Photos of the exact same lesion will not cross train/validation/test partitions.

    Args:
        metadata_df: HAM10000_metadata DataFrame.
        lesion_id_col: Column identifying unique skin lesions.
        train_ratio: Ratio for training.
        val_ratio: Ratio for validation.
        test_ratio: Ratio for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0.")

    if lesion_id_col not in metadata_df.columns:
        raise KeyError(f"Lesion ID column '{lesion_id_col}' not found in metadata DataFrame.")

    unique_lesions = metadata_df[lesion_id_col].unique()

    rng = np.random.RandomState(random_state)
    shuffled_lesions = rng.permutation(unique_lesions)

    n_lesions = len(shuffled_lesions)
    n_train = int(n_lesions * train_ratio)
    n_val = int(n_lesions * val_ratio)

    train_lesions = set(shuffled_lesions[:n_train])
    val_lesions = set(shuffled_lesions[n_train : n_train + n_val])
    test_lesions = set(shuffled_lesions[n_train + n_val :])

    train_df = metadata_df[metadata_df[lesion_id_col].isin(train_lesions)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df[lesion_id_col].isin(val_lesions)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df[lesion_id_col].isin(test_lesions)].copy().reset_index(drop=True)

    # Verify zero lesion overlap
    train_ids = set(train_df[lesion_id_col].unique())
    val_ids = set(val_df[lesion_id_col].unique())
    test_ids = set(test_df[lesion_id_col].unique())

    overlap = len(train_ids.intersection(val_ids)) + len(train_ids.intersection(test_ids)) + len(val_ids.intersection(test_ids))
    if overlap > 0:
        raise ValueError(f"Lesion-level split failed: found {overlap} overlapping lesion IDs.")

    return train_df, val_df, test_df


def parse_ham10000_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Parse HAM10000 Skin Lesion dataset directory and metadata CSV.

    Args:
        dataset_dir: Path to `datasets/skin` directory.

    Returns:
        DataFrame with columns ['file_path', 'relative_path', 'lesion_id', 'image_id', 'dx', 'class_name', 'label_idx', 'label']
    """
    from pathlib import Path

    skin_path = Path(dataset_dir).resolve()
    csv_file = skin_path / "HAM10000_metadata.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found at: {csv_file}")

    df_meta = pd.read_csv(csv_file)
    records = []
    for _, row in df_meta.iterrows():
        img_id = str(row["image_id"]).strip()
        img_file = skin_path / f"{img_id}.jpg"
        if not img_file.exists():
            continue

        dx = str(row["dx"]).strip().lower()
        lbl_idx = HAM10000DataSpec.CLASS_TO_IDX.get(dx, 0)
        cls_name = HAM10000DataSpec.CLASS_NAMES.get(dx, dx)
        rel_path = img_file.relative_to(skin_path)

        rec = dict(row)
        rec.update({
            "file_path": str(img_file.resolve()),
            "relative_path": str(rel_path),
            "label_idx": int(lbl_idx),
            "label": int(lbl_idx),
            "class_name": cls_name,
        })
        records.append(rec)

    df_parsed = pd.DataFrame(records)
    if df_parsed.empty:
        raise ValueError("No valid HAM10000 images found in dataset directory.")

    return df_parsed


def parse_pneumonia_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Parse the Chest X-Ray Pneumonia dataset directory, extract labels and patient IDs.

    Args:
        dataset_dir: Path to `datasets/pneumonia` directory.

    Returns:
        DataFrame with columns ['file_path', 'relative_path', 'label_name', 'label_idx', 'patient_id', 'original_split']
    """
    import re
    from pathlib import Path

    p_dir = Path(dataset_dir).resolve()
    if not p_dir.exists():
        raise FileNotFoundError(f"Pneumonia dataset directory not found at: {p_dir}")

    clean_images = []
    rel_paths = set()
    for f in p_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".jpeg", ".jpg", ".png"]:
            if "__MACOSX" in f.parts or f.name.startswith("._") or f.name == ".DS_Store":
                continue
            parts = list(f.parts)
            split_idx = -1
            for i, p in enumerate(parts):
                if p.lower() in ["train", "val", "test"]:
                    split_idx = i
                    break
            if split_idx != -1:
                rel_path = Path(*parts[split_idx:])
                if rel_path not in rel_paths:
                    rel_paths.add(rel_path)
                    clean_images.append((f, rel_path))

    records = []
    for full_path, rel_path in clean_images:
        orig_split = rel_path.parts[0].lower()
        label_name = rel_path.parts[1].upper()
        if label_name not in PneumoniaDataSpec.CLASS_TO_IDX:
            continue
        label_idx = PneumoniaDataSpec.CLASS_TO_IDX[label_name]
        fname = rel_path.name

        if label_name == "PNEUMONIA":
            m = re.match(r"^(person\d+)_", fname, re.IGNORECASE)
            pid = m.group(1).lower() if m else f"pneumonia_unknown_{fname}"
        else:
            m1 = re.match(r"^(NORMAL\d*-IM-\d+)", fname, re.IGNORECASE)
            if m1:
                pid = m1.group(1).upper()
            else:
                m2 = re.match(r"^(IM-\d+)", fname, re.IGNORECASE)
                pid = m2.group(1).upper() if m2 else f"normal_unknown_{fname}"

        records.append({
            "file_path": str(full_path),
            "relative_path": str(rel_path),
            "label_name": label_name,
            "label_idx": label_idx,
            "patient_id": pid,
            "original_split": orig_split,
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid pneumonia images found in dataset directory.")
    return df


def split_pneumonia_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split Pneumonia DataFrame by `patient_id` with stratified class ratios to prevent patient-level data leakage.

    Args:
        df: DataFrame produced by `parse_pneumonia_dataset`.
        train_ratio: Ratio for training.
        val_ratio: Ratio for validation.
        test_ratio: Ratio for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0.")

    if "patient_id" not in df.columns or "label_idx" not in df.columns:
        raise KeyError("DataFrame must contain 'patient_id' and 'label_idx' columns.")

    patient_df = df.groupby("patient_id").agg({"label_idx": "first"}).reset_index()

    normal_patients = patient_df[patient_df["label_idx"] == 0]["patient_id"].values
    pneumonia_patients = patient_df[patient_df["label_idx"] == 1]["patient_id"].values

    rng = np.random.RandomState(random_state)
    rng.shuffle(normal_patients)
    rng.shuffle(pneumonia_patients)

    def partition(arr):
        n = len(arr)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        return set(arr[:n_tr]), set(arr[n_tr : n_tr + n_va]), set(arr[n_tr + n_va :])

    norm_tr, norm_va, norm_te = partition(normal_patients)
    pneu_tr, pneu_va, pneu_te = partition(pneumonia_patients)

    train_pids = norm_tr.union(pneu_tr)
    val_pids = norm_va.union(pneu_va)
    test_pids = norm_te.union(pneu_te)

    train_df = df[df["patient_id"].isin(train_pids)].copy().reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].copy().reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pids)].copy().reset_index(drop=True)

    # Verify zero patient overlap
    train_ids = set(train_df["patient_id"].unique())
    val_ids = set(val_df["patient_id"].unique())
    test_ids = set(test_df["patient_id"].unique())

    overlap = (
        len(train_ids.intersection(val_ids))
        + len(train_ids.intersection(test_ids))
        + len(val_ids.intersection(test_ids))
    )
    if overlap > 0:
        raise ValueError(f"Patient-level split failed: found {overlap} overlapping patient IDs.")

    return train_df, val_df, test_df


def parse_tb_dataset(dataset_dir: str, site: Optional[str] = None) -> pd.DataFrame:
    """
    Parse Combined Montgomery + Shenzhen Tuberculosis datasets directory.

    Args:
        dataset_dir: Path to `datasets/tb` directory.
        site: Optional site filter ('montgomery' or 'shenzhen'). If None, parses both.

    Returns:
        DataFrame with columns ['file_path', 'relative_path', 'site', 'file_name', 'patient_id', 'label_name', 'label_idx']
    """
    import re
    from pathlib import Path

    tb_path = Path(dataset_dir).resolve()
    if not tb_path.exists():
        raise FileNotFoundError(f"Tuberculosis dataset directory not found at: {tb_path}")

    sites = ["montgomery", "shenzhen"] if site is None else [site.lower()]

    records = []
    for s in sites:
        site_dir = tb_path / s
        if not site_dir.exists():
            continue

        img_files = list((site_dir / "images").rglob("*.png"))
        for f in img_files:
            fname = f.name
            m = re.match(r"^(MCUCXR_\d+|CHNCXR_\d+)_([01])\.png$", fname, re.IGNORECASE)
            if not m:
                continue

            pid = m.group(1).upper()
            lbl_idx = int(m.group(2))
            lbl_name = TuberculosisDataSpec.IDX_TO_CLASS[lbl_idx]
            rel_path = f.relative_to(tb_path)

            records.append({
                "file_path": str(f.resolve()),
                "relative_path": str(rel_path),
                "site": s,
                "file_name": fname,
                "patient_id": pid,
                "label_name": lbl_name,
                "label_idx": lbl_idx,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid tuberculosis images found in dataset directory.")

    df["label"] = df["label_idx"]
    return df


def split_tb_dataset(
    df: pd.DataFrame,
    strategy: str = "cross_site",
    val_ratio: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split Tuberculosis dataset using cross-site zero-shot or site-stratified group split strategy.

    Args:
        df: DataFrame produced by `parse_tb_dataset`.
        strategy: 'cross_site' (Train on Shenzhen 80/20 val, Test zero-shot on Montgomery)
                  or 'site_stratified' (Pooled 70/15/15 stratified by site and label).
        val_ratio: Validation ratio for Shenzhen training split in cross_site mode.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if strategy == "cross_site":
        shenzhen_df = df[df["site"] == "shenzhen"].copy().reset_index(drop=True)
        montgomery_df = df[df["site"] == "montgomery"].copy().reset_index(drop=True)

        if shenzhen_df.empty or montgomery_df.empty:
            raise ValueError("Cross-site split requires both 'shenzhen' and 'montgomery' sites in DataFrame.")

        # Split Shenzhen into Train and Validation
        patient_df = shenzhen_df.groupby("patient_id").agg({"label_idx": "first"}).reset_index()
        norm_pids = patient_df[patient_df["label_idx"] == 0]["patient_id"].values
        tb_pids = patient_df[patient_df["label_idx"] == 1]["patient_id"].values

        rng = np.random.RandomState(random_state)
        rng.shuffle(norm_pids)
        rng.shuffle(tb_pids)

        n_norm_val = int(len(norm_pids) * val_ratio)
        n_tb_val = int(len(tb_pids) * val_ratio)

        val_norm = set(norm_pids[:n_norm_val])
        train_norm = set(norm_pids[n_norm_val:])

        val_tb = set(tb_pids[:n_tb_val])
        train_tb = set(tb_pids[n_tb_val:])

        train_pids = train_norm.union(train_tb)
        val_pids = val_norm.union(val_tb)

        train_df = shenzhen_df[shenzhen_df["patient_id"].isin(train_pids)].copy().reset_index(drop=True)
        val_df = shenzhen_df[shenzhen_df["patient_id"].isin(val_pids)].copy().reset_index(drop=True)
        test_df = montgomery_df.copy().reset_index(drop=True)

    elif strategy == "site_stratified":
        # Pooled 70/15/15 stratified by both site and label_idx
        groups = df.groupby(["site", "label_idx"])
        train_dfs, val_dfs, test_dfs = [], [], []

        rng = np.random.RandomState(random_state)
        for _, group_df in groups:
            pids = group_df["patient_id"].unique()
            rng.shuffle(pids)
            n = len(pids)
            n_tr = int(n * 0.70)
            n_va = int(n * 0.15)

            tr_pids = set(pids[:n_tr])
            va_pids = set(pids[n_tr : n_tr + n_va])
            te_pids = set(pids[n_tr + n_va :])

            train_dfs.append(group_df[group_df["patient_id"].isin(tr_pids)])
            val_dfs.append(group_df[group_df["patient_id"].isin(va_pids)])
            test_dfs.append(group_df[group_df["patient_id"].isin(te_pids)])

        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_df = pd.concat(val_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown split strategy: '{strategy}'. Allowed: 'cross_site', 'site_stratified'.")

    # Verify zero patient overlap
    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())

    overlap = (
        len(train_pids.intersection(val_pids))
        + len(train_pids.intersection(test_pids))
        + len(val_pids.intersection(test_pids))
    )
    if overlap > 0:
        raise ValueError(f"Patient-level split failed: found {overlap} overlapping patient IDs.")

    return train_df, val_df, test_df

