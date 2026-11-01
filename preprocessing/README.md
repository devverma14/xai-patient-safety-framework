# Data Preprocessing Subsystem

This package provides clean, reproducible, leakage-aware preprocessing pipelines for the project datasets as established in Phase 3.

## Package Architecture

```
preprocessing/
├── __init__.py
├── lab/
│   ├── __init__.py
│   ├── audit.py          # Empirical data audit & target option analysis
│   ├── preprocess.py     # LabPreprocessor transformer (imputation, scaling, encoding)
│   └── split.py          # Patient-level split (patient_nbr grouping)
├── images/
│   ├── __init__.py
│   ├── common.py         # RGB conversion, image loading, ImageNet normalization
│   ├── transforms.py     # Train vs validation/test image transformation pipelines
│   ├── dataset_specific.py # Dataset specifications (Pneumonia, TB, HAM10000, APTOS 2019)
│   └── validation.py     # Image directory integrity check & corrupt file detector
└── README.md
```

## Key Preprocessing Principles

1. **Patient-Level Leakage Prevention (Lab Data)**:
   - Encounters are grouped by `patient_nbr`.
   - All encounters belonging to the same patient are strictly assigned to a single partition (train, validation, or test).

2. **Lesion-Level Grouping (HAM10000 Skin Data)**:
   - Images are grouped by `lesion_id` using `split_ham10000_by_lesion`.
   - Prevents multiple photos of the same skin lesion from appearing across training and evaluation splits.

3. **Strict Separation of Training Augmentation**:
   - Training transforms include random horizontal flips, rotations, and color jittering.
   - Validation and test transforms strictly perform deterministic resizing, tensor conversion, and ImageNet normalization.

4. **Target Preservation**:
   - The lab prediction target is left un-finalized in Phase 3. `LabPreprocessor` and `prepare_lab_splits` take `target_col` dynamically.

5. **APTOS 2019 DR Label Mapping**:
   - APTOS 5-grade severity (0 to 4) is mapped to binary classification: `Grade 0` (No DR) vs `Grades 1-4` (Diabetic Retinopathy Present).
