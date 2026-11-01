# Common XAI Evaluation Framework

The **Common XAI Evaluation Framework** (`xai/evaluation/`) provides a modality-aware, unified evaluation abstraction for all explainable AI (XAI) implementations across the V1 clinical application.

---

## 1. Overview & Core Principles

- **Zero Retraining / Zero Model Mutation**: Evaluates existing frozen model checkpoints and predictions without changing weights, re-running inference unnecessarily, or altering pre-existing artifacts.
- **Modality-Aware Evaluation**: Explicitly distinguishes between **Quantitative Feature Attribution** (tabular SHAP for Lab) and **Visual Spatial Attribution** (image Grad-CAM for Pneumonia, TB, Skin, Retina). SHAP and Grad-CAM are not evaluated identically.
- **Uncomputed Metric Integrity**: Metrics that have not been computed for a modality (e.g., Grad-CAM spatial resolution for tabular SHAP, or uncomputed perturbation drops) are explicitly marked as `"NOT_EVALUATED"`. Values are **never** filled with zeros or fabricated estimates.
- **Unified Machine-Readable Reporting**: Consolidates all modality XAI evaluations into [`evaluation/xai/common_xai_evaluation.json`](file:///E:/MProject/evaluation/xai/common_xai_evaluation.json).

---

## 2. Directory Structure

```text
xai/evaluation/
├── __init__.py           # Package exports
├── schemas.py            # Modality XAI schema definitions, Enums & global disclaimers
├── evaluator.py          # CommonXAIEvaluator master aggregation engine
├── fidelity.py           # SHAP fidelity (top-k vs bottom-k insertion/deletion AUC)
├── stability.py          # SHAP stability index (Spearman rank & cosine similarity)
├── robustness.py         # SHAP robustness sanity checks (feature shuffling)
└── README.md             # Architecture documentation
```

---

## 3. Schema & Modality Evaluation Matrix

| Modality | Task | Model | Explanation Method | Explanation Type | Evaluated Metrics | Uncomputed Metrics (`NOT_EVALUATED`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab** | Diabetes Risk | XGBoost | SHAP (TreeExplainer) | Quantitative Feature | Fidelity Primary Evidence (Top-5 drop: 0.0929 vs Bottom-5 drop: 0.0000; supplementary ratio: near-zero denominator), Stability Index (Spearman 0.9188), Robustness (Feature Shuffling) | Spatial Heatmap Range, Visual Masking Drop |
| **Pneumonia** | Pneumonia vs Normal | DenseNet121 | Grad-CAM | Visual Spatial | Heatmap Range (`[0,1]`), Spatial Activation Validity, Output Dimensions | SHAP Fidelity, SHAP Stability, Perturbation Masking Drop |
| **TB** | Tuberculosis vs Normal (Cross-Site) | EfficientNet-B0 | Grad-CAM | Visual Spatial | Heatmap Range (`[0,1]`), Spatial Activation Validity, Montgomery Zero-Shot | SHAP Fidelity, SHAP Stability, Perturbation Masking Drop |
| **Skin** | Skin Lesion Risk ($t=0.40$) | DenseNet121 | Grad-CAM | Visual Spatial | Heatmap Range (`[0,1]`), TP/TN/FP/FN Categories, Top-20% Perturbation Drop | SHAP Fidelity, SHAP Stability, SHAP Robustness |
| **Retina** | Binary Diabetic Retinopathy | EfficientNet-B0 | Grad-CAM | Visual Spatial | Heatmap Range (`[0,1]`), 16 TP/TN/FP/FN Categories, Controlled Perturbation Drop ($\Delta p = -0.1395$), Model SHA Verified (`4bf9260b...`) | SHAP Fidelity, SHAP Stability, SHAP Robustness, Causal Proof |

---

## 4. Disclaimers & Governance Constraints

1. **Correlation vs Causality**: Feature importance (SHAP) and spatial heatmaps (Grad-CAM) indicate focal feature correlations, **not** causal reasoning or pathological proof.
2. **Fidelity Ratio Interpretation**: The Lab SHAP fidelity ratio ($\approx 9.29 \times 10^6$) is a **supplementary diagnostic ratio** caused by a near-zero bottom-5 probability drop denominator ($+1e-8$ epsilon), **not** a literal claim of being "millions of times more faithful." The primary evidence of faithfulness remains the raw probability drop comparison (Top-5 drop = 0.0929 vs Bottom-5 drop = 0.0000).
3. **Coarse Spatial Resolution**: Grad-CAM heatmaps upsample $7 \times 7$ feature maps to $224 \times 224$ inputs, yielding coarse attribution regions.
4. **Patient Identifier Absence (APTOS)**: The APTOS metadata does not provide `patient_id` or `eye_id` identifiers; therefore patient-level leakage, including potential bilateral-eye dependence, cannot be completely excluded.
5. **No Autonomous Clinical Utility**: XAI attributions serve strictly as decision support tools and do not substitute for prospective clinical validation.
