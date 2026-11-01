# Development of a Framework Using Explainable Artificial Intelligence for Patient Safety

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-16.3.8-black?logo=next.js)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19.2.8-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?logo=tailwindcss)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## 1. Overview

This repository contains the complete implementation of the **Modality-Aware Explainable Artificial Intelligence (XAI) Framework for Patient Safety**. Designed as an investigational clinical decision-support research prototype (second-reader platform), the system addresses the "black-box" dilemma in medical machine learning by coupling deep neural inference with deterministic visual attribution and defensive quality gates.

The platform spans **five diagnostic modules**:
1. **Clinical Laboratory Report Understanding:** Deterministic OCR (Tesseract 5.x) and regex extraction engine evaluating biological reference intervals without statistical guesswork.
2. **Pediatric Pneumonia Detection:** DenseNet-121 on frontal chest radiographs with layer-targeted Grad-CAM visual attribution.
3. **Pulmonary Tuberculosis Screening:** DenseNet-121 on multi-site chest radiographs with apical parenchymal cavity attribution.
4. **Dermoscopic Skin Lesion Triage:** DenseNet-121 production checkpoint (canonical benchmark: **74.29% accuracy** at $t=0.50$; deployed API triage setting: **68.37% accuracy, 94.95% sensitivity** at $t=0.40$), alongside separate offline experimental research models (**E1 fine-tuned: 81.83% accuracy**; **E7 ensemble: 79.00% accuracy, 0.9206 AUROC**).
5. **Diabetic Retinopathy Screening:** Compound-scaled EfficientNet-B0 on digital fundus photography with macular microvascular attribution.

---

## 2. Problem Statement & Clinical Motivation

Deep learning models achieve diagnostic accuracy comparable to medical specialists on benchmark datasets. However, deploying them in safety-critical clinical environments introduces severe patient safety risks:
- **The Black-Box Failure Mode:** Standard convolutional networks produce scalar probability scores without explaining which anatomical regions drove the decision. A model can achieve 95% accuracy by learning confounding hospital markers, radiographic orientation tags, or imaging artifacts rather than genuine pathology.
- **Garbage-In, Catastrophe-Out:** Conventional inference APIs silently process out-of-distribution, severely blurred, inverted, or non-medical images, generating high-confidence false predictions on uninterpretable data.
- **Asymmetric Risk in Oncology:** Standard 0.50 decision thresholds optimize overall accuracy at the expense of false negatives. In an investigational triage setting, false negatives carry high risk by potentially delaying clinical review of suspicious lesions, whereas false positives increase clinician review burden or follow-up examination.

This project implements a multi-tier defensive architecture: **Defensive Quality Gates** intercept invalid inputs before inference, **Layer-Specific Grad-CAM** provides visual attribution for class-discriminative regions, and **Sensitivity-Prioritized Thresholding** provides an alternative research operating point that trades specificity for higher sensitivity.

---

## 3. Core Objectives

1. Develop a unified, multi-modal research decision-support framework covering radiological, dermatological, retinal, and biochemical data modalities.
2. Integrate mathematical Gradient-Weighted Class Activation Mapping (Grad-CAM) directly into the inference lifecycle, generating synchronized raw, heatmap, and overlay views.
3. Establish defensive pre-inference quality checks (MIME validation, dimension gating, and Laplacian focus blur estimation).
4. Build a deterministic laboratory report understanding module that extracts biomarkers and compares them against physiological reference ranges with zero statistical imputation.
5. Empirically validate explainability faithfulness using systematic salient pixel deletion experiments.

---

## 4. Key Architectural Features

- **Decoupled Four-Tier Architecture:** Complete separation between the Presentation Tier (Next.js 16/React 19), Application API Gateway (FastAPI/Pydantic), Inference Tier (PyTorch/torchvision), and Storage/Audit Tier.
- **Defensive Input Quality Gates:** Rejects unreadable inputs using Laplacian focus variance ($\sigma^2 < 100.0$) and MIME type sniffing before model execution.
- **Three-View Synchronized Explainability:** Presents the user with the original raw image, isolated jet saliency map, and interactive alpha-blended attribution overlay with dynamic opacity controls.
- **Zero Imputation Policy:** If a laboratory document lacks an analyte or unit, the system triggers a `review_required = True` flag rather than guessing synthetic values.
- **In-Memory Ephemeral Processing:** Patient imagery is converted to memory buffers (`io.BytesIO()`) and processed without persistent local caching, reducing persistent local storage of uploaded patient imagery during processing.

---

## 5. Implemented Diagnostic Modules

### Module 1: Clinical Laboratory Report Understanding
- **Methodology:** Hybrid Tesseract 5.x OCR engine + compiled regular expression parser + deterministic reference-interval validator.
- **Machine Learning Scope:** **Exactly 0.0% ML.** (Does NOT predict clinical diseases; performs syntactic extraction and biological range comparison).
- **Supported Analytes:** 34 standard clinical analytes (Hemoglobin, Fasting Glucose, HbA1c, Serum Creatinine, Platelet Count, BUN, Bilirubin, etc.) and 11 derived diagnostic ratios (BUN/Creatinine, Mentzer Index, A/G Ratio, etc.).
- **Benchmark Performance:** 100.0% field extraction match across 8 multi-format synthetic panels (44 evaluated parameter instances).

### Module 2: Pediatric Chest X-Ray Pneumonia Detection
- **Architecture:** DenseNet-121 initialized with ImageNet weights, adapted with a single-logit classification head.
- **Target Grad-CAM Hook:** `features.denseblock4.denselayer16.conv2` (final 3x3 convolutional layer of DenseBlock 4).
- **Performance (910 Kermany Test CXRs):** **90.66% Accuracy**, **90.15% Sensitivity**, **92.08% Specificity**, **0.9621 AUROC**, **0.9866 PR-AUC**.

### Module 3: Pulmonary Tuberculosis CXR Screening
- **Architecture:** DenseNet-121 evaluated on combined Shenzhen (China) and Montgomery (USA) multi-site cohorts.
- **Target Grad-CAM Hook:** `features.denseblock4.denselayer16.conv2`.
- **Performance (120 Multi-Site Test CXRs):** **85.83% Accuracy**, **81.36% Sensitivity**, **90.16% Specificity**, **0.9228 AUROC**, **0.9353 PR-AUC**.

### Module 4: Dermoscopic Skin Lesion Risk Triage
- **Architecture & Dataset Split:** DenseNet-121 initialized with ImageNet weights, trained with class-weighted cross-entropy on 7,015 images from HAM10000 (1,514 validation). Evaluated on an independent, strictly isolated held-out test cohort of 1,486 images (277 malignant, 1,209 benign) with zero cross-split lesion_id overlap (lesion-level grouping across 7,470 unique lesions, ensuring images of the same lesion never cross splits).
- **Production Checkpoint (models/skin/binary/best_skin_binary_model.pt):**
  - **Canonical Checkpoint Benchmark ($t=0.50$):** **74.29% Accuracy**, **81.00% Balanced Accuracy**, **91.70% Sensitivity**, **70.31% Specificity**, **0.8806 AUROC**, **0.5975 PR-AUC** (TN=850, FP=359, FN=23, TP=254). Represents the standard, uncalibrated baseline classification performance of the trained neural network weights.
  - **Deployed API Triage Configuration ($t=0.40$):** **68.37% Accuracy**, **78.61% Balanced Accuracy**, **94.95% Sensitivity**, **62.28% Specificity**, **0.8806 AUROC**, **0.5975 PR-AUC** (TN=753, FP=456, FN=14, TP=263). The deployed FastAPI endpoint (`/api/v1/skin/predict/ham10000`) and React frontend are actively configured with this sensitivity-prioritized triage threshold ($t=0.40$) to minimize missed potential malignancies in an investigational second-reader triage context (reducing false negatives from 23 down to 14, at the trade-off of additional false alarms).
- **Controlled Experimental Evaluation (E0 — Separate Bicubic Preprocessing):**
  - In a separate controlled evaluation harness using Pillow Bicubic in-memory image downsampling (rather than the TorchVision Bilinear interpolation used by the production API), the identical checkpoint achieves **75.17% Accuracy**, **90.25% Sensitivity**, **71.71% Specificity**, and **0.8787 AUROC** (TN=867, FP=342, FN=27, TP=250) at $t=0.40$, and **81.29% Accuracy** at $t=0.50$. This 6.80 percentage-point accuracy difference at $t=0.40$ coincides with differences between the evaluation preprocessing pipelines (e.g., Pillow Bicubic in-memory caching vs TorchVision Bilinear loading); further controlled testing is required to isolate the exact contributing causes. E0 is documented strictly as an offline controlled experimental evaluation.
- **Research Ablation & Ensemble Experiments (Not Deployed in Production API):**
  - **Experiment E1 (Partial Fine-Tuning, $t=0.50$):** Fine-tuning `denseblock4` with $\text{lr}=10^{-5}$ achieves **81.83% Accuracy**, **84.84% Sensitivity**, **81.14% Specificity**, **0.9127 AUROC**, and **0.7242 PR-AUC**, reducing false positives by 50% (from 456 down to 228) compared to the deployed baseline triage configuration at $t=0.40$ (not a matched-threshold comparison; against the standard $t=0.50$ baseline of 359 FP, E1 reduces false positives by 36.5%).
  - **Experiment E7 (Soft-Voting Ensemble, $t=0.50$):** Weighted ensemble ($0.55 \times \text{E1} + 0.45 \times \text{E3}$) achieves **79.00% Accuracy**, **90.25% Sensitivity**, **76.43% Specificity**, and project-high **0.9206 AUROC**; or **74.16% Accuracy** and **94.22% Sensitivity** at $t=0.42$. Both E1 and E7 are offline research candidates and are not loaded by the deployed API.
### Module 5: Diabetic Retinopathy Fundus Screening
- **Architecture:** EfficientNet-B0 utilizing compound scaling ($d=1.0, w=1.0, r=224$).
- **Target Grad-CAM Hook:** `features.8.0` (final pointwise convolutional layer).
- **Performance (547 APTOS 2019 Test Fundus Photos):** **97.99% Accuracy**, **98.55% Sensitivity**, **97.42% Specificity**, **0.9958 AUROC**, **0.9958 PR-AUC**.

---

## 6. Technology Stack

- **Backend Framework:** FastAPI 0.109.0, Uvicorn 0.27.0, Pydantic v2, Pydantic-Settings
- **Deep Learning & Inference:** PyTorch 2.2.0+, TorchVision 0.17.0+, Timm 0.9.12
- **Document OCR:** Tesseract 5.x, PyTesseract 0.3.10, PyPDF 4.0+
- **Image Processing & Vision:** OpenCV (Headless) 4.9.0, Pillow 10.2.0, NumPy, SciPy
- **Data Science & Evaluation:** Pandas 2.2.0, Scikit-Learn 1.4.0, Matplotlib, Seaborn
- **Frontend Framework:** Next.js 16.3.8 (App Router), React 19.2.8, TypeScript 5.0
- **UI & Styling:** Tailwind CSS v4, Lucide React Icons
- **Testing & Quality Assurance:** Pytest 8.0, Pytest-Cov, HTTPX

---

## 7. System Architecture

```
+-----------------------------------------------------------------------------------+
|                           PRESENTATION TIER (Next.js 16 / React 19)               |
|   Modality Selector | Dropzone Upload | Three-View Grad-CAM | Lab Table | Sliders  |
+-----------------------------------------+-----------------------------------------+
                                          | REST API (HTTP Multipart / JSON)
                                          v
+-----------------------------------------------------------------------------------+
|                           APPLICATION TIER (FastAPI Gateway)                     |
|  /api/v1/lab (OCR) | /api/v1/xray (CXR) | /api/v1/skin (Skin) | /api/v1/retina (DR) |
|         +---------------------------------------------------------------+         |
|         | DEFENSIVE QUALITY GATES: MIME Check | Laplacian Blur (Var<100)|         |
+---------+-------------------------------+-------------------------------+---------+
                                          | In-Memory Normalized Tensors (224x224)
                                          v
+-----------------------------------------------------------------------------------+
|                           INFERENCE & EXPLAINABILITY TIER                         |
|  DenseNet-121 (Pneumonia/TB) | DenseNet-121 (Skin) | EfficientNet-B0 (Retina)     |
|         |                                 |                                 |     |
|         +-------> Forward Pass ---------> Classification Probabilities     |     |
|         |                                                                   |     |
|         +-------> Target Conv Layer ----> Backward Gradient Computation     |     |
|                                           |                                       |
|                                           v                                       |
|                            Grad-CAM Saliency Weighting                            |
|                            L_c = ReLU(Sum_k alpha_k^c * A^k)                      |
|                                           |                                       |
|                                           v                                       |
|                            Jet Colormap & Alpha-Blended Overlay                   |
+-----------------------------------------------------------------------------------+
```

---

## 8. Explainable AI & Grad-CAM Formulation

Grad-CAM calculates the importance of feature map $k$ for target class $c$ by computing the global average pooling of gradients:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

where $Y^c$ is the pre-softmax model score for class $c$, $A_{i,j}^k$ is the activation at spatial coordinate $(i,j)$ of feature map $k$, and $Z = U \times V$ is the spatial dimensions.

The coarse class-discriminative saliency map is formed via a rectified linear combination:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### Implementation Hooks in Codebase:
- **DenseNet-121:** Registered on `features.denseblock4.denselayer16.conv2`.
- **EfficientNet-B0:** Registered on `features.8.0`.

### Quantitative Faithfulness Validation (Pixel Deletion Test):
When the top-20% most salient pixels identified by Grad-CAM are masked across 20 verified true positive cases, target model confidence degrades significantly:
- **Tuberculosis CXR:** $-61.16\%$ drop ($0.824 \rightarrow 0.320$)
- **Skin Lesion:** $-49.81\%$ drop ($0.781 \rightarrow 0.392$)
- **Diabetic Retinopathy:** $-28.43\%$ drop ($0.985 \rightarrow 0.705$)
- **Pneumonia CXR:** $-14.27\%$ drop ($0.897 \rightarrow 0.769$)

These pixel-deletion results provide an empirical measure of model sensitivity to regions highlighted by Grad-CAM. They do not establish causal pathological localization.

---

## 9. Defensive Input Quality Gates

To prevent uninterpretable images from generating misleading predictions, the platform enforces three mandatory pre-inference gates (`backend/app/services/validation_service.py`):

1. **Magic Bytes MIME Validation:** Inspects raw file header bytes to confirm genuine JPEG, PNG, or PDF formats; rejects non-medical payloads (.exe, .zip, .sh).
2. **Dimension & Aspect Ratio Verification:** Ensures minimum pixel dimensions ($224 \times 224$) and valid image channels.
3. **Laplacian Focus Blur Estimation:** Computes the variance of the 2D Laplacian operator over the grayscale input image $I$:
   $$\text{Blur Score} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu_{\nabla^2 I} \right)^2$$
   Images with $\text{Blur Score} < 100.0$ are automatically intercepted with HTTP 400 Bad Request and an explicit request to recapture.

---

## 10. Repository Directory Structure

```
.
├── backend/                        # FastAPI Backend Application
│   └── app/
│       ├── main.py                 # Application entrypoint & middleware
│       ├── api/                    # Route handlers & Pydantic schemas
│       ├── core/                   # Centralized configuration & exceptions
│       └── services/               # Inference, Grad-CAM, & validation services
├── frontend/                       # Next.js 16 / React 19 Frontend
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages
│   │   ├── components/             # Modality viewers, Dropzone, 3-View Grad-CAM
│   │   └── lib/                    # API client & TypeScript interfaces
│   ├── package.json                # Frontend package manifest
│   └── tsconfig.json               # TypeScript configuration
├── ocr/                            # Deterministic Laboratory OCR Engine
│   ├── extraction.py               # Regex analyte parsing
│   ├── validation.py               # Biological reference-interval matching
│   ├── clinical_summary.py         # Patient-friendly interpretations
│   └── pdf.py                      # Multi-format document ingestion
├── preprocessing/                  # Data Loading & Image Transformation
│   └── images/                     # Transforms, normalization, and quality checks
├── xai/                            # Explainable AI Algorithms & Validation
│   ├── gradcam/                    # PyTorch Grad-CAM engine & heatmap generation
│   └── evaluation/                 # Saliency faithfulness & deletion tests
├── models/                         # Production Model Checkpoints & Metadata
│   ├── pneumonia/best_pneumonia_model.pt
│   ├── tb/best_tb_model.pt
│   ├── skin/binary/best_skin_binary_model.pt
│   └── retina/best_retina_model.pt
├── evaluation/                     # Final Empirical Audit Logs & Plots
│   └── final_audit/                # ROC curves, PR curves, CMs, and Grad-CAM examples
├── experiments/                    # Controlled Research Experiments Records
│   └── skin/                       # E0-E7 comparison summaries and training curves
├── tests/                          # Automated Pytest Suite & Fixtures
│   ├── sample_reports/             # Synthetic laboratory test fixtures
│   └── test_backend_api.py         # End-to-end integration tests
├── docs/                           # Architectural Blueprints & Figures
├── requirements.txt                # Pinned Python dependencies
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Comprehensive multi-stack ignore rules
├── .gitattributes                  # Git LFS configuration for model checkpoints
└── LICENSE                         # MIT License with clinical disclaimer
```

---

## 11. Installation & Environment Setup

### Prerequisites:
- Python 3.10 to 3.13
- Node.js 18+ and pnpm (or npm)
- Tesseract OCR 5.x installed on the host system

### 1. Clone the Repository:
```bash
git clone https://github.com/devverma14/xai-patient-safety-framework.git
cd xai-patient-safety-framework
```

### 2. Configure Python Virtual Environment & Dependencies:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables:
```bash
cp .env.example .env
# Edit .env if custom Tesseract paths or ports are required
```

### 4. Install Frontend Dependencies:
```bash
cd frontend
npm install
cd ..
```

---

## 12. Execution Instructions

### Running the FastAPI Backend:
From the project root directory with virtual environment activated:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend service will be available at `http://localhost:8000`.  
Interactive Swagger API documentation: `http://localhost:8000/docs`.

### Running the Next.js Frontend:
In a separate terminal window:
```bash
cd frontend
npm run dev
```
The web interface will be accessible in your web browser at `http://localhost:3000`.

---

## 13. REST API Documentation & Endpoints

| HTTP Method | Endpoint Route | Description & Parameters | Response Payload |
|---|---|---|---|
| `GET` | `/` | System health check and metadata. | System status and supported modalities. |
| `POST` | `/api/v1/lab/analyze` | Ingests PDF or image laboratory reports for deterministic OCR. | Structured analytes, units, flags, and review status. |
| `POST` | `/api/v1/xray/predict/pneumonia` | Evaluates frontal CXR for pediatric pneumonia + Grad-CAM. | Prediction badge, confidence, and Base64 Grad-CAM overlay. |
| `POST` | `/api/v1/xray/predict/tb` | Evaluates frontal CXR for pulmonary tuberculosis + Grad-CAM. | Prediction badge, confidence, and Base64 Grad-CAM overlay. |
| `POST` | `/api/v1/skin/predict` | Triage dermoscopic skin lesion ($t=0.40$) + Grad-CAM. | Risk tier, confidence, threshold tag, and attribution overlay. |
| `POST` | `/api/v1/retina/predict` | Screens digital fundus photography for referable DR + Grad-CAM. | Prediction, confidence score, and macular attribution map. |

---

## 14. Testing & Verification Suite

Execute the complete automated test suite:
```bash
pytest tests/ -v
```
To run end-to-end integration tests:
```bash
pytest tests/test_backend_api.py -v
```

---

## 15. Verified Empirical Results

### Primary Multi-Modal Benchmark Performance:
| Diagnostic Task | Model Architecture & Configuration | Held-Out Test Cohort | Accuracy | Sensitivity (Recall) | Specificity | AUROC | PR-AUC | Primary Confusion Matrix | Operating Context |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **Pneumonia (CXR)** | DenseNet-121 ($t=0.50$) | 910 images | **90.66%** | **90.15%** | **92.08%** | **0.9621** | **0.9866** | TP=604, TN=221, FP=19, FN=66 | Standard decision threshold ($t=0.50$) |
| **Tuberculosis (CXR)** | DenseNet-121 ($t=0.50$) | 120 images | **85.83%** | **81.36%** | **90.16%** | **0.9228** | **0.9353** | TP=48, TN=55, FP=6, FN=11 | Site-stratified test split ($t=0.50$) |
| **Skin Lesion (Canonical Benchmark)** | DenseNet-121 ($t=0.50$) | 1,486 images | **74.29%** | **91.70%** | **70.31%** | **0.8806** | **0.5975** | TP=254, TN=850, FP=359, FN=23 | Canonical uncalibrated benchmark (~74.3% accuracy) |
| **Skin Lesion (Deployed Triage API)** | DenseNet-121 ($t=0.40$) | 1,486 images | **68.37%** | **94.95%** | **62.28%** | **0.8806** | **0.5975** | TP=263, TN=753, FP=456, FN=14 | Active API runtime setting; sensitivity prioritized (14 missed) |
| **Diabetic Retinopathy** | EfficientNet-B0 ($t=0.50$) | 547 images | **97.99%** | **98.55%** | **97.42%** | **0.9958** | **0.9958** | TP=272, TN=264, FP=7, FN=4 | Cryptographic hash-isolated test split ($t=0.50$) |
| **Laboratory Reports** | Tesseract 5.x + Rules | 44 parameters | **100.0%** | N/A | N/A | N/A | N/A | Deterministic field extraction | Rule-based syntactic parsing (0.0% ML) |

### Controlled Experimental Evaluations & Research Experiments (Not Deployed in Production API):
| Experiment / Candidate | Architecture & Strategy | Operating Threshold | Accuracy | Sensitivity | Specificity | AUROC | PR-AUC | Primary Confusion Matrix | Technical Scope & Status |
|---|---|:---:|---:|---:|---:|---:|---:|---|---|
| **Controlled E0 Evaluation** | DenseNet-121 (Pillow Bicubic Preprocessing) | **0.40** | **75.17%** | **90.25%** | **71.71%** | **0.8787** | **0.5915** | TP=250, TN=867, FP=342, FN=27 | Controlled re-run with Bicubic interpolation (experimental comparison only) |
| *Controlled E0 Evaluation* | DenseNet-121 (Pillow Bicubic Preprocessing) | 0.50 | **81.29%** | **79.06%** | **81.80%** | **0.8787** | **0.5915** | TP=219, TN=989, FP=220, FN=58 | Controlled re-run at standard threshold |
| **Research E1 (Fine-Tuning)** | DenseNet-121 (denseblock4 unfrozen, lr=10^-5) | **0.50** | **81.83%** | **84.84%** | **81.14%** | **0.9127** | **0.7242** | TP=235, TN=981, FP=228, FN=42 | Experimental candidate at t=0.50; reduces FP by 50% vs deployed t=0.40 triage baseline (offline research) |
| *Research E1 (Balanced)* | DenseNet-121 (denseblock4 unfrozen, lr=10^-5) | 0.45 | **79.95%** | **88.81%** | **77.92%** | **0.9127** | **0.7242** | TP=246, TN=942, FP=267, FN=31 | Experimental candidate at balanced operating point |
| **Research E7 (Ensemble)** | Soft-Voting Ensemble (0.55 * E1 + 0.45 * E3) | **0.50** | **79.00%** | **90.25%** | **76.43%** | **0.9206** | **0.7425** | TP=250, TN=924, FP=285, FN=27 | Experimental ensemble; peak overall AUROC (offline research) |
| *Research E7 (Triage)* | Soft-Voting Ensemble (0.55 * E1 + 0.45 * E3) | 0.42 | **74.16%** | **94.22%** | **69.56%** | **0.9206** | **0.7425** | TP=261, TN=841, FP=368, FN=16 | Experimental ensemble triage candidate (offline research) |

---

## 16. Benchmark Datasets & Attribution

1. **Pediatric Chest Radiographs (Pneumonia):** Kermany et al. (2018), *Cell*. 5,856 pediatric radiographs partitioned with strict patient-level separation (910 test).
2. **Shenzhen & Montgomery Chest Radiographs (Tuberculosis):** Jaeger et al. (2014), *QIMS*. 800 frontal chest radiographs from Shenzhen No. 3 People's Hospital and Montgomery County Health Department (120 test).
3. **HAM10000 Dermatoscopy (Skin Lesion):** Tschandl et al. (2018), *Scientific Data (Nature)*. 10,015 dermoscopic images partitioned strictly by `lesion_id` across 7,470 unique lesions (1,486 test).
4. **APTOS 2019 Blindness Detection (Diabetic Retinopathy):** Kaggle / Asia Pacific Tele-Ophthalmology Society. 3,662 retina fundus photographs partitioned with cryptographic SHA-256 deduplication (547 test).

---

## 17. Methodological Limitations & Ethical Safeguards

1. **Coarse Attribution Boundary:** Grad-CAM heatmaps upsampled from $7 \times 7$ feature maps highlight class-discriminative anatomical regions for clinician verification. They do **not** represent cellular segmentations or surgical margins.
2. **Public Benchmark Metadata Constraints:** Public APTOS 2019 metadata lacks patient identifiers or lateral eye tags. While duplicate image hashes were strictly isolated, bilateral subject correlation cannot be mathematically disproven.
3. **Domain Shift in Tuberculosis:** Cross-site testing revealed lower sensitivity on the Montgomery cohort (66.67%) compared to Shenzhen (84.00%), reflecting demographic differences and X-ray tube calibration shifts.

---

## 18. Research Prototype & Responsible-Use Disclaimer

**IMPORTANT NOTICE:** This software is a research prototype developed strictly for educational, scientific, and experimental evaluation. It is **not certified by the US FDA, CE, or CDSCO** as a Software as a Medical Device (SaMD) for standalone diagnostic use, automated prescribing, or unsupervised patient triage.

- The system has not undergone prospective clinical trials, does not guarantee diagnostic accuracy or patient-safety outcomes, and must not be used as a substitute for professional medical advice, clinical judgment, diagnosis, or treatment.
- Visual attribution heatmaps (Grad-CAM) represent coarse, feature-level saliency distributions and do not constitute precise histological boundaries or anatomical segmentations.
- All model outputs, predictive scores, and visual explanations must be reviewed and verified by a licensed healthcare professional before any clinical consideration.

---

## 19. Literature References

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *IEEE ICCV*, pp. 618-626.
2. Huang, G., et al. (2017). "Densely Connected Convolutional Networks." *IEEE CVPR*, pp. 4700-4708.
3. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, PMLR 97, pp. 6105-6114.
4. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), pp. 1122-1131.
5. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific Data*, 5, 180161.
6. Jaeger, S., et al. (2014). "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quant. Imaging in Med. & Surg.*, 4(6), pp. 475-477.
7. Smith, R. (2007). "An Overview of the Tesseract OCR Engine." *IEEE ICDAR*, pp. 629-633.
