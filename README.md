# Development of a Framework Using Explainable Artificial Intelligence for Patient Safety

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-16.3.8-black?logo=next.js)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19.2.8-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?logo=tailwindcss)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Status](https://img.shields.io/badge/B.Tech%20Major%20Project-GNIOT%20%2F%20AKTU-blue)](https://aktu.ac.in)

> **Academic Affiliation:** Greater Noida Institute of Technology (GNIOT) / Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow, Uttar Pradesh, India  
> **Department:** Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
> **Authors:** Dev Verma, Ashish Kumar Singh, Anshuman Tripathi  
> **Project Supervisor:** Dr. Mohd Jawed Khan  
> **Academic Session:** 2025–2026  

---

## 1. Overview

This repository contains the complete implementation of the **Modality-Aware Explainable Artificial Intelligence (XAI) Framework for Patient Safety**. Designed as an investigational clinical decision-support research prototype (second-reader platform), the system addresses the "black-box" dilemma in medical machine learning by coupling deep neural inference with deterministic visual attribution and defensive quality gates.

The platform spans **five diagnostic modules**:
1. **Clinical Laboratory Report Understanding:** Deterministic OCR (Tesseract 5.x) and regex extraction engine evaluating biological reference intervals without statistical guesswork.
2. **Pediatric Pneumonia Detection:** DenseNet-121 on frontal chest radiographs with layer-targeted Grad-CAM visual attribution.
3. **Pulmonary Tuberculosis Screening:** DenseNet-121 on multi-site chest radiographs with apical parenchymal cavity attribution.
4. **Dermoscopic Skin Lesion Triage:** DenseNet-121 production baseline operating at a sensitivity-prioritized threshold ($t=0.40$), alongside prominent experimental research models (**E1 fine-tuned: 81.83% accuracy**; **E7 soft-voting ensemble: 79.00% accuracy, 0.9206 AUROC**).
5. **Diabetic Retinopathy Screening:** Compound-scaled EfficientNet-B0 on digital fundus photography with macular microvascular attribution.

---

## 2. Problem Statement & Clinical Motivation

Deep learning models achieve diagnostic accuracy comparable to medical specialists on benchmark datasets. However, deploying them in safety-critical clinical environments introduces severe patient safety risks:
- **The Black-Box Failure Mode:** Standard convolutional networks produce scalar probability scores without explaining which anatomical regions drove the decision. A model can achieve 95% accuracy by learning confounding hospital markers, radiographic orientation tags, or imaging artifacts rather than genuine pathology.
- **Garbage-In, Catastrophe-Out:** Conventional inference APIs silently process out-of-distribution, severely blurred, inverted, or non-medical images, generating high-confidence false predictions on uninterpretable data.
- **Asymmetric Risk in Oncology:** Standard 0.50 decision thresholds optimize overall accuracy at the expense of false negatives. In melanoma triage, a missed cancer is life-threatening, whereas a false alarm triggers a non-invasive dermoscopic biopsy.

This project implements a multi-tier safety architecture: **Defensive Quality Gates** intercept invalid inputs before inference, **Layer-Specific Grad-CAM** highlights causal pathological features, and **Sensitivity-Prioritized Thresholding** protects clinical safety.

---

## 3. Core Objectives

1. Develop a unified, multi-modal clinical decision-support framework covering radiological, dermatological, retinal, and biochemical diagnostic data.
2. Integrate mathematical Gradient-Weighted Class Activation Mapping (Grad-CAM) directly into the inference lifecycle, generating synchronized raw, heatmap, and overlay views.
3. Establish defensive pre-inference quality checks (MIME validation, dimension gating, and Laplacian focus blur estimation).
4. Build a deterministic laboratory report understanding module that extracts biomarkers and compares them against physiological reference ranges with zero statistical imputation.
5. Empirically validate explainability faithfulness using systematic salient pixel deletion experiments.

---

## 4. Key Architectural Features

- **Decoupled Four-Tier Architecture:** Complete separation between the Presentation Tier (Next.js 16/React 19), Application API Gateway (FastAPI/Pydantic), Inference Tier (PyTorch/torchvision), and Storage/Audit Tier.
- **Defensive Input Quality Gates:** Rejects unreadable inputs using Laplacian focus variance ($\sigma^2 < 100.0$) and MIME type sniffing before model execution.
- **Three-View Synchronized Explainability:** Presents the clinician with the original raw image, isolated jet saliency map, and interactive alpha-blended diagnostic overlay with dynamic opacity controls.
- **Zero Imputation Policy:** If a laboratory document lacks an analyte or unit, the system triggers a `review_required = True` flag rather than guessing synthetic values.
- **In-Memory Ephemeral Processing:** Patient imagery is converted to memory buffers (`io.BytesIO()`) and processed without persistent local caching, preserving patient data privacy.

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
- **Production Baseline Model:** DenseNet-121 evaluated on 1,486 lesion-grouped HAM10000 images using TorchVision Bilinear interpolation.
  - **Deployed Threshold ($t=0.40$):** **68.37% Accuracy**, **94.95% Sensitivity** (only 14 missed malignancies out of 277), **62.28% Specificity**, **0.8806 AUROC**, **0.5975 PR-AUC**.
  - **Standard Threshold ($t=0.50$):** **74.29% Accuracy**, **91.70% Sensitivity**, **70.31% Specificity** (23 missed malignancies).
- **Prominent Research Breakthroughs (E1 & E7):**
  - **Experiment E1 (Partial Fine-Tuning):** DenseNet-121 with `denseblock4` unfrozen ($	ext{lr}=10^{-5}$). Achieves **81.83% Accuracy**, **84.84% Sensitivity**, **81.14% Specificity**, **0.9127 AUROC**, and **0.7242 PR-AUC** at $t=0.50$, slashing false alarms by 50% (from 456 down to 228).
  - **Experiment E7 (Soft-Voting Ensemble):** Weighted ensemble ($0.55	ext{E1} + 0.45	ext{E3}$). Achieves **79.00% Accuracy**, **90.25% Sensitivity**, and project-high **0.9206 AUROC** at $t=0.50$; or **74.16% Accuracy**, **94.22% Sensitivity**, and **98.13% NPV** at $t=0.42$.

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

$$lpha_k^c = rac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} rac{\partial Y^c}{\partial A_{i,j}^k}$$

where $Y^c$ is the pre-softmax model score for class $c$, $A_{i,j}^k$ is the activation at spatial coordinate $(i,j)$ of feature map $k$, and $Z = U 	imes V$ is the spatial dimensions.

The coarse class-discriminative saliency map is formed via a rectified linear combination:

$$L_{	ext{Grad-CAM}}^c = 	ext{ReLU}\left( \sum_{k} lpha_k^c A^k 
ight)$$

### Implementation Hooks in Codebase:
- **DenseNet-121:** Registered on `features.denseblock4.denselayer16.conv2`.
- **EfficientNet-B0:** Registered on `features.8.0`.

### Quantitative Faithfulness Validation (Pixel Deletion Test):
When the top-20% most salient pixels identified by Grad-CAM are masked across 20 verified true positive cases, target model confidence degrades significantly:
- **Tuberculosis CXR:** $-61.16\%$ drop ($0.824 
ightarrow 0.320$)
- **Skin Lesion:** $-49.81\%$ drop ($0.781 
ightarrow 0.392$)
- **Diabetic Retinopathy:** $-28.43\%$ drop ($0.985 
ightarrow 0.705$)
- **Pneumonia CXR:** $-14.27\%$ drop ($0.897 
ightarrow 0.769$)

This confirms that the visual heatmaps correspond to causally relevant pathological features rather than background noise.

---

## 9. Defensive Input Quality Gates

To prevent uninterpretable images from generating misleading predictions, the platform enforces three mandatory pre-inference gates (`backend/app/services/validation_service.py`):

1. **Magic Bytes MIME Validation:** Inspects raw file header bytes to confirm genuine JPEG, PNG, or PDF formats; rejects non-medical payloads (.exe, .zip, .sh).
2. **Dimension & Aspect Ratio Verification:** Ensures minimum pixel dimensions ($224 	imes 224$) and valid image channels.
3. **Laplacian Focus Blur Estimation:** Computes the variance of the 2D Laplacian operator over the grayscale input image $I$:
   $$	ext{Blur Score} = 	ext{Var}(
abla^2 I) = rac{1}{N} \sum_{x,y} \left( 
abla^2 I(x,y) - \mu_{
abla^2 I} 
ight)^2$$
   Images with $	ext{Blur Score} < 100.0$ are automatically intercepted with HTTP 400 Bad Request and an explicit request to recapture.

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
git clone https://github.com/your-username/xai-patient-safety.git
cd xai-patient-safety
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
The clinical interface will be accessible in your web browser at `http://localhost:3000`.

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
| Diagnostic Task | Model Architecture | Test Cohort Size | Accuracy | Sensitivity | Specificity | AUROC | PR-AUC | Primary Confusion Matrix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Pneumonia (CXR)** | DenseNet-121 | 910 images | **90.66%** | **90.15%** | **92.08%** | **0.9621** | **0.9866** | TP=604, TN=221, FP=19, FN=66 |
| **Tuberculosis (CXR)** | DenseNet-121 | 120 images | **85.83%** | **81.36%** | **90.16%** | **0.9228** | **0.9353** | TP=48, TN=55, FP=6, FN=11 |
| **Skin Lesion (Deployed)** | DenseNet-121 ($t=0.40$) | 1,486 images | **68.37%** | **94.95%** | **62.28%** | **0.8806** | **0.5975** | TP=263, TN=753, FP=456, FN=14 |
| **Skin Lesion (Standard)** | DenseNet-121 ($t=0.50$) | 1,486 images | **74.29%** | **91.70%** | **70.31%** | **0.8806** | **0.5975** | TP=254, TN=850, FP=359, FN=23 |
| **Diabetic Retinopathy** | EfficientNet-B0 | 547 images | **97.99%** | **98.55%** | **97.42%** | **0.9958** | **0.9958** | TP=272, TN=264, FP=7, FN=4 |
| **Laboratory Reports** | Tesseract 5.x + Rules | 44 parameters | **100.0%** | N/A | N/A | N/A | N/A | Deterministic syntactic parsing |

### Skin Lesion Research Breakthroughs (E1 & E7 Candidates):
| Candidate Configuration | Strategy | Operating Threshold | Accuracy | Sensitivity | Specificity | AUROC | Role & Status |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Production Baseline (Deployed)** | Frozen Baseline | **0.40** | **68.37%** | **94.95%** | **62.28%** | **0.8806** | **Deployed Production System (14 FN)** |
| **E1 — Partial Fine-Tuning** | Block 4 Unfrozen | **0.50** | **81.83%** | **84.84%** | **81.14%** | **0.9127** | **Peak Single-Model Research Accuracy** |
| *E1 — Balanced Operating Point* | Block 4 Unfrozen | 0.45 | **79.95%** | **88.81%** | **77.92%** | **0.9127** | Balanced Research Point (Bal Acc 83.36%) |
| **E7 — Soft-Voting Ensemble** | Ensemble ($0.55	ext{E1}+0.45	ext{E3}$) | **0.50** | **79.00%** | **90.25%** | **76.43%** | **0.9206** | **Peak Overall Research AUROC (0.9206)** |
| *E7 — Clinical Triage Point* | Ensemble ($0.55	ext{E1}+0.45	ext{E3}$) | **0.42** | **74.16%** | **94.22%** | **69.56%** | **0.9206** | Near-Production Recall (16 FN), +88 Specificity |

---

## 16. Benchmark Datasets & Attribution

1. **Pediatric Chest Radiographs (Pneumonia):** Kermany et al. (2018), *Cell*. 5,856 pediatric radiographs partitioned with strict patient-level separation (910 test).
2. **Shenzhen & Montgomery Chest Radiographs (Tuberculosis):** Jaeger et al. (2014), *QIMS*. 800 frontal chest radiographs from Shenzhen No. 3 People's Hospital and Montgomery County Health Department (120 test).
3. **HAM10000 Dermatoscopy (Skin Lesion):** Tschandl et al. (2018), *Scientific Data (Nature)*. 10,015 dermoscopic images partitioned strictly by `lesion_id` across 7,470 unique lesions (1,486 test).
4. **APTOS 2019 Blindness Detection (Diabetic Retinopathy):** Kaggle / Asia Pacific Tele-Ophthalmology Society. 3,662 retina fundus photographs partitioned with cryptographic SHA-256 deduplication (547 test).

---

## 17. Methodological Limitations & Ethical Safeguards

1. **Coarse Attribution Boundary:** Grad-CAM heatmaps upsampled from $7 	imes 7$ feature maps highlight class-discriminative anatomical regions for clinician verification. They do **not** represent cellular segmentations or surgical margins.
2. **Public Benchmark Metadata Constraints:** Public APTOS 2019 metadata lacks patient identifiers or lateral eye tags. While duplicate image hashes were strictly isolated, bilateral subject correlation cannot be mathematically disproven.
3. **Domain Shift in Tuberculosis:** Cross-site testing revealed lower sensitivity on the Montgomery cohort (66.67%) compared to Shenzhen (84.00%), reflecting demographic differences and X-ray tube calibration shifts.

---

## 18. Academic Disclaimer

**IMPORTANT NOTICE:** This software is an academic proof-of-concept and research prototype developed for educational and experimental purposes. It is **not certified by the US FDA, CE, or CDSCO** as a Software as a Medical Device (SaMD) for independent diagnostic use, automated prescribing, or unsupervised patient triage. All outputs and attribution heatmaps must be independently reviewed by a qualified healthcare provider.

---

## 19. Contributors & Academic Supervision

### Student Engineering Team:
- **Dev Verma** - Department of Computer Science & Engineering (AI & ML)
- **Ashish Kumar Singh** - Department of Computer Science & Engineering (AI & ML)
- **Anshuman Tripathi** - Department of Computer Science & Engineering (AI & ML)

### Project Supervisor:
- **Dr. Mohd Jawed Khan** — Associate Professor, Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning), Greater Noida Institute of Technology (GNIOT), Greater Noida, Uttar Pradesh, India.

---

## 20. Acknowledgements & Literature References

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *IEEE ICCV*, pp. 618-626.
2. Huang, G., et al. (2017). "Densely Connected Convolutional Networks." *IEEE CVPR*, pp. 4700-4708.
3. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, PMLR 97, pp. 6105-6114.
4. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), pp. 1122-1131.
5. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific Data*, 5, 180161.
6. Jaeger, S., et al. (2014). "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quant. Imaging in Med. & Surg.*, 4(6), pp. 475-477.
7. Smith, R. (2007). "An Overview of the Tesseract OCR Engine." *IEEE ICDAR*, pp. 629-633.
