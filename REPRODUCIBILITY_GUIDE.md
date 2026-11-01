# REPRODUCIBILITY & DEPLOYMENT GUIDE

This guide provides instructions to install, verify, and execute the **Modality-Aware Explainable AI Framework for Patient Safety**.

---

## 1. Environment Setup

### System Prerequisites:
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS.
- **Python:** Version 3.10, 3.11, 3.12, or 3.13.
- **Node.js:** Version 18.0.0 or higher (with `npm` or `pnpm`).
- **Tesseract OCR:** 5.x installed and accessible via system PATH.
  - *Ubuntu:* `sudo apt-get install tesseract-ocr`
  - *Windows:* Download installer from UB-Mannheim and add to Environment Variables.

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <YOUR_REPOSITORY_URL>
cd xai-patient-safety

python -m venv .venv

# Activate virtual environment:
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 2. Configuration & Model Weights

### Environment Configuration:
```bash
cp .env.example .env
```
Default settings run locally out-of-the-box (`http://localhost:8000` for backend, `http://localhost:3000` for frontend).

### Model Weights Verification:
Ensure the four production weights reside in `models/`:
- `models/pneumonia/best_pneumonia_model.pt` (27.09 MB)
- `models/tb/best_tb_model.pt` (27.08 MB)
- `models/skin/binary/best_skin_binary_model.pt` (27.09 MB)
- `models/retina/best_retina_model.pt` (15.59 MB)

---

## 3. Launching the Services

### Start the FastAPI Backend:
In terminal 1 (with virtual environment activated):
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Health Endpoint: `http://localhost:8000/`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### Start the Next.js Frontend:
In terminal 2:
```bash
cd frontend
npm run dev
```
Open your browser to `http://localhost:3000`.

---

## 4. Running the Test Suite

Execute the automated test suite to verify all modules and API routes:
```bash
pytest tests/ -v
```
To run the end-to-end integration test specifically:
```bash
pytest tests/test_backend_api.py -v
```

---

## 5. Sample API Execution via cURL

### 1. Health Check:
```bash
curl -X GET http://localhost:8000/
```

### 2. Pneumonia CXR Screening with Grad-CAM:
```bash
curl -X POST http://localhost:8000/api/v1/xray/predict/pneumonia \
  -F "file=@tests/sample_reports/sample_image_report.jpg"
```

### 3. Laboratory Report OCR Parsing:
```bash
curl -X POST http://localhost:8000/api/v1/lab/analyze \
  -F "file=@tests/sample_reports/synthetic_comprehensive_panel.pdf"
```
