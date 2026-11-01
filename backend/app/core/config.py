"""
Centralized backend configuration management using Pydantic Settings.
Environment variables override default settings. Absolute machine-specific paths are prohibited.
"""

import os
from typing import List, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application setting configurations."""

    PROJECT_NAME: str = "XAI Framework for Patient Safety API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Security & Limits
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Model & Artifact Directory (Relative to Project Root)
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models")
    LAB_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "lab", "final", "xgb_model.joblib")
    LAB_PREPROCESSOR_PATH: str = os.path.join(PROJECT_ROOT, "models", "lab", "final", "preprocessor.joblib")
    LAB_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "models", "lab", "final", "model_metadata.json")

    PNEUMONIA_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "pneumonia", "best_pneumonia_model.pt")
    PNEUMONIA_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "models", "pneumonia", "model_metadata.json")

    TB_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "tb", "best_tb_model.pt")
    TB_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "models", "tb", "model_metadata.json")

    SKIN_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "skin", "binary", "best_skin_binary_model.pt")
    SKIN_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "models", "skin", "binary", "model_metadata.json")

    RETINA_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "retina", "best_retina_model.pt")
    RETINA_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "models", "retina", "model_metadata.json")

    # OCR Tesseract Configuration
    # Uses env TESSERACT_CMD if defined, or checks standard installation path dynamically
    TESSERACT_CMD: str = Field(
        default_factory=lambda: os.environ.get(
            "TESSERACT_CMD",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            else "tesseract"
        )
    )

    class Config:
        case_sensitive = True


settings = Settings()
