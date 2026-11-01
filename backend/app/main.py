"""
Main FastAPI Application Entry Point for XAI Framework for Patient Safety Backend.
Configures CORS, registers modality routers, mounts custom exception handlers, and exposes OpenAPI/Swagger documentation.
"""

import logging
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    BaseAPIException,
    api_exception_handler,
    generic_exception_handler,
)
from backend.app.api.routes import api_router

# Configure Application Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.main")

# Instantiate FastAPI Application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Modality-Aware Explainable AI (XAI) Framework for Patient Safety API. "
        "Supports Laboratory Report analysis (OCR, regex extraction, deterministic reference-range comparison) "
        "and parallel image modalities (Chest X-Ray, Skin Lesion, Retina + Grad-CAM)."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Custom Exception Handlers
app.add_exception_handler(BaseAPIException, api_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register API Routers under /api/v1
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", status_code=status.HTTP_200_OK, tags=["System Health"])
async def root():
    """System health check and API metadata endpoint."""
    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "openapi_url": f"{settings.API_V1_STR}/openapi.json",
        "supported_modalities": ["lab", "xray", "skin", "retina"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
