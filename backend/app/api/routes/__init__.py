"""
API routes package export.
"""

from fastapi import APIRouter
from backend.app.api.routes.lab import router as lab_router
from backend.app.api.routes.xray import router as xray_router
from backend.app.api.routes.skin import router as skin_router
from backend.app.api.routes.retina import router as retina_router

api_router = APIRouter()
api_router.include_router(lab_router)
api_router.include_router(xray_router)
api_router.include_router(skin_router)
api_router.include_router(retina_router)
