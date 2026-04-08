"""Health check and utility routes."""

from fastapi import APIRouter, Depends
from app.models.responses import HealthResponse
from app.config import Settings, get_settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Health check endpoint.
    Returns service status and version information.
    """
    return HealthResponse(
        status="ok",
        message="Service is running",
        version=settings.app_version
    )


@router.get("/", response_model=dict)
async def root():
    """Root endpoint - simple welcome message."""
    return {"message": "It's working..."}
