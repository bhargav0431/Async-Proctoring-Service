"""API routes package."""

from .health import router as health_router
from .proctoring import router as proctoring_router

__all__ = ["health_router", "proctoring_router"]
