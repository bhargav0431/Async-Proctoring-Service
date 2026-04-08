"""
FastAPI application entry point.
Main application with middleware, routing, and lifecycle management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request, status

from app.config import get_settings
from app.api.routes import health_router, proctoring_router
from app.dependencies import cleanup_detectors
from app import __version__
import logging
import tensorflow as tf
import os
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # Enable all TF logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} v{__version__}")
    print(f"📊 Configuration loaded from environment")
    print(f"🔧 Workers: {settings.workers}")
    print(f"🌐 CORS Origins: {settings.cors_origins}")
    
    # Check and Log GPU Availability
    logger.info("Initializing GPU Check...")
    
    # TensorFlow GPU Memory Growth & Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✅ TensorFlow GPU is AVAILABLE! Found {len(gpus)} GPUs: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ TensorFlow GPU memory growth configured.")
        except RuntimeError as e:
            logger.error(f"❌ TensorFlow GPU configuration error: {e}")
    else:
        logger.warning("❌ TensorFlow GPU is NOT AVAILABLE. Explicit GPU requirement might fail or fallback to CPU.")

    # PyTorch GPU Check (Used for face_recognition/dlib underlying if using torch)
    if torch.cuda.is_available():
        logger.info(f"✅ PyTorch GPU is AVAILABLE! Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("❌ PyTorch GPU is NOT AVAILABLE.")

    import dlib
    logger.info(f"{'✅' if dlib.DLIB_USE_CUDA else '❌'} Dlib CUDA Support: {dlib.DLIB_USE_CUDA}")
    
    # Detectors will be lazy-loaded on first request
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application...")
    await cleanup_detectors()
    print("✅ Cleanup complete")


def create_application() -> FastAPI:
    """
    Application factory - creates and configures FastAPI app.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Advanced proctoring microservice with face recognition, object detection, and behavioral analysis",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json"
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions gracefully."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error": str(exc) if settings.debug else "An error occurred"
            }
        )
    
    # Register routers
    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(proctoring_router, prefix=settings.api_v1_prefix)
    
    # Legacy compatibility routes (optional - for backward compatibility)
    @app.get("/home")
    async def legacy_home():
        """Legacy endpoint for backward compatibility."""
        return {"message": "It's working..."}
    
    return app


# Create app instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level="info"
    )
