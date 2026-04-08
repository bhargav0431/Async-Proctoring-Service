"""Response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List


class DetectionResult(BaseModel):
    """Detailed detection result data."""
    
    category_name: str
    total_score: float
    count: int
    brightness: float


class ProctoringResponse(BaseModel):
    """Standard response for proctoring analysis."""
    
    response: str = Field(..., description="Primary detection message")
    response_orange: Optional[str] = Field(None, description="Warning/orange flag message")
    face_angle: Optional[str] = Field(None, description="Head pose direction")
    no_of_faces: int = Field(default=0, description="Number of faces detected")
    mobile: int = Field(default=0, description="Device detection flag (0=none, 1=detected, 2=maybe)")
    eye_result: str = Field(default="NA", description="Eye tracking result")
    error_code: int = Field(..., description="Status/error code")
    Facial_distance: float = Field(default=0.0, description="Face matching distance metric")
    Threshold_value: float = Field(default=0.5, description="Matching threshold used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Matched",
                "response_orange": None,
                "face_angle": "Looking forward",
                "no_of_faces": 1,
                "mobile": 0,
                "eye_result": "Looking center",
                "error_code": 1,
                "Facial_distance": 0.35,
                "Threshold_value": 0.55
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="ok", description="Service status")
    message: str = Field(default="Service is running", description="Status message")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "message": "Service is running",
                "version": "2.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[int] = Field(None, description="Application error code")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid base64 image data",
                "error_code": 0,
                "status_code": 400
            }
        }
