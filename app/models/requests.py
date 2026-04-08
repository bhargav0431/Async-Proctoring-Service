"""Request models with Pydantic validation."""

from pydantic import BaseModel, Field, field_validator
import base64
from typing import Optional


class ProctoringRequest(BaseModel):
    """Request model for full proctoring analysis."""
    
    img2: str = Field(..., description="Base64 encoded image from camera")
    page: str = Field(..., description="Page identifier")
    source: str = Field(..., description="Source identifier (e.g., exam source)")
    regimage: str = Field(..., description="Reference/registration image filename")
    camimage: str = Field(..., description="Camera image identifier (membership number)")
    
    @field_validator('img2')
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        """Validate that img2 is a valid base64 string."""
        if not v:
            raise ValueError("Base64 image cannot be empty")
        
        # Remove data URL prefix if present
        test_string = v
        if v.startswith('data:image'):
            test_string = v.split(',')[1]
        
        # Add padding if needed
        padding = len(test_string) % 4
        if padding:
            test_string += '=' * (4 - padding)
        
        # Validate base64
        try:
            base64.b64decode(test_string)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")
        
        return v
    
    @field_validator('camimage', 'page', 'source', 'regimage')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that required fields are not empty."""
        if not v or not v.strip():
            raise ValueError(f"Field cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "img2": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "page": "1",
                "source": "exam_system",
                "regimage": "19147.jpg",
                "camimage": "19147"
            }
        }


class ProctoringTestRequest(BaseModel):
    """Request model for test/simplified proctoring endpoint."""
    
    img2: str = Field(..., description="Base64 encoded image from camera")
    page: str = Field(..., description="Page identifier")
    source: str = Field(..., description="Source identifier")
    regimage: str = Field(..., description="Reference image filename")
    camimage: str = Field(..., description="Camera image identifier")
    
    @field_validator('img2')
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        """Validate base64 image."""
        if not v:
            raise ValueError("Base64 image cannot be empty")
        
        test_string = v
        if v.startswith('data:image'):
            test_string = v.split(',')[1]
        
        padding = len(test_string) % 4
        if padding:
            test_string += '=' * (4 - padding)
        
        try:
            base64.b64decode(test_string)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "img2": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "page": "1",
                "source": "test_system",
                "regimage": "19147.jpg",
                "camimage": "19147"
            }
        }
