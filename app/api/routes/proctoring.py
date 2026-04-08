"""
Proctoring API routes.
Handles image analysis and face verification endpoints.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.requests import ProctoringRequest, ProctoringTestRequest
from app.models.responses import ProctoringResponse, ErrorResponse
from app.services import ProctoringService
from app.dependencies import get_proctoring_service
from app.config import Settings, get_settings

router = APIRouter(prefix="/proctor", tags=["Proctoring"])


@router.post(
    "/analyze",
    response_model=ProctoringResponse,
    summary="Full proctoring analysis",
    description="Performs complete proctoring analysis including face matching, object detection, head pose, and eye tracking."
)
async def analyze_proctoring(
    request: ProctoringRequest,
    service: ProctoringService = Depends(get_proctoring_service),
    settings: Settings = Depends(get_settings)
):
    """
    
    This endpoint performs:
    - Image quality validation
    - Object detection (devices, multiple people)
    - Face matching against reference
    - Head pose detection
    - Eye tracking
    """
    try:
        # Determine reference image path based on source
        reg_file = request.regimage
        # If no extension, default to .jpg
        if not any(reg_file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            reg_file += ".jpg"

        if request.source == "sifycamp_not_valid":
            reference_path = f"/home/sites/saasunified/saas/exam/{request.source}/photos/candidate_photos/{reg_file}"
        else:
            reference_path = f"/data/{request.source}/exam/photos/candidate_photos/{reg_file}"
        
        # For local testing, use hardcoded path from settings
        if not os.path.exists(reference_path):
            reference_path = os.path.join(settings.reference_image_path, reg_file)
        
        # Perform analysis
        result = await service.analyze_full(
            base64_image=request.img2,
            camimage=request.camimage,
            reference_image_path=reference_path,
            source=request.source
        )
        
        # Convert to response model
        return ProctoringResponse(
            response=result.response,
            response_orange=result.response_orange,
            face_angle=result.face_angle,
            no_of_faces=result.no_of_faces,
            mobile=result.mobile,
            eye_result=result.eye_result,
            error_code=result.error_code,
            Facial_distance=result.facial_distance,
            Threshold_value=result.threshold_value
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        import traceback

        print(f"ERROR DURING ANALYSIS: {str(e)}")
        traceback.print_exc()

        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post(
    "/test",
    response_model=ProctoringResponse,
    summary="Simplified test analysis",
    description="Simplified proctoring test with basic face matching and device detection."
)
async def test_proctoring(
    request: ProctoringTestRequest,
    service: ProctoringService = Depends(get_proctoring_service),
    settings: Settings = Depends(get_settings)
):
    """
    
    
    This endpoint performs:
    - Image quality validation
    - Basic object detection
    - Face matching
    
    Faster than full analysis, suitable for testing.
    """
    try:
        # Use hardcoded local path for testing
        reference_path = os.path.join(
            "C:\\Users\\019147\\OneDrive - Sify Technologies Limited\\Documents\\saas_proctor\\saas_proctor",
            "19147.jpg"
        )
        
        # Or use from request if not hardcoded
        if not os.path.exists(reference_path):
            reference_path = request.regimage
        
        # Perform simplified analysis
        result = await service.analyze_test(
            base64_image=request.img2,
            camimage=request.camimage,
            reference_image_path=reference_path
        )
        
        return ProctoringResponse(
            response=result.response,
            response_orange=result.response_orange,
            face_angle=result.face_angle,
            no_of_faces=result.no_of_faces,
            mobile=result.mobile,
            eye_result=result.eye_result,
            error_code=result.error_code,
            Facial_distance=result.facial_distance,
            Threshold_value=result.threshold_value
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test analysis failed: {str(e)}"
        )
