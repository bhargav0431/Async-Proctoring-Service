"""
Proctoring service - business logic layer.
Combines all detection services for complete proctoring analysis.
"""

import asyncio
import os
from typing import Tuple, Optional
from dataclasses import dataclass

from app.core.detection import (
    ObjectDetector,
    FaceDetector,
    HeadPoseDetector,
    EyeTracker
)
from app.core.utils import (
    convert_base64_to_image,
    save_base64_image,
    analyze_image_quality
)
from app.config import Settings


@dataclass
class ProctoringResult:
    """Complete proctoring analysis result."""
    response: str
    response_orange: Optional[str] = None
    face_angle: str = "NA"
    no_of_faces: int = 0
    mobile: int = 0
    eye_result: str = "NA"
    error_code: int = 0
    facial_distance: float = 0.0
    threshold_value: float = 0.5


class ProctoringService:
    """
    Main proctoring service coordinating all detection modules.
    Designed for async concurrent processing.
    """
    
    def __init__(
        self,
        settings: Settings,
        object_detector: ObjectDetector,
        face_detector: FaceDetector,
        head_pose_detector: HeadPoseDetector,
        eye_tracker: EyeTracker
    ):
        """
        Initialize proctoring service with detector instances.
        
        Args:
            settings: Application settings
            object_detector: Object detection service
            face_detector: Face detection service
            head_pose_detector: Head pose detection service
            eye_tracker: Eye tracking service
        """
        self.settings = settings
        self.object_detector = object_detector
        self.face_detector = face_detector
        self.head_pose_detector = head_pose_detector
        self.eye_tracker = eye_tracker
    
    async def analyze_full(
        self,
        base64_image: str,
        camimage: str,
        reference_image_path: str,
        source: str
    ) -> ProctoringResult:
        
        # Convert and analyze image quality concurrently
        image_task = asyncio.create_task(convert_base64_to_image(base64_image))
        image = await image_task
        
        # Analyze image quality
        is_valid, error_msg, metrics = await analyze_image_quality(
            image,
            brightness_threshold=self.settings.brightness_threshold,
            black_threshold=self.settings.black_thresh_rgb,
            black_ratio_threshold=self.settings.black_pixel_threshold
        )
        
        if not is_valid:
            return ProctoringResult(
                response=error_msg,
                response_orange="NA",
                error_code=9
            )
        
        # Save input image for processing with unique name to prevent collisions
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        image_filename = f"{camimage}_{unique_id}.png"
        image_path = os.path.join(self.settings.image_storage_path, image_filename)
        
        try:
            await save_base64_image(base64_image, image_path)
        except Exception as e:
            print(f"FAILED TO SAVE IMAGE: {e}")
            return ProctoringResult(
                response="Image processing error",
                error_code=9
            )
        
        try:
            # Run all detections concurrently
            object_task = asyncio.create_task(
                self.object_detector.detect_objects(image_path, metrics["brightness"])
            )
            head_pose_task = asyncio.create_task(
                self.head_pose_detector.detect_pose(image_path)
            )
            
            object_result, head_pose_result = await asyncio.gather(
                object_task,
                head_pose_task
            )
            
            # Initialize result values
            display_result = ["none", "none", "none"]
            error_code = 0
            face_distance = 0.0
            face_count = object_result.person_count if object_result.has_detection else 0
            eye_result = "NA"
            ui_result = ""
            face_angle = head_pose_result.gaze_direction
            dist_from_screen = None
            
            # Check for device detection
            if object_result.device_flag == 1:
                return ProctoringResult(
                    response="Device detected",
                    error_code=6,
                    mobile=1,
                    face_angle=face_angle
                )
            elif object_result.device_flag == 2:
                display_result[1] = "Mobile maybe detected"
                error_code = 12
            
            # Get person category
            person_category = object_result.categories.get("person")
            
            if not person_category:
                return ProctoringResult(
                    response="no object and face detected",
                    error_code=4,
                    face_angle=face_angle
                )
            
            if person_category.count > 1:
                return ProctoringResult(
                    response="many faces detected",
                    no_of_faces=person_category.count,
                    error_code=3,
                    face_angle=face_angle
                )
            
            # Face matching for single person
            face_match_result = await self.face_detector.compare_faces(
                reference_image_path,
                image_path,
                threshold=self.settings.face_match_threshold
            )
            
            face_distance = face_match_result.face_distance
            # Distance calculation disabled
            # dist_from_screen = face_match_result.screen_distance
            dist_from_screen = None
            
            # Check face brightness
            if face_match_result.face_brightness < self.settings.face_brightness_threshold:
                return ProctoringResult(
                    response="Face not clear",
                    error_code=5,
                    facial_distance=face_distance,
                    face_angle=face_angle
                )
            
            if not face_match_result.is_match:
                return ProctoringResult(
                    response="Not Matched",
                    error_code=2,
                    facial_distance=face_distance,
                    face_angle=face_angle
                )
            
            # Face matched - proceed with detailed analysis
            error_code = 1
            
            # Check head pose
            if head_pose_result.gaze_direction != "Looking forward":
                error_code = 8
                display_result[0] = head_pose_result.gaze_direction
            else:
                # Check eye tracking if looking forward
                eye_track_result = await self.eye_tracker.detect_gaze(image_path)
                eye_result = eye_track_result.status
                
                if eye_result != "Looking center":
                    error_code = 7
                    display_result[0] = eye_result
            
            # Check screen distance
            if dist_from_screen:
                if dist_from_screen > 100 or dist_from_screen < 30:
                    error_code = 8
                    display_result[0] = "Face outside the box"
            
            # Determine final response
            response = self._determine_final_response(
                display_result,
                error_code,
                face_match_result.is_match,
                eye_result
            )
        
            return ProctoringResult(
                response=response,
                response_orange=display_result[1] if display_result[1] != "none" else None,
                face_angle=face_angle,
                no_of_faces=face_count,
                mobile=object_result.device_flag,
                eye_result=eye_result,
                error_code=error_code,
                facial_distance=face_distance,
                threshold_value=self.settings.face_match_threshold
            )
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(image_path):
                    # print(image_path)
                    os.remove(image_path)
            except Exception:
                pass
    
    async def analyze_test(
        self,
        base64_image: str,
        camimage: str,
        reference_image_path: str
    ) -> ProctoringResult:
        """
        Simplified test analysis (imgtest endpoint logic).
        
        Args:
            base64_image: Base64 encoded image
            camimage: Camera image identifier
            reference_image_path: Reference image path
            
        Returns:
            ProctoringResult with basic analysis
        """
        # Convert and analyze image quality
        image = await convert_base64_to_image(base64_image)
        
        is_valid, error_msg, metrics = await analyze_image_quality(
            image,
            brightness_threshold=self.settings.brightness_threshold,
            black_threshold=20,  # Different threshold for test
            black_ratio_threshold=0.3
        )
        
        if not is_valid:
            return ProctoringResult(
                response=error_msg,
                error_code=0
            )
        
        # Check for very low brightness
        if metrics["brightness"] < 50:
            return ProctoringResult(
                response="Surrounding brightness very low",
                error_code=0
            )
        
        # Save and process image
        image_path = os.path.join(self.settings.image_storage_path, f"{camimage}.png")
        await save_base64_image(base64_image, image_path)
        
        # Run object detection and face matching concurrently
        object_task = asyncio.create_task(
            self.object_detector.detect_objects(image_path, metrics["brightness"])
        )
        face_task = asyncio.create_task(
            self.face_detector.compare_faces(
                reference_image_path,
                image_path,
                threshold=self.settings.face_match_threshold
            )
        )
        
        object_result, face_match_result = await asyncio.gather(
            object_task,
            face_task
        )
        
        # Check for device
        if object_result.device_flag in [1, 2]:
            return ProctoringResult(
                response="Device detected",
                error_code=0
            )
        
        # Check face brightness
        if face_match_result.face_brightness < 40:
            return ProctoringResult(
                response="Face not clear",
                error_code=0
            )
        
        # Check face match
        if face_match_result.is_match:
            return ProctoringResult(
                response=face_match_result.message,
                error_code=1,
                facial_distance=face_match_result.face_distance,
                no_of_faces=1
            )
        
        # Handle multiple faces
        if object_result.person_count > 1:
            return ProctoringResult(
                response="many faces detected",
                no_of_faces=object_result.person_count,
                error_code=0
            )
        
        return ProctoringResult(
            response="Not Matched",
            error_code=0,
            facial_distance=face_match_result.face_distance
        )
    
    def _determine_final_response(
        self,
        display_result: list,
        error_code: int,
        is_face_matched: bool,
        eye_result: str
    ) -> str:
        """Determine final response message based on all checks."""
        if error_code == 1 and is_face_matched:
            if display_result[0] in ["none", "Looking center", "Looking forward"]:
                return "Matched"
        
        if display_result[0] not in ["none", "Looking center", "Looking forward", "Matched"]:
            return display_result[0]
        
        if display_result[1] != "none":
            return display_result[1]
        
        if eye_result not in ["Looking center", "NA"]:
            return eye_result
        
        return "Matched" if is_face_matched else "Not Matched"
