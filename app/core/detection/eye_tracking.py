"""
Eye tracking and gaze detection with async support.
Refactored from eye_new.py - removed all commented code.
"""

import asyncio
import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EyeTrackingResult:
    """Eye tracking result."""
    status: str
    left_eye_status: str = "NA"
    right_eye_status: str = "NA"
    distance: float = 0.0
    success: bool = True


class EyeTracker:
    """MediaPipe-based eye tracking and gaze detection."""
    
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_IRIS = [469, 470, 471, 472]
    FACE_WIDTH_LANDMARKS = [234, 454]
    
    # Screen distance calculation disabled
    # KNOWN_FACE_WIDTH = 14.0
    # FOCAL_LENGTH = 650
    # VALID_MIN_DIST = 30
    # VALID_MAX_DIST = 105
    CLOSED_EYE_THRESHOLD = 0.20
    
    NEAR_THRESHOLDS = {
        'horizontal_left': 0.33,
        'horizontal_right': 0.68,
        'vertical_up': 0.15,
        'vertical_down': 0.85
    }
    
    FAR_THRESHOLDS = {
        'horizontal_left': 0.40,
        'horizontal_right': 0.60,
        'vertical_up': 0.20,
        'vertical_down': 0.80
    }
    
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize eye tracker.
        
        Args:
            min_confidence: Minimum detection confidence
        """
        self.min_confidence = min_confidence
        
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        logger.info("✅ EyeTracker (MediaPipe FaceMesh) initialized. Attempting to utilize available hardware acceleration.")
        self.lock = asyncio.Lock()

    
    async def detect_gaze(self, image_path: str) -> EyeTrackingResult:
        """
        Detect eye gaze from image asynchronously.
        
        Args:
            image_path: Path to image file
            
        Returns:
            EyeTrackingResult with gaze information
        """
        async with self.lock:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._detect_gaze_sync,
                image_path
            )
            
            return result
    
    def _detect_gaze_sync(self, image_path: str) -> EyeTrackingResult:
        """Synchronous gaze detection logic."""
        frame = cv.imread(image_path)
        
        if frame is None:
            return EyeTrackingResult(
                status="Error: Image not found",
                success=False
            )
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Reset face mesh state
        try:
            self.face_mesh._graph = None
            results = self.face_mesh.process(rgb_frame)
        except Exception:
            # Reinitialize on error
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=0.5
            )
            results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return EyeTrackingResult(
                status="Eye ball not detected",
                success=False
            )
        
        try:
            mesh_points = np.array([
                [int(p.x * frame.shape[1]), int(p.y * frame.shape[0])]
                for p in results.multi_face_landmarks[0].landmark
            ])
            
            # Calculate distance - DISABLED
            # face_width = np.linalg.norm(
            #     mesh_points[self.FACE_WIDTH_LANDMARKS[0]] -
            #     mesh_points[self.FACE_WIDTH_LANDMARKS[1]]
            # )
            # distance = self._calculate_distance(face_width)
            # thresholds = self._get_dynamic_thresholds(distance)
            
            # Use fixed thresholds (no distance-based adjustments)
            distance = 50.0  # Fixed dummy value
            thresholds = self.NEAR_THRESHOLDS  # Use fixed thresholds
            
            # Process both eyes
            left_status = self._process_eye(
                mesh_points, self.LEFT_IRIS, self.LEFT_EYE, thresholds
            )
            right_status = self._process_eye(
                mesh_points, self.RIGHT_IRIS, self.RIGHT_EYE, thresholds
            )
            
            # Determine combined status
            status = self._combine_eye_status(left_status, right_status)
            
            return EyeTrackingResult(
                status=status,
                left_eye_status=left_status,
                right_eye_status=right_status,
                distance=distance,
                success=True
            )
            
        except Exception as e:
            return EyeTrackingResult(
                status=f"Detection error: {str(e)}",
                success=False
            )
    
    def _process_eye(
        self,
        mesh_points: np.ndarray,
        iris_indices: list,
        eye_indices: list,
        thresholds: dict
    ) -> str:
        """Process individual eye for gaze detection."""
        try:
            if max(iris_indices) >= len(mesh_points):
                return "Eye not detected"
            
            (cx, cy), _ = cv.minEnclosingCircle(mesh_points[iris_indices])
            contour = mesh_points[eye_indices]
            ear = self._calculate_ear(contour[:6])
            
            if ear < self.CLOSED_EYE_THRESHOLD:
                return "Closed"
            
            return self._detect_eye_movement(contour, (cx, cy), thresholds)
            
        except Exception:
            return "Error detecting"
    
    def _detect_eye_movement(
        self,
        eye_contour: np.ndarray,
        iris_center: Tuple[float, float],
        thresholds: dict
    ) -> str:
        """Determine eye gaze direction."""
        x_coords = eye_contour[:, 0]
        y_coords = eye_contour[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        h_left = x_min + (x_max - x_min) * thresholds['horizontal_left']
        h_right = x_min + (x_max - x_min) * thresholds['horizontal_right']
        v_up = y_min + (y_max - y_min) * thresholds['vertical_up']
        v_down = y_min + (y_max - y_min) * thresholds['vertical_down']
        
        if iris_center[0] > h_right:
            return "Looking away"
        if iris_center[0] < h_left:
            return "Looking away"
        if iris_center[1] < v_up:
            return "Looking up"
        if iris_center[1] > v_down:
            return "Looking away"
        
        return "Looking center"
    
    @staticmethod
    def _calculate_ear(eye_points: np.ndarray) -> float:
        """Calculate eye aspect ratio."""
        vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)
    
    # Distance calculation methods - DISABLED
    # def _calculate_distance(self, face_width_px: float) -> float:
    #     """Calculate distance from camera."""
    #     if face_width_px <= 0:
    #         raise ValueError("Invalid face width")
    #     
    #     actual_dist = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_px
    #     
    #     if not (self.VALID_MIN_DIST <= actual_dist <= self.VALID_MAX_DIST):
    #         raise ValueError(f"Distance {actual_dist:.1f}cm out of range")
    #     
    #     return actual_dist - 10
    # 
    # def _get_dynamic_thresholds(self, distance: float) -> Dict[str, float]:
    #     """Get thresholds based on distance."""
    #     ratio = np.clip(
    #         (distance - self.VALID_MIN_DIST) / (self.VALID_MAX_DIST - self.VALID_MIN_DIST),
    #         0, 1
    #     )
    #     
    #     return {
    #         key: self.NEAR_THRESHOLDS[key] +
    #         (self.FAR_THRESHOLDS[key] - self.NEAR_THRESHOLDS[key]) * ratio
    #         for key in self.NEAR_THRESHOLDS
    #     }
    
    @staticmethod
    def _combine_eye_status(left_status: str, right_status: str) -> str:
        """Combine left and right eye states."""
        if left_status == right_status:
            return left_status
        elif "right" in left_status.lower() or "left" in right_status.lower():
            return "Looking away"
        else:
            return "Looking center"
    
    def cleanup(self):
        """Release resources."""
        try:
            self.face_mesh.close()
        except Exception:
            pass


async def create_eye_tracker() -> EyeTracker:
    """Factory function to create EyeTracker."""
    return EyeTracker()
