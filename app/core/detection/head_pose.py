"""
Head pose detection with async support.
Refactored from headpose.py - removed all commented code.
"""

import asyncio
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class HeadPoseResult:
    """Head pose detection result."""
    gaze_direction: str
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    success: bool = True


class HeadPoseDetector:
    """MediaPipe-based head pose detection."""
    
    def __init__(
        self,
        yaw_threshold: int = 37,
        pitch_up_threshold: int = 40,
        pitch_down_threshold: int = -37
    ):
        """
        Initialize head pose detector.
        
        Args:
            yaw_threshold: Threshold for left/right detection
            pitch_up_threshold: Threshold for looking up
            pitch_down_threshold: Threshold for looking down
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_up_threshold = pitch_up_threshold
        self.pitch_down_threshold = pitch_down_threshold
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1
        )
        logger.info("✅ HeadPoseDetector (MediaPipe FaceMesh) initialized. Utilizing underlying MediaPipe acceleration.")
        self.lock = asyncio.Lock()
    
    async def detect_pose(self, image_path: str) -> HeadPoseResult:
        """
        Detect head pose from image asynchronously.
        
        Args:
            image_path: Path to image file
            
        Returns:
            HeadPoseResult with pose information
        """
        async with self.lock:
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                self._detect_pose_sync,
                image_path
            )
            
            return result
    
    def _detect_pose_sync(self, image_path: str) -> HeadPoseResult:
        """Synchronous pose detection logic."""
        frame = cv2.imread(image_path)
        
        if frame is None:
            return HeadPoseResult(
                gaze_direction="Face direction Not available",
                success=False
            )
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return HeadPoseResult(
                gaze_direction="Face direction unknown",
                success=False
            )
        
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
        
        image_points = self._ref_2d_image_points(landmarks)
        camera_matrix = self._camera_matrix(8 * w, (w / 2, h / 2))
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self._ref_3d_model(),
            image_points,
            camera_matrix,
            dist_coeffs
        )
        
        if not success:
            return HeadPoseResult(
                gaze_direction="Face direction unknown",
                success=False
            )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        pitch = angles[0]
        yaw_angle = angles[1]
        roll = angles[2]
        
        # Mirror pitch
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch
        
        # Determine gaze direction
        gaze = self._determine_gaze(yaw_angle, pitch)
        
        return HeadPoseResult(
            gaze_direction=gaze,
            pitch=pitch,
            yaw=yaw_angle,
            roll=roll,
            success=True
        )
    
    def _determine_gaze(self, yaw: float, pitch: float) -> str:
        """Determine gaze direction from angles."""
        if yaw < -self.yaw_threshold:
            return "Looking left"
        elif yaw > self.yaw_threshold:
            return "Looking right"
        elif abs(yaw) <= self.yaw_threshold:
            if pitch > self.pitch_up_threshold:
                return "Looking up"
            elif pitch < self.pitch_down_threshold:
                return "Looking down"
            else:
                return "Looking forward"
        else:
            return "Face direction unknown"
    
    @staticmethod
    def _ref_3d_model() -> np.ndarray:
        """3D reference model points."""
        return np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float64)
    
    @staticmethod
    def _ref_2d_image_points(landmarks) -> np.ndarray:
        """Extract 2D image points from landmarks."""
        return np.array([
            landmarks[1],
            landmarks[152],
            landmarks[226],
            landmarks[446],
            landmarks[57],
            landmarks[287],
        ], dtype=np.float64)
    
    @staticmethod
    def _camera_matrix(fl: float, center: Tuple[float, float]) -> np.ndarray:
        """Create camera matrix."""
        return np.array([
            [fl, 0, center[0]],
            [0, fl, center[1]],
            [0, 0, 1]
        ], dtype=float)
    
    def cleanup(self):
        """Release resources."""
        try:
            self.face_mesh.close()
        except Exception:
            pass


async def create_head_pose_detector() -> HeadPoseDetector:
    """Factory function to create HeadPoseDetector."""
    return HeadPoseDetector()
