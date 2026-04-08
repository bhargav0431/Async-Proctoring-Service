"""Detection modules package."""

from .object_detection import ObjectDetector, create_object_detector, ObjectDetectionResult
from .face_detection import FaceDetector, create_face_detector, FaceMatchResult
from .head_pose import HeadPoseDetector, create_head_pose_detector, HeadPoseResult
from .eye_tracking import EyeTracker, create_eye_tracker, EyeTrackingResult

__all__ = [
    "ObjectDetector",
    "create_object_detector",
    "ObjectDetectionResult",
    "FaceDetector",
    "create_face_detector",
    "FaceMatchResult",
    "HeadPoseDetector",
    "create_head_pose_detector",
    "HeadPoseResult",
    "EyeTracker",
    "create_eye_tracker",
    "EyeTrackingResult"
]
