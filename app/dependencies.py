"""
Dependency injection for FastAPI.
Provides singleton detector instances and services.
"""

from functools import lru_cache
from app.config import Settings, get_settings
from app.core.detection import (
    ObjectDetector,
    FaceDetector,
    HeadPoseDetector,
    EyeTracker
)
from app.services import ProctoringService


_object_detector: ObjectDetector = None
_face_detector: FaceDetector = None
_head_pose_detector: HeadPoseDetector = None
_eye_tracker: EyeTracker = None


async def get_object_detector() -> ObjectDetector:
    """Get or create object detector singleton."""
    global _object_detector
    if _object_detector is None:
        settings = get_settings()
        _object_detector = ObjectDetector(
            model_path=settings.efficientdet_model_path,
            max_results=settings.mediapipe_max_results,
            score_threshold=settings.mediapipe_score_threshold
        )
    return _object_detector


async def get_face_detector() -> FaceDetector:
    """Get or create face detector singleton."""
    global _face_detector
    if _face_detector is None:
        settings = get_settings()
        _face_detector = FaceDetector(
            min_detection_confidence=settings.min_face_detection_confidence,
            encoding_timeout=settings.face_encoding_timeout,
            max_workers=settings.max_process_workers
        )
    return _face_detector


async def get_head_pose_detector() -> HeadPoseDetector:
    """Get or create head pose detector singleton."""
    global _head_pose_detector
    if _head_pose_detector is None:
        settings = get_settings()
        _head_pose_detector = HeadPoseDetector(
            yaw_threshold=settings.head_pose_yaw_threshold,
            pitch_up_threshold=settings.head_pose_pitch_up_threshold,
            pitch_down_threshold=settings.head_pose_pitch_down_threshold
        )
    return _head_pose_detector


async def get_eye_tracker() -> EyeTracker:
    """Get or create eye tracker singleton."""
    global _eye_tracker
    if _eye_tracker is None:
        settings = get_settings()
        _eye_tracker = EyeTracker(
            min_confidence=settings.eye_tracking_min_confidence
        )
    return _eye_tracker


async def get_proctoring_service() -> ProctoringService:
    """
    Get proctoring service with all detectors injected.
    This is the main service dependency for routes.
    """
    settings = get_settings()
    
    # Get all detector instances
    object_detector = await get_object_detector()
    face_detector = await get_face_detector()
    head_pose_detector = await get_head_pose_detector()
    eye_tracker = await get_eye_tracker()
    
    return ProctoringService(
        settings=settings,
        object_detector=object_detector,
        face_detector=face_detector,
        head_pose_detector=head_pose_detector,
        eye_tracker=eye_tracker
    )


async def cleanup_detectors():
    """Cleanup all detector resources on shutdown."""
    global _object_detector, _face_detector, _head_pose_detector, _eye_tracker
    
    if _object_detector:
        _object_detector.cleanup()
    if _face_detector:
        _face_detector.cleanup()
    if _head_pose_detector:
        _head_pose_detector.cleanup()
    if _eye_tracker:
        _eye_tracker.cleanup()
