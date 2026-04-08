"""
Configuration management using Pydantic Settings.
Environment variables override default values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server Configuration
    app_name: str = "Proctoring Microservice"
    app_version: str = "2.0.0"
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=5030, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    workers: int = Field(default=4, env="WORKERS")
    
    # CORS Configuration
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # Model Paths (relative to project root)
    efficientdet_model_path: str = Field(
        default="efficientdet_lite0.tflite",
        env="EFFICIENTDET_MODEL_PATH"
    )
    ssd_model_path: str = Field(
        default="ssd_mobilenet_v2_coco_2018_03_29",
        env="SSD_MODEL_PATH"
    )
    frozen_graph_path: str = Field(
        default="frozen_inference_graph.pb",
        env="FROZEN_GRAPH_PATH"
    )
    
    # Detection Thresholds
    brightness_threshold: int = Field(default=40, env="BRIGHTNESS_THRESHOLD")
    black_pixel_threshold: float = Field(default=0.4, env="BLACK_PIXEL_THRESHOLD")
    black_thresh_rgb: int = Field(default=30, env="BLACK_THRESH_RGB")
    high_exposure_threshold: int = Field(default=245, env="HIGH_EXPOSURE_THRESHOLD")
    high_exposure_percentage: float = Field(default=30.0, env="HIGH_EXPOSURE_PERCENTAGE")
    face_match_threshold: float = Field(default=0.55, env="FACE_MATCH_THRESHOLD")
    detection_confidence: float = Field(default=0.5, env="DETECTION_CONFIDENCE")
    face_brightness_threshold: int = Field(default=40, env="FACE_BRIGHTNESS_THRESHOLD")
    
    # MediaPipe Configuration
    mediapipe_max_results: int = Field(default=5, env="MEDIAPIPE_MAX_RESULTS")
    mediapipe_score_threshold: float = Field(default=0.2, env="MEDIAPIPE_SCORE_THRESHOLD")
    
    # Face Detection Configuration
    min_face_detection_confidence: float = Field(default=0.5, env="MIN_FACE_DETECTION_CONFIDENCE")
    face_encoding_timeout: int = Field(default=9, env="FACE_ENCODING_TIMEOUT")
    
    # Head Pose Configuration
    head_pose_yaw_threshold: int = Field(default=37, env="HEAD_POSE_YAW_THRESHOLD")
    head_pose_pitch_up_threshold: int = Field(default=40, env="HEAD_POSE_PITCH_UP_THRESHOLD")
    head_pose_pitch_down_threshold: int = Field(default=-37, env="HEAD_POSE_PITCH_DOWN_THRESHOLD")
    
    # Eye Tracking Configuration
    eye_tracking_min_confidence: float = Field(default=0.6, env="EYE_TRACKING_MIN_CONFIDENCE")
    closed_eye_threshold: float = Field(default=0.20, env="CLOSED_EYE_THRESHOLD")
    
    # Distance Calculation
    known_face_width_cm: float = Field(default=14.0, env="KNOWN_FACE_WIDTH_CM")
    focal_length_px: int = Field(default=650, env="FOCAL_LENGTH_PX")
    distance_offset_cm: int = Field(default=10, env="DISTANCE_OFFSET_CM")
    valid_min_distance_cm: int = Field(default=30, env="VALID_MIN_DISTANCE_CM")
    valid_max_distance_cm: int = Field(default=105, env="VALID_MAX_DISTANCE_CM")
    
    # Storage Paths
    image_storage_path: str = Field(default="img", env="IMAGE_STORAGE_PATH")
    reference_image_path: str = Field(default="reference_img", env="REFERENCE_IMAGE_PATH")
    cache_directory: str = Field(default="cache_directory", env="CACHE_DIRECTORY")
    
    # Process Pool Configuration
    max_process_workers: int = Field(default=2, env="MAX_PROCESS_WORKERS")
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function for settings."""
    return settings
