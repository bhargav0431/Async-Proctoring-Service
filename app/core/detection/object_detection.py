"""
Object detection using MediaPipe with async support.
Refactored from main.py with professional architecture.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import asyncio
import mediapipe as mp
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectionCategory:
    """Single detection category result."""
    name: str
    score: float
    count: int = 1


@dataclass
class ObjectDetectionResult:
    """Complete object detection result."""
    categories: Dict[str, DetectionCategory] = field(default_factory=dict)
    device_flag: int = 0
    person_count: int = 0
    has_detection: bool = False
    raw_detections: List[str] = field(default_factory=list)


class ObjectDetector:
    """
    MediaPipe-based object detector with async support.
    Thread-safe for concurrent requests.
    """
    
    ACCEPTED_OBJECTS = ["book", "person", "cell phone", "laptop"]
    DEVICES = ["book", "cell phone", "laptop"]
    
    def __init__(self, model_path: str, max_results: int = 5, score_threshold: float = 0.2):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to EfficientDet model file
            max_results: Maximum number of detections
            score_threshold: Minimum confidence score
        """
        self.model_path = model_path
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.frame_counter = 0
        
        # Initialize MediaPipe components
        self.BaseOptions = mp.tasks.BaseOptions
        self.ObjectDetector = mp.tasks.vision.ObjectDetector
        self.ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Configure GPU Delegate if possible
        logger.info("Initializing Object Detector GPU settings...")
        try:
            base_options = self.BaseOptions(
                model_asset_path=model_path,
                delegate=self.BaseOptions.Delegate.GPU
            )
            logger.info("✅ MediaPipe Object Detection initialized with GPU Delegate")
        except Exception as e:
            logger.warning(f"❌ Failed to set MediaPipe GPU Delegate. Falling back to CPU: {e}")
            base_options = self.BaseOptions(model_asset_path=model_path)
        
        # Create options
        options = self.ObjectDetectorOptions(
            base_options=base_options,
            max_results=max_results,
            running_mode=self.VisionRunningMode.IMAGE,
            score_threshold=score_threshold
        )
        
        # Initialize detector
        self.detector = self.ObjectDetector.create_from_options(options)
        self.lock = asyncio.Lock()
    
    async def detect_objects(
        self,
        image_path: str,
        brightness: float
    ) -> ObjectDetectionResult:
        """
        Detect objects in image asynchronously.
        
        Args:
            image_path: Path to image file
            brightness: Average image brightness
            
        Returns:
            ObjectDetectionResult with detected categories
        """
        async with self.lock:
            loop = asyncio.get_event_loop()
            
            # Run detection in executor (CPU-bound)
            result = await loop.run_in_executor(
                None,
                self._detect_sync,
                image_path,
                brightness
            )
            
            return result
    
    def _detect_sync(self, image_path: str, brightness: float) -> ObjectDetectionResult:
        """Synchronous detection logic."""
        # Create MediaPipe image
        mp_image = mp.Image.create_from_file(image_path)
        
        # Perform detection (Stateless IMAGE mode)
        detection_result = self.detector.detect(mp_image)
        
        # Process results
        return self._process_detections(detection_result.detections, brightness)
    
    def _process_detections(
        self,
        detections: List,
        brightness: float
    ) -> ObjectDetectionResult:
        """
        Process raw detections into structured result.
        
        Args:
            detections: Raw MediaPipe detections
            brightness: Image brightness value
            
        Returns:
            Structured ObjectDetectionResult
        """
        result = ObjectDetectionResult()
        category_scores: Dict[str, float] = {}
        category_counts: Dict[str, int] = {}
        
        for detection in detections:
            for category in detection.categories:
                name = category.category_name
                score = category.score
                
                # Filter accepted objects
                if name in self.ACCEPTED_OBJECTS:
                    # Person needs higher threshold
                    if (name == "person" and score > 0.4) or name != 'person':
                        # Track max score for each category
                        if name in category_scores:
                            category_scores[name] = max(category_scores[name], score)
                            category_counts[name] += 1
                        else:
                            category_scores[name] = score
                            category_counts[name] = 1
                
                # Device detection flags
                if name in self.DEVICES:
                    if score > 0.5:
                        result.device_flag = 1
                    elif 0.2 < score < 0.5 and result.device_flag == 0:
                        result.device_flag = 2
        
        # Build category results
        for name, score in category_scores.items():
            count = category_counts.get(name, 0)
            result.categories[name] = DetectionCategory(
                name=name,
                score=score,
                count=count
            )
            
            # Add to raw detections for logging
            result.raw_detections.append(
                f"Category Name: {name}, Total Score: {score}, Count: {count}, Brightness: {brightness}"
            )
            
            # Track person count
            if name == "person":
                result.person_count = count
        
        result.has_detection = len(result.categories) > 0
        
        return result
    
    def cleanup(self):
        """Release detector resources."""
        try:
            if hasattr(self, 'detector'):
                self.detector.close()
        except Exception:
            pass


async def create_object_detector(model_path: str) -> ObjectDetector:
    """
    Factory function to create ObjectDetector asynchronously.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Initialized ObjectDetector instance
    """
    loop = asyncio.get_event_loop()
    
    # Initialize in executor to avoid blocking
    detector = await loop.run_in_executor(
        None,
        lambda: ObjectDetector(model_path)
    )
    
    return detector
