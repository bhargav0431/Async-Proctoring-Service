"""
Face detection and matching with async support.
Refactored from main.py with professional architecture.
"""

import asyncio
import concurrent.futures
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from typing import Tuple, Optional, List
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)

@dataclass
class FaceExtractionResult:
    """Result of face extraction from image."""
    face_image: Optional[np.ndarray]
    brightness: float
    screen_distance: Optional[float]
    message: str
    success: bool


@dataclass
class FaceMatchResult:
    """Result of face matching comparison."""
    is_match: bool
    message: str
    face_brightness: float
    face_distance: float
    error_code: int
    screen_distance: Optional[float]


class FaceDetector:
    """
    Face detection and matching with MediaPipe and face_recognition.
    Uses process pool for CPU-intensive encoding operations.
    """
    
    # Screen distance calculation disabled
    # KNOWN_FACE_WIDTH_CM = 14.0
    # FOCAL_LENGTH_PX = 650
    # DISTANCE_OFFSET_CM = 10
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        encoding_timeout: int = 9,
        max_workers: int = 2
    ):
        """
        Initialize face detector.
        
        Args:
            min_detection_confidence: Minimum face detection confidence
            encoding_timeout: Timeout for face encoding operations
            max_workers: Maximum process pool workers
        """
        self.min_detection_confidence = min_detection_confidence
        self.encoding_timeout = encoding_timeout
        
        # ThreadPoolExecutor for face encoding (face_recognition releases GIL)
        # Using threads instead of processes fixes Windows compatibility issues
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        logger.info("✅ FaceDetector initialized. Relies on Dlib CUDA for GPU processing when encoding faces.")
    
    async def extract_face(self, image_path: str) -> FaceExtractionResult:
        """
        Extract face from image with padding and distance calculation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            FaceExtractionResult with extracted face data
        """
        loop = asyncio.get_event_loop()
        
        # Run extraction in executor
        result = await loop.run_in_executor(
            None,
            self._extract_face_sync,
            image_path
        )
        
        return result
    
    def _extract_face_sync(self, image_path: str) -> FaceExtractionResult:
        """Synchronous face extraction logic."""
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"[FACE DEBUG] ❌ Failed to read image: {image_path}")
            return FaceExtractionResult(
                face_image=None,
                brightness=0.0,
                screen_distance=None,
                message="Not available - Image read error",
                success=False
            )
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence
        ) as face_detection:
            results = face_detection.process(rgb_img)
            
            # print(f"\n[FACE DEBUG] Extracting from: {image_path}")
            
            if not results.detections:
                print(f"[FACE DEBUG] ❌ No faces detected in: {image_path}")
                return FaceExtractionResult(
                    face_image=None,
                    brightness=0.0,
                    screen_distance=None,
                    message="Not available - No faces detected",
                    success=False
                )
            
            # print(f"[FACE DEBUG] Found {len(results.detections)} faces")
            
            if len(results.detections) > 1:
                print(f"[FACE DEBUG] ❌ Multiple faces detected")
                return FaceExtractionResult(
                    face_image=None,
                    brightness=0.0,
                    screen_distance=None,
                    message="Not available - More than 1 face detected",
                    success=False
                )
            
            # Extract face with padding
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            x_max = int((bboxC.xmin + bboxC.width) * iw)
            y_max = int((bboxC.ymin + bboxC.height) * ih)
            
            # Add 20% padding
            padding_ratio = 0.2
            pad_x = int((x_max - x_min) * padding_ratio)
            pad_y = int((y_max - y_min) * padding_ratio)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(iw, x_max + pad_x)
            y_max = min(ih, y_max + pad_y)
            
            # Extract face region
            face_region = img[y_min:y_max, x_min:x_max]
            # print(f"[FACE DEBUG] Face region: {x_min},{y_min} to {x_max},{y_max} (Shape: {face_region.shape})")
            
            if face_region.shape[0] == 0 or face_region.shape[1] == 0:
                print(f"[FACE DEBUG] ❌ Invalid face region dimensions")
                return FaceExtractionResult(
                    face_image=None,
                    brightness=0.0,
                    screen_distance=None,
                    message="Not available - Invalid face region",
                    success=False
                )
            
            # Calculate brightness
            try:
                f_brightness = float(np.mean(face_region))
            except:
                f_brightness = 0.0
            
            # Calculate distance - DISABLED
            # try:
            #     perceived_width = x_max - x_min
            #     screen_distance = (
            #         self.KNOWN_FACE_WIDTH_CM * self.FOCAL_LENGTH_PX
            #     ) / perceived_width + self.DISTANCE_OFFSET_CM
            # except:
            #     screen_distance = None
            screen_distance = None  # Distance calculation disabled
            
            # Resize for encoding
            face_resized = cv2.resize(face_region, (200, 200))
            
            return FaceExtractionResult(
                face_image=face_resized,
                brightness=f_brightness,
                screen_distance=screen_distance,
                message="Success",
                success=True
            )
    
    async def compare_faces(
        self,
        reference_image_path: str,
        input_image_path: str,
        threshold: float = 0.5
    ) -> FaceMatchResult:
        """
        Compare two face images asynchronously.
        
        Args:
            reference_image_path: Path to reference image
            input_image_path: Path to input image to match
            threshold: Matching threshold (lower = more strict)
            
        Returns:
            FaceMatchResult with matching details
        """
        # ====== REFERENCE IMAGE PROCESSING WITH CACHING ======
        # Import cache module locally to avoid circular imports if any
        from app.core.utils import face_encoding_cache
        
        cache_path = face_encoding_cache.get_cache_path(reference_image_path)
        face1_encoding = face_encoding_cache.load_encoding(cache_path)
        
        # If cache miss, we need to extract and encode
        if face1_encoding is None:
            print(f"CACHE MISS: Loading and encoding reference image from file: {reference_image_path}")
            ref_extraction_task = asyncio.create_task(self.extract_face(reference_image_path))
            input_extraction_task = asyncio.create_task(self.extract_face(input_image_path))
            
            ref_result, input_result = await asyncio.gather(ref_extraction_task, input_extraction_task)
            
            # Check extraction success
            if not ref_result.success or not input_result.success:
                return FaceMatchResult(
                    is_match=False,
                    message="Invalid face data extracted",
                    face_brightness=input_result.brightness if input_result else 0.0,
                    face_distance=0.0,
                    error_code=10,
                    screen_distance=input_result.screen_distance if input_result else None
                )
                
            # Encode Reference Image (and save to cache)
            try:
                loop = asyncio.get_event_loop()
                # Run encoding in executor
                enc1_future = loop.run_in_executor(self.executor, self._encode_task, ref_result.face_image, False)
                
                try:
                    face1_encoding_list = await asyncio.wait_for(enc1_future, timeout=self.encoding_timeout)
                except asyncio.TimeoutError:
                     # Try enhanced if timeout/fail
                    enc1_enhanced = loop.run_in_executor(self.executor, self._encode_task, ref_result.face_image, True)
                    face1_encoding_list = await asyncio.wait_for(enc1_enhanced, timeout=self.encoding_timeout)

                if not face1_encoding_list:
                     # One last try with enhancement if empty
                    enc1_enhanced = loop.run_in_executor(self.executor, self._encode_task, ref_result.face_image, True)
                    face1_encoding_list = await asyncio.wait_for(enc1_enhanced, timeout=self.encoding_timeout)

                if not face1_encoding_list:
                    return FaceMatchResult(
                        is_match=False,
                        message="Face encoding failed (Reference)",
                        face_brightness=input_result.brightness,
                        face_distance=0.0,
                        error_code=5,
                        screen_distance=input_result.screen_distance
                    )
                
                face1_encoding = face1_encoding_list[0]
                # Save to cache
                face_encoding_cache.save_encoding(cache_path, face1_encoding)
                
            except Exception as e:
                return FaceMatchResult(
                    is_match=False,
                    message=f"Face encoding error: {str(e)}",
                    face_brightness=input_result.brightness,
                    face_distance=0.0,
                    error_code=5,
                    screen_distance=input_result.screen_distance
                )
        else:
             # Cache Hit! Only need to process input image
            print(f"CACHE HIT: Using cached encoding for: {reference_image_path}")
            input_result = await self.extract_face(input_image_path)
            
            if not input_result.success:
                return FaceMatchResult(
                    is_match=False,
                    message=input_result.message,
                    face_brightness=input_result.brightness,
                    face_distance=0.0,
                    error_code=10,
                    screen_distance=input_result.screen_distance
                )

        # ====== INPUT IMAGE PROCESSING (NO CACHING) ======
        # At this point we have face1_encoding (ref) and input_result (test image extracted)
        
        try:
            loop = asyncio.get_event_loop()
            
            # Encode Input Image
            enc2_future = loop.run_in_executor(self.executor, self._encode_task, input_result.face_image, False)
            
            try:
                face2_encoding_list = await asyncio.wait_for(enc2_future, timeout=self.encoding_timeout)
            except asyncio.TimeoutError:
                 # Try enhanced
                enc2_enhanced = loop.run_in_executor(self.executor, self._encode_task, input_result.face_image, True)
                face2_encoding_list = await asyncio.wait_for(enc2_enhanced, timeout=self.encoding_timeout)

            if not face2_encoding_list:
                 # Try enhanced
                enc2_enhanced = loop.run_in_executor(self.executor, self._encode_task, input_result.face_image, True)
                face2_encoding_list = await asyncio.wait_for(enc2_enhanced, timeout=self.encoding_timeout)
                
            if not face2_encoding_list:
                return FaceMatchResult(
                    is_match=False,
                    message="Face encoding failed (Input)",
                    face_brightness=input_result.brightness,
                    face_distance=0.0,
                    error_code=5,
                    screen_distance=input_result.screen_distance
                )
            
            face2_encoding = face2_encoding_list[0]
            
            # ====== COMPARE ======
            face_distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
            
            # Retry with enhancement if distance is high (borderline match)
            if face_distance > 0.5:
                # print("High distance, retrying input with enhancement...")
                enc2_enhanced = loop.run_in_executor(self.executor, self._encode_task, input_result.face_image, True)
                face2_encoding_list_enh = await asyncio.wait_for(enc2_enhanced, timeout=self.encoding_timeout)
                
                if face2_encoding_list_enh:
                     dist_enh = face_recognition.face_distance([face1_encoding], face2_encoding_list_enh[0])[0]
                     if dist_enh < face_distance:
                         face_distance = dist_enh
            
            is_match = face_distance < threshold
            
            return FaceMatchResult(
                is_match=is_match,
                message="Photo Matched" if is_match else "Not Matched",
                face_brightness=input_result.brightness,
                face_distance=float(face_distance),
                error_code=1 if is_match else 2,
                screen_distance=input_result.screen_distance
            )

        except Exception as e:
            return FaceMatchResult(
                is_match=False,
                message=f"Comparison failed: {str(e)}",
                face_brightness=input_result.brightness,
                face_distance=0.0,
                error_code=5,
                screen_distance=input_result.screen_distance
            )
    
    async def _encode_and_compare(
        self,
        face1: np.ndarray,
        face2: np.ndarray,
        threshold: float
    ) -> dict:
        """Encode and compare faces with timeout."""
        # print(f"\n[FACE DEBUG] Comparison Started: Handing off to Process Pool...")
        loop = asyncio.get_event_loop()
        
        # Try direct encoding first
        # print(f"  - Starting encoding tasks (Timeout: {self.encoding_timeout}s)...")
        enc1_future = loop.run_in_executor(self.executor, self._encode_task, face1, False)
        enc2_future = loop.run_in_executor(self.executor, self._encode_task, face2, False)
        
        try:
            enc1_task = asyncio.create_task(
                asyncio.wait_for(enc1_future, timeout=self.encoding_timeout)
            )
            enc2_task = asyncio.create_task(
                asyncio.wait_for(enc2_future, timeout=self.encoding_timeout)
            )
            
            face1_encoding, face2_encoding = await asyncio.gather(enc1_task, enc2_task)
            print(f"  - Base encoding complete")
            
        except asyncio.TimeoutError:
            print(f"  - ⚠️ Timeout reached, attempting enhanced encoding...")
            # Try with enhancement
            enc1_enhanced = loop.run_in_executor(
                self.executor, self._encode_task, face1, True
            )
            enc2_enhanced = loop.run_in_executor(
                self.executor, self._encode_task, face2, True
            )
            
            face1_encoding, face2_encoding = await asyncio.gather(
                asyncio.wait_for(enc1_enhanced, timeout=self.encoding_timeout),
                asyncio.wait_for(enc2_enhanced, timeout=self.encoding_timeout)
            )
            print(f"  - Enhanced encoding complete")
        
        # Check encoding success
        if not face1_encoding or not face2_encoding:
            print(f"[FACE DEBUG] ❌ Face encoding failed (Ref: {'Yes' if face1_encoding else 'No'}, Input: {'Yes' if face2_encoding else 'No'})")
            return {
                "match": False,
                "message": "Face encoding failed",
                "distance": 0.0,
                "error_code": 5
            }
        
        print(f"[FACE DEBUG] ✅ Both faces encoded successfully")
        
        # Calculate distance
        try:
            face_distance = face_recognition.face_distance(
                [face1_encoding[0]], face2_encoding[0]
            )[0]
        except Exception as e:
            return {
                "match": False,
                "message": f"Distance calculation failed: {str(e)}",
                "distance": 0.0,
                "error_code": 5
            }
        
        # Determine match
        is_match = face_distance < threshold
        
        print(f"[FACE DEBUG] Result: {'✅ MATCH' if is_match else '❌ NO MATCH'} (Dist: {face_distance:.4f}, Thresh: {threshold})")
        print("="*30 + "\n")
        
        return {
            "match": is_match,
            "message": "Photo Matched" if is_match else "Not Matched",
            "distance": float(face_distance),
            "error_code": 1 if is_match else 2
        }
    
    @staticmethod
    def _encode_task(face: np.ndarray, enhance: bool = False) -> List:
        """
        Encode face image. Runs in process pool.
        
        Args:
            face: Face image array
            enhance: Whether to enhance image before encoding
            
        Returns:
            List of face encodings
        """
        try:
            if enhance:
                face = FaceDetector._enhance_face(face)
            
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            return face_recognition.face_encodings(rgb)
        except Exception:
            return []
    
    @staticmethod
    def _enhance_face(face: np.ndarray) -> np.ndarray:
        """Enhance face image for better encoding."""
        # Upscale
        upscaled = cv2.resize(face, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Sharpen
        sharpening_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(upscaled, -1, sharpening_kernel)
        
        # CLAHE for contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def cleanup(self):
        """Shutdown process pool."""
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass


async def create_face_detector(
    min_detection_confidence: float = 0.5,
    encoding_timeout: int = 9
) -> FaceDetector:
    """
    Factory function to create FaceDetector.
    
    Args:
        min_detection_confidence: Minimum detection confidence
        encoding_timeout: Encoding timeout in seconds
        
    Returns:
        Initialized FaceDetector instance
    """
    return FaceDetector(
        min_detection_confidence=min_detection_confidence,
        encoding_timeout=encoding_timeout
    )
