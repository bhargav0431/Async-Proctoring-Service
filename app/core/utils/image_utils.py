"""Image processing utilities with async support."""

import base64
import asyncio
import aiofiles
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Tuple, Optional
import cv2


async def convert_base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image asynchronously.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If base64 string is invalid
    """
    try:
        # Run CPU-bound decoding in executor
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(
            None,
            _decode_base64,
            base64_string
        )
        
        def _load_image():
            img = Image.open(BytesIO(image_data))
            img.load()  # Force load data into memory
            return img

        # Convert to PIL Image
        image = await loop.run_in_executor(
            None,
            _load_image
        )
        
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to convert base64 to image: {str(e)}")


def _decode_base64(base64_string: str) -> bytes:
    """Synchronous base64 decoding helper."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    # Ensure proper padding
    padding = len(base64_string) % 4
    if padding:
        base64_string += '=' * (4 - padding)
    
    # Decode
    return base64.b64decode(base64_string)


async def save_base64_image(base64_string: str, file_path: str) -> None:
    """
    Save base64 image to disk asynchronously.
    
    Args:
        base64_string: Base64 encoded image
        file_path: Destination file path
        
    Raises:
        ValueError: If decoding fails
        IOError: If file save fails
    """
    try:
        # Decode in executor
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(
            None,
            _decode_base64,
            base64_string
        )
        
        # Write file asynchronously
        async with aiofiles.open(file_path, 'wb') as file:
            await file.write(image_data)
            
    except Exception as e:
        raise IOError(f"Failed to save image: {str(e)}")


async def calculate_brightness(image: Image.Image) -> float:
    """
    Calculate average illuminance of an image asynchronously.
    
    Args:
        image: PIL Image object
        
    Returns:
        Average brightness value (0-255)
    """
    loop = asyncio.get_event_loop()
    
    def _calc():
        try:
            if not image.im:  # Check if image is loaded
                image.load()
            grayscale = image.convert('L')
            pixel_values = np.array(grayscale)
            return float(np.mean(pixel_values))
        except Exception:
            return 0.0
    
    brightness = await loop.run_in_executor(None, _calc)
    return brightness


async def calculate_black_pixel_ratio(image: Image.Image, black_threshold: int = 30) -> float:
    """
    Calculate ratio of black pixels in image.
    
    Args:
        image: PIL Image object
        black_threshold: RGB threshold for considering a pixel black
        
    Returns:
        Ratio of black pixels (0.0 to 1.0)
    """
    loop = asyncio.get_event_loop()
    
    def _calc():
        try:
            if not image.im:
                image.load()
            pixels = image.getdata()
            nblack = sum(
                1 for pixel in pixels
                if pixel[0] <= black_threshold 
                and pixel[1] <= black_threshold 
                and pixel[2] <= black_threshold
            )
            return nblack / float(len(pixels))
        except Exception:
            return 1.0  # Fail safe: assume worst case (all black) to trigger validation failure
    
    ratio = await loop.run_in_executor(None, _calc)
    return ratio


async def calculate_high_exposure_ratio(
    image: Image.Image,
    exposure_threshold: int = 245
) -> float:
    """
    Calculate ratio of overexposed pixels.
    
    Args:
        image: PIL Image object
        exposure_threshold: RGB threshold for overexposure
        
    Returns:
        Ratio of overexposed pixels (0.0 to 1.0)
    """
    loop = asyncio.get_event_loop()
    
    def _calc():
        try:
            if not image.im:
                image.load()
            pixels = image.getdata()
            nhigh = sum(
                1 for pixel in pixels
                if pixel[0] >= exposure_threshold
                and pixel[1] >= exposure_threshold
                and pixel[2] >= exposure_threshold
            )
            return nhigh / float(len(pixels))
        except Exception:
            return 0.0
    
    ratio = await loop.run_in_executor(None, _calc)
    return ratio


async def analyze_image_quality(
    image: Image.Image,
    brightness_threshold: int = 40,
    black_threshold: int = 30,
    black_ratio_threshold: float = 0.4,
    exposure_threshold: int = 245,
    exposure_ratio_threshold: float = 0.3
) -> Tuple[bool, Optional[str], dict]:
    """
    Analyze image quality metrics concurrently.
    
    Args:
        image: PIL Image to analyze
        brightness_threshold: Minimum acceptable brightness
        black_threshold: RGB threshold for black pixels
        black_ratio_threshold: Maximum acceptable black pixel ratio
        exposure_threshold: RGB threshold for overexposure
        exposure_ratio_threshold: Maximum acceptable overexposed ratio
        
    Returns:
        Tuple of (is_valid, error_message, metrics_dict)
    """
    try:
        # Run all calculations concurrently
        brightness_task = asyncio.create_task(calculate_brightness(image))
        black_ratio_task = asyncio.create_task(
            calculate_black_pixel_ratio(image, black_threshold)
        )
        exposure_ratio_task = asyncio.create_task(
            calculate_high_exposure_ratio(image, exposure_threshold)
        )
        
        # Wait for all tasks to complete
        brightness, black_ratio, exposure_ratio = await asyncio.gather(
            brightness_task,
            black_ratio_task,
            exposure_ratio_task
        )
        
        metrics = {
            "brightness": brightness,
            "black_pixel_ratio": black_ratio,
            "high_exposure_ratio": exposure_ratio
        }
        
        # Validate quality
        if black_ratio > black_ratio_threshold:
            return False, "Screen not clear", metrics
        
        if brightness < brightness_threshold:
            return False, "Screen not clear", metrics
        
        if exposure_ratio > exposure_ratio_threshold:
            return False, "High exposure detected", metrics
        
        return True, None, metrics
        
    except Exception as e:
        # Fallback in case of any processing error
        print(f"Image analysis failed: {e}")
        return False, "Image processing failed", {
            "brightness": 0,
            "black_pixel_ratio": 1.0,
            "high_exposure_ratio": 0.0
        }


async def convert_pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format asynchronously.
    
    Args:
        image: PIL Image object
        
    Returns:
        OpenCV image array (BGR format)
    """
    loop = asyncio.get_event_loop()
    
    def _convert():
        # Convert to RGB if needed
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Convert to numpy array
        numpy_image = np.array(rgb_image)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    
    cv2_image = await loop.run_in_executor(None, _convert)
    return cv2_image
