"""
Face Encoding Cache Module
Provides caching functionality for reference image face encodings to optimize performance.
Replaced pickle with numpy.save for security.
"""
import os
from datetime import datetime
import numpy as np
from typing import Optional


def get_cache_path(reference_image_path: str) -> str:
    """
    Generate the cache file path for a reference image.
    
    Args:
        reference_image_path: Full path to the reference image
        
    Returns:
        Path to the npy file: reference_text/{basename}_{YYYY-MM-DD}.npy
    """
    basename = os.path.splitext(os.path.basename(reference_image_path))[0]
    today = datetime.now().strftime("%Y-%m-%d")
    cache_filename = f"{basename}_{today}.npy"
    cache_path = os.path.join("reference_text", cache_filename)
    return cache_path


def load_encoding(cache_path: str) -> Optional[np.ndarray]:
    """
    Load face encoding from npy file.
    
    Args:
        cache_path: Path to the npy file
        
    Returns:
        Face encoding (numpy array) or None if loading fails
    """
    try:
        if not os.path.exists(cache_path):
            return None
            
        # load with allow_pickle=False for security
        encoding = np.load(cache_path, allow_pickle=False)
        # print(f"[CACHE DEBUG] Cache hit: Loaded encoding from {cache_path}")
        return encoding
    except Exception as e:
        print(f"[CACHE DEBUG] Cache load failed for {cache_path}: {e}")
        return None


def save_encoding(cache_path: str, encoding: np.ndarray) -> bool:
    """
    Save face encoding to npy file.
    
    Args:
        cache_path: Path to save the npy file
        encoding: Face encoding (numpy array) to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        np.save(cache_path, encoding, allow_pickle=False)
        print(f"[CACHE DEBUG] Cache saved: {cache_path}")
        return True
    except Exception as e:
        print(f"[CACHE DEBUG] Cache save failed for {cache_path}: {e}")
        return False
