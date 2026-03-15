# filename: canny.py
"""Canny edge detection with configurable parameters and blending.

Example:
    >>> import numpy as np
    >>> from src.func.data.edge_detection.canny import detect_edges_canny
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = detect_edges_canny(img)
    >>> result.shape == img.shape
    True
"""
import cv2
import numpy as np

# =============================================================================
# Canny edge detection function
# =============================================================================

def detect_edges_canny(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 200,
    aperture_size: int = 3,
    blend_alpha: float = 0.20,
) -> np.ndarray:
    """Apply Canny edge detection and blend with original image.

    Args:
        image: Input grayscale image (uint8 or float32 in [0,1]).
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Upper threshold for hysteresis.
        aperture_size: Aperture size for Sobel operator (3, 5, or 7).
        blend_alpha: Weight of edge map in final blend (0.0 to 1.0).

    Returns:
        Blended image as float32 in [0, 1].

    Raises:
        ValueError: If blend_alpha not in [0, 1] or invalid aperture_size.
    """
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")

    if aperture_size not in (3, 5, 7):
        raise ValueError(f"aperture_size must be 3, 5, or 7, got {aperture_size}")

    image_8bit = _to_uint8(image)
    edges = cv2.Canny(
        image_8bit, low_threshold, high_threshold, apertureSize=aperture_size
    )

    image_float = _to_float32(image)
    edges_float = edges.astype(np.float32) / 255.0

    blended = cv2.addWeighted(
        image_float,
        1.0 - blend_alpha,
        edges_float,
        blend_alpha,
        0.0,
    )
    return blended


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 for OpenCV operations."""
    if image.dtype == np.uint8:
        return image
    if image.dtype in (np.float32, np.float64):
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image.astype(np.uint8)


def _to_float32(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1]."""
    if image.dtype == np.float32:
        return np.clip(image, 0, 1)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)
