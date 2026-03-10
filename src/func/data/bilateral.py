# filename: bilateral.py
"""Bilateral filtering for noise reduction with edge preservation.

Example:
    >>> import numpy as np
    >>> from src.func.data.bilateral import apply_bilateral_filter
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = apply_bilateral_filter(img)
    >>> result.shape == img.shape
    True
"""
import cv2
import numpy as np


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing.

    Args:
        image: Input grayscale image.
        diameter: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Filtered image with same dtype as input.

    Raises:
        ValueError: If diameter is not positive odd number.
    """
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")

    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
