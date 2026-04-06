# filename: grayscale.py
"""Grayscale conversion for MRI images.

Example:
    >>> import numpy as np
    >>> from src.func.data.grayscale import convert_to_grayscale
    >>> rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> gray = convert_to_grayscale(rgb)
    >>> gray.shape
    (64, 64)
"""

import cv2
import numpy as np
# =============================================================================
# Grayscale Conversion
# =============================================================================


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed.

    Args:
        image: Input image (grayscale or BGR).

    Returns:
        Grayscale image as 2D array.

    Raises:
        ValueError: If image has unsupported number of dimensions.
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    raise ValueError(f"Unsupported image shape: {image.shape}")
