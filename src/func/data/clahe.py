# filename: clahe.py
"""CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement.

Example:
    >>> import numpy as np
    >>> from src.func.data.clahe import apply_clahe
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = apply_clahe(img)
    >>> result.dtype
    dtype('uint8')
"""

from typing import Tuple

import cv2
import numpy as np

# =============================================================================
# CLAHE Enhancement
# =============================================================================


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE enhancement to image.

    Args:
        image: Input grayscale image (uint8 or float32 in [0,1]).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced image as uint8.

    Raises:
        ValueError: If clip_limit is not positive.
    """
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")

    image_8bit = _to_uint8(image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_8bit)


# convert image to uint8 format
def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 format."""
    if image.dtype == np.uint8:
        return image
    if image.dtype in (np.float32, np.float64):
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image.astype(np.uint8)
