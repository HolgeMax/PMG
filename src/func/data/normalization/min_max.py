# filename: min_max.py
"""Min-max normalization for MRI images.

Rescales pixel intensities to a specified range, typically [0, 1].

Example:
    >>> import numpy as np
    >>> from src.func.data.normalization.min_max import normalize_min_max
    >>> img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    >>> result = normalize_min_max(img)
    >>> float(result.min()), float(result.max())
    (0.0, 1.0)
"""
from typing import Tuple

import numpy as np


def normalize_min_max(
    image: np.ndarray,
    output_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Normalize image to specified range using min-max scaling.

    Args:
        image: Input image as numpy array.
        output_range: Target (min, max) range for output values.

    Returns:
        Normalized image as float32 array.

    Raises:
        ValueError: If output_range[0] >= output_range[1].
    """
    out_min, out_max = output_range
    if out_min >= out_max:
        raise ValueError(f"Invalid output_range: {output_range}")

    image_min = float(np.min(image))
    image_max = float(np.max(image))

    if image_max == image_min:
        return np.full_like(image, out_min, dtype=np.float32)

    normalized = (image.astype(np.float32) - image_min) / (image_max - image_min)
    scaled = normalized * (out_max - out_min) + out_min

    return scaled.astype(np.float32)
