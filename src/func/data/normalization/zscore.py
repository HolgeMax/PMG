# filename: zscore.py
"""Z-score normalization for MRI images.

Standardizes pixel intensities to zero mean and unit variance.
More robust than min-max for MRI where intensity is relative.

Example:
    >>> import numpy as np
    >>> from src.func.data.normalization.zscore import normalize_zscore
    >>> img = np.array([[10, 20], [30, 40]], dtype=np.float32)
    >>> result = normalize_zscore(img)
    >>> abs(float(result.mean())) < 0.01
    True
"""

import numpy as np

# =============================================================================
# Z-score normalization function
# =============================================================================


def normalize_zscore(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize image using z-score (zero mean, unit variance).

    Args:
        image: Input image as numpy array.
        mask: Optional binary mask to compute statistics only from
            non-zero regions (useful for brain MRI with background).

    Returns:
        Z-score normalized image as float32 array.

    Raises:
        ValueError: If masked region has zero standard deviation.
    """
    image_float = image.astype(np.float32)

    if mask is not None:
        valid_pixels = image_float[mask > 0]
    else:
        valid_pixels = image_float.ravel()

    mean = float(np.mean(valid_pixels))
    std = float(np.std(valid_pixels))

    if std < 1e-8:
        raise ValueError("Image has zero variance; cannot z-score normalize.")

    return (image_float - mean) / std
