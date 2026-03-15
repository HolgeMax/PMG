# filename: dog.py
"""Difference of Gaussians (DoG) edge detection.

Alternative to Canny that may be more suitable for neuroimaging
as it produces closed loops even in noisy images.

Example:
    >>> import numpy as np
    >>> from src.func.data.edge_detection.dog import detect_edges_dog
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = detect_edges_dog(img)
    >>> result.shape == img.shape
    True
"""
import cv2
import numpy as np

# =============================================================================
# DoG Edge Detection with Blending
# =============================================================================

def detect_edges_dog(
    image: np.ndarray,
    sigma1: float = 1.0,
    sigma2: float = 2.0,
    blend_alpha: float = 0.20,
) -> np.ndarray:
    """Apply Difference of Gaussians edge detection with blending.

    DoG approximates the Laplacian of Gaussian and produces edges
    as zero-crossings. Particularly useful for cortical features.

    Args:
        image: Input grayscale image (uint8 or float32 in [0,1]).
        sigma1: Standard deviation of first Gaussian (smaller).
        sigma2: Standard deviation of second Gaussian (larger).
        blend_alpha: Weight of edge map in final blend.

    Returns:
        Blended image as float32 in [0, 1].

    Raises:
        ValueError: If sigma1 >= sigma2 or blend_alpha not in [0, 1].
    """
    if sigma1 >= sigma2:
        raise ValueError(f"sigma1 must be < sigma2, got {sigma1} >= {sigma2}")

    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")

    image_float = _to_float32(image)

    ksize1 = _sigma_to_ksize(sigma1)
    ksize2 = _sigma_to_ksize(sigma2)

    blur1 = cv2.GaussianBlur(image_float, (ksize1, ksize1), sigma1)
    blur2 = cv2.GaussianBlur(image_float, (ksize2, ksize2), sigma2)

    dog = blur1 - blur2
    edges = np.abs(dog)
    edges = edges / (float(edges.max()) + 1e-8)

    blended = cv2.addWeighted(
        image_float,
        1.0 - blend_alpha,
        edges.astype(np.float32),
        blend_alpha,
        0.0,
    )
    return blended

# =============================================================================
# Helper functions
# =============================================================================

def _sigma_to_ksize(sigma: float) -> int:
    """Convert sigma to appropriate kernel size (must be odd)."""
    ksize = int(np.ceil(sigma * 6)) | 1
    return max(3, ksize)


def _to_float32(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1]."""
    if image.dtype == np.float32:
        return np.clip(image, 0, 1)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)
