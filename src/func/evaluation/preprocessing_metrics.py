# filename: preprocessing_metrics.py
"""Metrics for evaluating preprocessing quality.

Provides PSNR, SSIM, and entropy metrics to quantify preprocessing effects.

Example:
    >>> import numpy as np
    >>> from src.func.evaluation.preprocessing_metrics import compute_entropy
    >>> uniform = np.full((10, 10), 128, dtype=np.uint8)
    >>> compute_entropy(uniform)
    0.0
"""

from typing import TypedDict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class PreprocessingMetrics(TypedDict):
    """Container for preprocessing evaluation metrics."""

    psnr: float
    ssim: float
    entropy_original: float
    entropy_processed: float
    entropy_change: float


def compute_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: float | None = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between images.

    Args:
        original: Reference image.
        processed: Processed/distorted image.
        data_range: Dynamic range of the images. If None, inferred from dtype.

    Returns:
        PSNR value in dB. Higher is better (less distortion).

    Raises:
        ValueError: If images have different shapes.
    """
    if original.shape != processed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {processed.shape}")

    if data_range is None:
        data_range = _infer_data_range(original)

    return float(peak_signal_noise_ratio(original, processed, data_range=data_range))


def compute_ssim(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index between images.

    Args:
        original: Reference image.
        processed: Processed image.
        data_range: Dynamic range of the images. If None, inferred from dtype.

    Returns:
        SSIM value in [0, 1]. Higher means more similar structure.

    Raises:
        ValueError: If images have different shapes.
    """
    if original.shape != processed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {processed.shape}")

    if data_range is None:
        data_range = _infer_data_range(original)

    return float(structural_similarity(original, processed, data_range=data_range))


def compute_entropy(image: np.ndarray, bins: int = 256) -> float:
    """Compute Shannon entropy of image intensity distribution.

    Higher entropy indicates more information content / contrast.

    Args:
        image: Input image.
        bins: Number of histogram bins.

    Returns:
        Entropy value in bits.
    """
    hist, _ = np.histogram(image.ravel(), bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))


def evaluate_preprocessing(
    original: np.ndarray,
    processed: np.ndarray,
) -> PreprocessingMetrics:
    """Evaluate preprocessing quality with multiple metrics.

    Args:
        original: Original image before preprocessing.
        processed: Image after preprocessing.

    Returns:
        Dictionary containing PSNR, SSIM, and entropy metrics.
    """
    orig_norm = _normalize_for_comparison(original)
    proc_norm = _normalize_for_comparison(processed)

    entropy_orig = compute_entropy(original)
    entropy_proc = compute_entropy(processed)
    return PreprocessingMetrics(
        psnr=compute_psnr(orig_norm, proc_norm, data_range=1.0),
        ssim=compute_ssim(orig_norm, proc_norm, data_range=1.0),
        entropy_original=entropy_orig,
        entropy_processed=entropy_proc,
        entropy_change=entropy_proc - entropy_orig,
    )


def _infer_data_range(image: np.ndarray) -> float:
    """Infer data range from image dtype."""
    if image.dtype == np.uint8:
        return 255.0
    if image.dtype == np.uint16:
        return 65535.0
    return 1.0


def _normalize_for_comparison(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] float32 for metric comparison."""
    img = image.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max == img_min:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)
