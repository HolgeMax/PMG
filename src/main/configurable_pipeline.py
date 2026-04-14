# filename: configurable_pipeline.py
"""Configurable preprocessing pipeline with logging support.

This module provides a pipeline that uses centralized configuration
and logs all parameters for reproducibility.

Example:
    >>> import numpy as np
    >>> from src.main.configurable_pipeline import preprocess_image
    >>> from src.config import PreprocessingConfig
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result, log = preprocess_image(img, PreprocessingConfig())
    >>> result.shape == img.shape
    True
"""

import logging
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from src.config.preprocessing_config import PreprocessingConfig
from src.func.data.bilateral import apply_bilateral_filter
from src.func.data.clahe import apply_clahe
from src.func.data.edge_detection.canny import detect_edges_canny
from src.func.data.grayscale import convert_to_grayscale
from src.func.data.normalization.apply_norm import _apply_normalization
from src.func.utils.cfg import _config_to_dict

logger = logging.getLogger(__name__)


class PipelineLog(TypedDict):
    """Log of preprocessing parameters and steps applied."""

    config: dict
    steps_applied: list[str]
    input_shape: tuple[int, ...]
    input_dtype: str
    output_shape: tuple[int, ...]
    output_dtype: str


def _save_step(
    image: np.ndarray,
    step_name: str,
    image_name: str,
    save_dir: str,
) -> None:
    """Save an intermediate pipeline image to disk as PNG.

    Args:
        image: Current image array (uint8 or float in [0, 1]).
        step_name: Label used as the filename (e.g. '01_grayscale').
        image_name: Sub-directory name, typically the source file stem.
        save_dir: Root directory under which per-image folders are created.
    """
    out_dir = Path(save_dir) / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        img_save = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_save = image

    cv2.imwrite(str(out_dir / f"{step_name}.png"), img_save)


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig,
    edge_first: bool = False,
    image_name: str = "image",
) -> tuple[np.ndarray, PipelineLog]:
    """Apply full preprocessing pipeline with configurable parameters.

    Args:
        image: Input image (grayscale or RGB).
        config: PreprocessingConfig with all parameters.

    Returns:
        Tuple of (processed_image, pipeline_log).

    Raises:
        ValueError: If normalization method is unknown.
    """
    steps_applied: list[str] = []
    result = image.copy()

    log = PipelineLog(
        config=_config_to_dict(config),
        steps_applied=[],
        input_shape=image.shape,
        input_dtype=str(image.dtype),
        output_shape=image.shape,
        output_dtype="",
    )

    if config.save:
        _save_step(result, "00_input", image_name, config.save_dir)

    # Step 0: Grayscale conversion
    if config.convert_to_grayscale and result.ndim == 3:
        result = convert_to_grayscale(result)
        steps_applied.append("grayscale")
        logger.info("Applied grayscale conversion")
        if config.save:
            _save_step(result, "01_grayscale", image_name, config.save_dir)

    # Step 1: Normalization
    result = _apply_normalization(result, config)
    steps_applied.append(f"normalization_{config.normalization.method}")
    logger.info("Applied %s normalization", config.normalization.method)
    if config.save:
        _save_step(result, f"02_normalization_{config.normalization.method}", image_name, config.save_dir)

    # Step 2: CLAHE
    if config.clahe is not None:
        result = apply_clahe(
            result,
            clip_limit=config.clahe.clip_limit,
            tile_grid_size=config.clahe.tile_grid_size,
        )
        steps_applied.append("clahe")
        logger.info("Applied CLAHE (clip=%s)", config.clahe.clip_limit)
        if config.save:
            _save_step(result, "03_clahe", image_name, config.save_dir)

    # Step 3: Bilateral filter (and optional order swap with Canny)
    # edge_first only applies when bilateral is also active — the point is to test order, not skip bilateral
    if edge_first and config.bilateral is not None:
        result = detect_edges_canny(
            result,
            low_threshold=config.canny.low_threshold,
            high_threshold=config.canny.high_threshold,
            aperture_size=config.canny.aperture_size,
            blend_alpha=config.canny.blend_alpha,
        )
        steps_applied.append("edge_first")
        logger.info(
            "Applied edge-first Canny (thresholds=%s/%s)",
            config.canny.low_threshold,
            config.canny.high_threshold,
        )
        if config.save:
            _save_step(result, "04_edge_first_canny", image_name, config.save_dir)
    if config.bilateral is not None:
        result = apply_bilateral_filter(
            result,
            diameter=config.bilateral.diameter,
            sigma_color=config.bilateral.sigma_color,
            sigma_space=config.bilateral.sigma_space,
        )
        steps_applied.append("bilateral")
        logger.info("Applied bilateral filter (d=%s)", config.bilateral.diameter)
        if config.save:
            _save_step(result, "04_bilateral", image_name, config.save_dir)

    # Step 4: Canny edge detection
    if not edge_first:  # Only apply if not already applied as edge-first
        result = detect_edges_canny(
            result,
            low_threshold=config.canny.low_threshold,
            high_threshold=config.canny.high_threshold,
            aperture_size=config.canny.aperture_size,
            blend_alpha=config.canny.blend_alpha,
        )
        steps_applied.append("canny")
        logger.info(
            "Applied Canny (thresholds=%s/%s)",
            config.canny.low_threshold,
            config.canny.high_threshold,
        )
        if config.save:
            _save_step(result, "05_canny", image_name, config.save_dir)

    log["steps_applied"] = steps_applied
    log["output_shape"] = result.shape
    log["output_dtype"] = str(result.dtype)

    return result, log
