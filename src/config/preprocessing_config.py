# filename: preprocessing_config.py
"""Centralized configuration for all preprocessing parameters.

This module provides a single source of truth for preprocessing parameters,
enabling reproducible experiments and easy parameter tuning.

Example:
    >>> from src.config.preprocessing_config import PreprocessingConfig
    >>> config = PreprocessingConfig()
    >>> config.clahe.clip_limit
    2.0
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for normalization step.

    Attributes:
        method: Normalization method ('min_max', 'zscore').
        output_range: Target range for min_max normalization.
    """

    method: str = "min_max"
    output_range: Tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class CLAHEConfig:
    """Configuration for CLAHE enhancement.

    Attributes:
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
    """

    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class BilateralFilterConfig:
    """Configuration for bilateral filtering.

    Attributes:
        diameter: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
    """

    diameter: int = 9
    sigma_color: float = 75.0
    sigma_space: float = 75.0


@dataclass(frozen=True)
class CannyConfig:
    """Configuration for Canny edge detection.

    Attributes:
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Upper threshold for hysteresis.
        aperture_size: Aperture size for Sobel operator.
        blend_alpha: Weight of edge map in blending.
    """

    low_threshold: int = 50
    high_threshold: int = 200
    aperture_size: int = 3
    blend_alpha: float = 0.20


@dataclass
class PreprocessingConfig:
    """Master configuration for the preprocessing pipeline.

    Example:
        >>> config = PreprocessingConfig()
        >>> config.normalization.method
        'min_max'
        >>> config.bilateral.diameter
        9
    """

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    bilateral: BilateralFilterConfig = field(default_factory=BilateralFilterConfig)
    canny: CannyConfig = field(default_factory=CannyConfig)
    convert_to_grayscale: bool = True


# Default configuration matching Guha & Bhandage (2025) paper
PAPER_CONFIG = PreprocessingConfig()
