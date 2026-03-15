
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict

from src.config.preprocessing_config import PreprocessingConfig

# =============================================================================
# Config conversion utilities
# =============================================================================

def config_to_preprocessing_config(cfg: DictConfig) -> PreprocessingConfig:
    """Convert Hydra DictConfig to PreprocessingConfig dataclass."""
    from src.config.preprocessing_config import (
        BilateralFilterConfig,
        CannyConfig,
        CLAHEConfig,
        NormalizationConfig,
    )

    return PreprocessingConfig(
        normalization=NormalizationConfig(
            method=cfg.preprocessing.normalization.method,
            output_range=tuple(cfg.preprocessing.normalization.output_range),
        ),
        clahe=CLAHEConfig(
            clip_limit=cfg.preprocessing.clahe.clip_limit,
            tile_grid_size=tuple(cfg.preprocessing.clahe.tile_grid_size),
        ),
        bilateral=BilateralFilterConfig(
            diameter=cfg.preprocessing.bilateral.diameter,
            sigma_color=cfg.preprocessing.bilateral.sigma_color,
            sigma_space=cfg.preprocessing.bilateral.sigma_space,
        ),
        canny=CannyConfig(
            low_threshold=cfg.preprocessing.canny.low_threshold,
            high_threshold=cfg.preprocessing.canny.high_threshold,
            aperture_size=cfg.preprocessing.canny.aperture_size,
            blend_alpha=cfg.preprocessing.canny.blend_alpha,
        ),
        convert_to_grayscale=cfg.preprocessing.convert_to_grayscale,
    )

# =============================================================================
# Helper function to convert dataclass to dict for logging
# =============================================================================

def _config_to_dict(config: PreprocessingConfig) -> dict:
    """Convert config to serializable dict for logging."""
    return {
        "normalization": asdict(config.normalization),
        "clahe": asdict(config.clahe),
        "bilateral": asdict(config.bilateral),
        "canny": asdict(config.canny),
        "convert_to_grayscale": config.convert_to_grayscale,
    }