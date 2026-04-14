from omegaconf import DictConfig
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

    pre = cfg.preprocessing
    norm_cfg = pre.get("normalization", None)
    clahe_cfg = pre.get("clahe", None)
    bilateral_cfg = pre.get("bilateral", None)
    canny_cfg = pre.get("canny", None)

    return PreprocessingConfig(
        normalization=NormalizationConfig(
            method=norm_cfg.method,
            output_range=tuple(norm_cfg.output_range),
        )
        if norm_cfg is not None
        else NormalizationConfig(),
        clahe=CLAHEConfig(
            clip_limit=clahe_cfg.clip_limit,
            tile_grid_size=tuple(clahe_cfg.tile_grid_size),
        )
        if clahe_cfg is not None
        else None,
        bilateral=BilateralFilterConfig(
            diameter=bilateral_cfg.diameter,
            sigma_color=bilateral_cfg.sigma_color,
            sigma_space=bilateral_cfg.sigma_space,
        )
        if bilateral_cfg is not None
        else None,
        canny=CannyConfig(
            low_threshold=canny_cfg.low_threshold,
            high_threshold=canny_cfg.high_threshold,
            aperture_size=canny_cfg.aperture_size,
            blend_alpha=canny_cfg.blend_alpha,
        )
        if canny_cfg is not None
        else CannyConfig(),
        convert_to_grayscale=pre.get("convert_to_grayscale", True),
        save=pre.get("save", False),
        save_dir=pre.get("save_dir", "results/preprocessing_debug"),
    )


# =============================================================================
# Helper function to convert dataclass to dict for logging
# =============================================================================


def _config_to_dict(config: PreprocessingConfig) -> dict:
    """Convert config to serializable dict for logging."""
    return {
        "normalization": asdict(config.normalization),
        "clahe": asdict(config.clahe) if config.clahe is not None else None,
        "bilateral": asdict(config.bilateral) if config.bilateral is not None else None,
        "canny": asdict(config.canny),
        "convert_to_grayscale": config.convert_to_grayscale,
        "save": config.save,
        "save_dir": config.save_dir,
    }
