import numpy as np

from src.config.preprocessing_config import PreprocessingConfig
from src.func.data.normalization.min_max import normalize_min_max
from src.func.data.normalization.zscore import normalize_zscore

# =============================================================================
# Normalization
# =============================================================================


def _apply_normalization(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply configured normalization method."""
    method = config.normalization.method
    if method == "min_max":
        return normalize_min_max(image, config.normalization.output_range)
    if method == "zscore":
        return normalize_zscore(image)
    raise ValueError(f"Unknown normalization method: {method}")
