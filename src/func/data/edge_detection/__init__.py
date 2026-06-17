# filename: __init__.py
"""Edge detection methods for MRI preprocessing."""

from src.func.data.edge_detection.canny import detect_edges_canny

__all__ = ["detect_edges_canny"]
