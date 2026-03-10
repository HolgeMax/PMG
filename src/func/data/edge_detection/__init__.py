# filename: __init__.py
"""Edge detection methods for MRI preprocessing."""
from src.func.data.edge_detection.canny import detect_edges_canny
from src.func.data.edge_detection.dog import detect_edges_dog

__all__ = ["detect_edges_canny", "detect_edges_dog"]
