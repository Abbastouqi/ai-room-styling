"""
Stage 2: Object Detection & Segmentation
Author: [Team Member Name] - Restructured for architecture compliance
"""

from .detector import ObjectDetector
from .segmentation import ObjectSegmenter
from .inventory import InventoryManager

__all__ = ['ObjectDetector', 'ObjectSegmenter', 'InventoryManager']