"""
Stage 1: Input Processing & Depth Estimation
Handles video/image input and generates depth maps using MiDaS
"""

from .processor import InputProcessor
from .depth import DepthEstimator

__all__ = ['InputProcessor', 'DepthEstimator']
# boht piyarai baat ha jo __all__ it means publicly exposed to all , as this part is totally exposed to all as in this thre is we defined the two classes from the othere files in the same folder (as we have imported it above 2 lines)
