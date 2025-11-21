"""
YOLOv2-PyTorch
A clean, modular implementation of YOLOv2 in PyTorch
"""

__version__ = '2.0.0'

from .models import YOLOv2, create_yolov2, Darknet19Improved
from .data import datasets
from .utils import loss, general

__all__ = [
    'YOLOv2',
    'create_yolov2',
    'Darknet19Improved',
    '__version__'
]
