"""VisionBox - AI-powered video surveillance."""

from .detection import Detector
from .preprocessing import preprocess, letterbox
from .nms import nms, compute_iou
from .kalman import KalmanBoxTracker
from .tracker import Tracker
from .detector_v2 import (
    MultiModelDetector,
    ModelConfig,
    create_surveillance_detector,
    CLASS_PRESETS_V2
)
from .motion import MotionDetector, MotionRegion, merge_overlapping_regions

__all__ = [
    'Detector', 'preprocess', 'letterbox', 'nms', 'compute_iou',
    'KalmanBoxTracker', 'Tracker',
    'MultiModelDetector', 'ModelConfig', 'create_surveillance_detector', 'CLASS_PRESETS_V2',
    'MotionDetector', 'MotionRegion', 'merge_overlapping_regions'
]
