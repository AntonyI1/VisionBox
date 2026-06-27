"""VisionBox - AI-powered video surveillance."""

from .kalman import KalmanBoxTracker
from .tracker import Tracker
from .detector_v2 import (
    MultiModelDetector,
    ModelConfig,
    create_surveillance_detector,
    export_tensorrt,
    export_openvino,
    CLASS_PRESETS_V2
)
from .motion import MotionDetector, MotionRegion, merge_overlapping_regions
from .recorder import EventRecorder
from .config import VisionBoxConfig, StorageConfig, load_config
from .clean_recorder import CleanRecorder
from .recording_manager import RecordingManager
from .database import RecordingDatabase
from .zones import ZoneFilter, Zone
from .api import CamerasState, CameraView, PipelineState, start_api_server

__all__ = [
    'KalmanBoxTracker', 'Tracker',
    'MultiModelDetector', 'ModelConfig', 'create_surveillance_detector',
    'export_tensorrt', 'export_openvino',
    'CLASS_PRESETS_V2',
    'MotionDetector', 'MotionRegion', 'merge_overlapping_regions',
    'EventRecorder',
    'VisionBoxConfig', 'StorageConfig', 'load_config',
    'CleanRecorder', 'RecordingManager', 'RecordingDatabase',
    'ZoneFilter', 'Zone',
    'CamerasState', 'CameraView', 'PipelineState', 'start_api_server',
]
