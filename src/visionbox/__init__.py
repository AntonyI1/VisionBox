"""VisionBox - AI-powered video surveillance."""

from .api import CamerasState, CameraView, PipelineState, start_api_server
from .clean_recorder import CleanRecorder
from .config import StorageConfig, VisionBoxConfig, load_config
from .database import RecordingDatabase
from .detector_v2 import (
    CLASS_PRESETS_V2,
    ModelConfig,
    MultiModelDetector,
    create_surveillance_detector,
    export_openvino,
    export_tensorrt,
)
from .kalman import KalmanBoxTracker
from .motion import MotionDetector, MotionRegion, merge_overlapping_regions
from .recorder import EventRecorder
from .recording_manager import RecordingManager
from .tracker import Tracker
from .zones import Zone, ZoneFilter

__all__ = [
    'CLASS_PRESETS_V2',
    'CameraView',
    'CamerasState',
    'CleanRecorder',
    'EventRecorder',
    'KalmanBoxTracker',
    'ModelConfig',
    'MotionDetector',
    'MotionRegion',
    'MultiModelDetector',
    'PipelineState',
    'RecordingDatabase',
    'RecordingManager',
    'StorageConfig',
    'Tracker',
    'VisionBoxConfig',
    'Zone',
    'ZoneFilter',
    'create_surveillance_detector',
    'export_openvino',
    'export_tensorrt',
    'load_config',
    'merge_overlapping_regions',
    'start_api_server',
]
