"""YAML configuration with environment variable resolution."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    url: str = ''
    test_input: str = ''  # Path to video file for testing without camera


@dataclass
class DetectionConfig:
    mode: str = 'outdoor'
    confidence: float = 0.3
    detect_fps: int = 5
    model: str = 'yolov8n.pt'
    imgsz: int = 640
    class_conf: dict = field(default_factory=lambda: {
        0: 0.35,   # person
        2: 0.45,   # car
        5: 0.50,   # bus
        7: 0.50,   # truck
        14: 0.50,  # bird
        15: 0.40,  # cat
        16: 0.40,  # dog
    })


@dataclass
class MotionConfig:
    min_area: int = 2000


@dataclass
class TrackerConfig:
    max_age: int = 75
    min_hits: int = 2
    iou_threshold: float = 0.2
    max_coast: int = 15


@dataclass
class CleanRecordingConfig:
    enabled: bool = True


@dataclass
class AnnotatedRecordingConfig:
    enabled: bool = True
    fps: float = 15.0
    cooldown: float = 10.0


@dataclass
class RetentionConfig:
    days: int = 14
    check_interval: int = 3600  # seconds


@dataclass
class RecordingConfig:
    output_dir: str = 'recordings'
    clean: CleanRecordingConfig = field(default_factory=CleanRecordingConfig)
    annotated: AnnotatedRecordingConfig = field(default_factory=AnnotatedRecordingConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)


@dataclass
class CaptureConfig:
    interval: float = 10.0
    uncertain_low: float = 0.3
    uncertain_high: float = 0.6
    uncertain_interval: float = 5.0


@dataclass
class DisplayConfig:
    web: bool = True
    web_port: int = 8085
    max_fps: int = 15


@dataclass
class VisionBoxConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, '')
    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _resolve_recursive(obj):
    """Walk a dict/list and resolve env vars in all string values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(item) for item in obj]
    return obj


def _apply_dict(dc, data: dict):
    """Apply a flat dict to a dataclass, handling nested dataclasses."""
    for key, value in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | Path = 'config.yml') -> VisionBoxConfig:
    """Load configuration from YAML file.

    Resolves ${ENV_VAR} patterns from environment.
    Falls back to defaults for any missing values.
    Returns defaults if the file doesn't exist.
    """
    config = VisionBoxConfig()
    path = Path(path)

    if not path.exists():
        return config

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return config

    resolved = _resolve_recursive(raw)
    _apply_dict(config, resolved)

    return config
