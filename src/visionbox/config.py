"""YAML configuration with environment variable resolution."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    name: str = ''
    url: str = ''
    detect_url: str = ''  # optional: lower-res stream for motion/detection
    test_input: str = ''
    enabled: bool = True


@dataclass
class DetectionConfig:
    mode: str = 'outdoor'
    confidence: float = 0.3
    detect_fps: int = 5
    model: str = 'yolov8n.pt'
    imgsz: int = 640
    device: str = 'auto'   # 'auto' -> iGPU if available else CPU; or 'gpu' / 'cpu'
    class_conf: dict = field(default_factory=lambda: {
        0: 0.35,   # person
        2: 0.45,   # car
        5: 0.50,   # bus
        7: 0.50,   # truck
        14: 0.50,  # bird
        15: 0.40,  # cat
        16: 0.40,  # dog
    })
    confirm_threshold: float = 0.45  # median track score to confirm; keep <= class_conf to avoid a dead-zone
    confirm_window: int = 10         # recent detection scores to median over
    confirm_min_count: int = 2       # min detections before a track can confirm (kills single-frame flicker)
    confirm_threshold_by_class: dict = field(default_factory=dict)  # per-class-id overrides


@dataclass
class MotionConfig:
    min_area: int = 2000


@dataclass
class TrackerConfig:
    max_age: int = 75
    min_hits: int = 2
    iou_threshold: float = 0.2
    max_coast: int = 15
    # Track is "stationary" if its center moved < min_displacement pixels
    # within the last stationary_window seconds. Stationary tracks do NOT
    # trigger recording or get captured for review.
    stationary_min_displacement: float = 30.0
    stationary_window: float = 2.5


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
    max_storage_gb: float = 0  # 0 = unlimited
    check_interval: int = 3600
    max_per_label: int = 0
    priority_labels: list = field(default_factory=lambda: ['person'])


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
class TrainingConfig:
    enabled: bool = False
    dataset_dir: str = 'datasets/yolo'
    replay_dir: str = 'datasets/coco_replay'
    corrected_dir: str = 'datasets/corrected'
    runs_dir: str = 'runs/train'
    epochs: int = 50
    freeze: int = 10
    lr0: float = 0.001
    imgsz: int = 640
    batch: int = 16
    device: str = 'cpu'          # training is CPU-only here (the iGPU can't train)
    min_train_images: int = 50
    regression_tolerance: float = 0.01  # new base-class mAP must be >= old - this
    auto_promote: bool = False
    schedule: str = '03:30'      # overnight HH:MM for the systemd timer / cron


@dataclass
class StorageConfig:
    recordings: str = 'recordings'
    crops: str = 'captures/crops'
    dataset: str = 'captures/dataset'
    review: str = 'datasets/review'
    training: str = 'datasets/training'
    zones_dir: str = 'zones'  # per-camera zone files: zones/<camera>.json


@dataclass
class DisplayConfig:
    web: bool = True
    web_port: int = 8085
    bind_host: str = '0.0.0.0'   # set to the tailscale0 IP to keep the UI off the LAN
    max_fps: int = 15


@dataclass
class VisionBoxConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    storage: StorageConfig = field(default_factory=StorageConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def camera_recordings_dir(self, name: str) -> Path:
        return Path(self.recording.output_dir) / name

    def camera_crops_dir(self, name: str) -> Path:
        return Path(self.storage.crops) / name

    def camera_dataset_dir(self, name: str) -> Path:
        return Path(self.storage.dataset) / name

    def camera_review_dir(self, name: str) -> Path:
        return Path(self.storage.review) / name

    def camera_zones_path(self, name: str) -> Path:
        return Path(self.storage.zones_dir) / f'{name}.json'

    def enabled_cameras(self) -> dict[str, CameraConfig]:
        return {n: c for n, c in self.cameras.items() if c.enabled}


def _resolve_env_vars(value: str) -> str:
    def replacer(match):
        return os.environ.get(match.group(1), '')
    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _resolve_recursive(obj):
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(item) for item in obj]
    return obj


def _apply_dict(dc, data: dict):
    """Apply a dict to a dataclass, recursing into nested dataclasses."""
    for key, value in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(dc, key, value)


def _build_cameras(raw_cameras: dict) -> dict[str, CameraConfig]:
    cameras = {}
    for name, body in (raw_cameras or {}).items():
        body = body or {}
        cam = CameraConfig(name=name)
        _apply_dict(cam, body)
        cam.name = name  # body cannot override the dict key
        cameras[name] = cam
    return cameras


def load_config(path: str | Path = 'config.yml') -> VisionBoxConfig:
    """Load configuration from YAML with ${ENV_VAR} resolution."""
    config = VisionBoxConfig()
    path = Path(path)

    if not path.exists():
        return config

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return config

    resolved = _resolve_recursive(raw)
    cameras_raw = resolved.pop('cameras', {})
    _apply_dict(config, resolved)
    config.cameras = _build_cameras(cameras_raw)

    return config
