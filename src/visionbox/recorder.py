"""
Event-based video recorder with state machine.

Records video clips only when events (motion/detections) occur.
Uses a cooldown to avoid flickery start/stop behavior:

    IDLE → motion detected → RECORDING → motion stops → COOLDOWN → IDLE
                                ↑                          │
                                └── motion resumes ────────┘

Each event produces:
- Video clip (.mp4)
- Metadata JSON (start time, duration, detections seen)
"""

import cv2
import json
import time
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


class RecorderState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    COOLDOWN = "cooldown"


@dataclass
class EventMetadata:
    """Metadata for a recorded event clip."""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    detections: list = field(default_factory=list)
    detection_count: int = 0
    max_objects_in_frame: int = 0

    def add_detections(self, dets: list[dict]):
        if not dets:
            return
        self.detection_count += len(dets)
        self.max_objects_in_frame = max(self.max_objects_in_frame, len(dets))
        # Log unique classes seen
        for d in dets:
            cls = d.get('class_name', d.get('class', 'unknown'))
            conf = d.get('confidence', 0)
            entry = {'class': cls, 'confidence': round(conf, 3)}
            if entry not in self.detections[-10:]:  # avoid massive logs
                self.detections.append(entry)


class EventRecorder:
    """
    Records video clips triggered by motion/detection events.

    Args:
        output_dir: Where to save clips and metadata.
        cooldown: Seconds to keep recording after last trigger.
                  Prevents flickery start/stop. 10s is a good default.
        fps: Frame rate for output video.
        codec: FourCC codec string.
    """

    def __init__(
        self,
        output_dir: str = 'recordings/events',
        cooldown: float = 10.0,
        fps: float = 15.0,
        codec: str = 'mp4v',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.fps = fps
        self.codec = codec

        self.state = RecorderState.IDLE
        self._writer: cv2.VideoWriter | None = None
        self._current_path: Path | None = None
        self._meta: EventMetadata | None = None
        self._last_trigger: float = 0
        self._start_time: float = 0
        self._frame_size: tuple[int, int] | None = None

    @property
    def is_recording(self) -> bool:
        return self.state in (RecorderState.RECORDING, RecorderState.COOLDOWN)

    def update(self, frame: np.ndarray, triggered: bool, detections: list[dict] | None = None):
        """
        Call this every frame.

        Args:
            frame: The video frame.
            triggered: Whether motion/detection is active this frame.
            detections: Optional detection results for metadata logging.
        """
        now = time.time()

        if self.state == RecorderState.IDLE:
            if triggered:
                self._start_recording(frame, now)
                self.state = RecorderState.RECORDING

        elif self.state == RecorderState.RECORDING:
            if triggered:
                self._last_trigger = now
            else:
                # Motion stopped, enter cooldown
                self.state = RecorderState.COOLDOWN

        elif self.state == RecorderState.COOLDOWN:
            if triggered:
                # Motion resumed, back to recording
                self._last_trigger = now
                self.state = RecorderState.RECORDING
            elif now - self._last_trigger >= self.cooldown:
                # Cooldown expired, stop recording
                self._stop_recording(now)
                self.state = RecorderState.IDLE
                return

        # Write frame if recording
        if self.is_recording and self._writer is not None:
            self._writer.write(frame)
            if detections and self._meta:
                self._meta.add_detections(detections)

    def _start_recording(self, frame: np.ndarray, now: float):
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        self._start_time = now
        self._last_trigger = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_path = self.output_dir / f"event_{timestamp}.mp4"
        meta_path = self.output_dir / f"event_{timestamp}.json"

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self._current_path), fourcc, self.fps, self._frame_size
        )

        self._meta = EventMetadata(
            start_time=datetime.now().isoformat(),
        )
        self._meta_path = meta_path

    def _stop_recording(self, now: float):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._meta is not None:
            self._meta.end_time = datetime.now().isoformat()
            self._meta.duration_seconds = round(now - self._start_time, 1)

            with open(self._meta_path, 'w') as f:
                json.dump({
                    'start_time': self._meta.start_time,
                    'end_time': self._meta.end_time,
                    'duration_seconds': self._meta.duration_seconds,
                    'detection_count': self._meta.detection_count,
                    'max_objects_in_frame': self._meta.max_objects_in_frame,
                    'clip': self._current_path.name,
                }, f, indent=2)

        self._meta = None
        self._current_path = None

    def release(self):
        """Clean up on shutdown."""
        if self.is_recording:
            self._stop_recording(time.time())
