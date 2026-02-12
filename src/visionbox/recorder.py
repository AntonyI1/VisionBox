"""Event-based video recorder with state machine.

State: IDLE → RECORDING → COOLDOWN → IDLE
"""

import json
import shutil
import subprocess
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
        for d in dets:
            cls = d.get('class_name', d.get('class', 'unknown'))
            conf = d.get('confidence', 0)
            entry = {'class': cls, 'confidence': round(conf, 3)}
            if entry not in self.detections[-10:]:
                self.detections.append(entry)


class EventRecorder:
    def __init__(
        self,
        output_dir: str = 'recordings/events',
        cooldown: float = 10.0,
        fps: float = 15.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.fps = fps
        self._use_ffmpeg = bool(shutil.which('ffmpeg'))

        self.state = RecorderState.IDLE
        self._process: subprocess.Popen | None = None
        self._current_path: Path | None = None
        self._meta: EventMetadata | None = None
        self._last_trigger: float = 0
        self._start_time: float = 0
        self._frame_size: tuple[int, int] | None = None

    @property
    def is_recording(self) -> bool:
        return self.state in (RecorderState.RECORDING, RecorderState.COOLDOWN)

    @property
    def event_id(self) -> str | None:
        if self._current_path is not None:
            return self._current_path.stem.replace('event_', '')
        return None

    def update(self, frame: np.ndarray, triggered: bool, detections: list[dict] | None = None):
        now = time.time()

        if self.state == RecorderState.IDLE:
            if triggered:
                self._start_recording(frame, now)
                self.state = RecorderState.RECORDING

        elif self.state == RecorderState.RECORDING:
            if triggered:
                self._last_trigger = now
            else:
                self.state = RecorderState.COOLDOWN

        elif self.state == RecorderState.COOLDOWN:
            if triggered:
                self._last_trigger = now
                self.state = RecorderState.RECORDING
            elif now - self._last_trigger >= self.cooldown:
                self._stop_recording(now)
                self.state = RecorderState.IDLE
                return

        if self.is_recording and self._process is not None:
            try:
                self._process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
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

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(self._current_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            self._process = None

        self._meta = EventMetadata(start_time=datetime.now().isoformat())
        self._meta_path = meta_path

    def _stop_recording(self, now: float):
        if self._process is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            self._process.wait(timeout=10)
            self._process = None

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
        if self.is_recording:
            self._stop_recording(time.time())
