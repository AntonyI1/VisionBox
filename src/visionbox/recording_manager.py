"""Orchestrates dual recording (clean + annotated), database, and retention."""

import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .clean_recorder import CleanRecorder
from .config import RecordingConfig
from .database import RecordingDatabase
from .recorder import EventRecorder, RecorderState


class RecordingManager:
    def __init__(self, config: RecordingConfig, rtsp_url: str = ''):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self._detection_counts: Counter = Counter()
        self._best_thumb: np.ndarray | None = None
        self._best_thumb_count: int = 0

        self.annotated = EventRecorder(
            output_dir=str(self.output_dir / 'annotated'),
            cooldown=config.annotated.cooldown,
            fps=config.annotated.fps,
        ) if config.annotated.enabled else None

        self.clean = CleanRecorder(
            output_dir=self.output_dir / 'clean',
            rtsp_url=rtsp_url,
        ) if config.clean.enabled else None

        db_path = self.output_dir / 'visionbox.db'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = RecordingDatabase(db_path)

        self._retention_stop = threading.Event()
        self._retention_thread: threading.Thread | None = None
        self._current_event_id: str | None = None
        self._event_start: datetime | None = None

    def start(self):
        (self.output_dir / 'thumbnails').mkdir(parents=True, exist_ok=True)
        if self.config.retention.days > 0:
            self._retention_thread = threading.Thread(
                target=self._retention_loop, daemon=True
            )
            self._retention_thread.start()

    def update(
        self,
        frame: np.ndarray,
        annotated_frame: np.ndarray | None,
        triggered: bool,
        detections: list[dict] | None = None,
    ):
        was_recording = self._current_event_id is not None

        if self.annotated:
            self.annotated.update(
                annotated_frame if annotated_frame is not None else frame,
                triggered, detections,
            )

        if not was_recording and self.annotated and self.annotated.is_recording:
            self._start_event()
        elif not was_recording and self.annotated is None and triggered:
            self._start_event()

        if self._current_event_id and detections:
            det_count = len(detections)
            if det_count > self._best_thumb_count and annotated_frame is not None:
                self._best_thumb = annotated_frame.copy()
                self._best_thumb_count = det_count
            for d in detections:
                self._detection_counts[d.get('class_name', 'unknown')] += 1

        if was_recording:
            annotated_idle = self.annotated is None or self.annotated.state == RecorderState.IDLE
            if annotated_idle:
                self._end_event()

    @property
    def is_recording(self) -> bool:
        return self._current_event_id is not None

    @property
    def state(self) -> RecorderState:
        if self.annotated:
            return self.annotated.state
        return RecorderState.IDLE

    @property
    def event_id(self) -> str | None:
        return self._current_event_id

    def _start_event(self):
        now = datetime.now()
        event_id = now.strftime("%Y%m%d_%H%M%S")
        self._current_event_id = event_id
        self._event_start = now
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_thumb_count = 0

        clean_clip = ''
        if self.clean:
            path = self.clean.start_event(event_id)
            if path:
                clean_clip = f"clean/event_{event_id}.mp4"

        annotated_clip = ''
        if self.annotated and self.annotated.event_id:
            annotated_clip = f"annotated/event_{self.annotated.event_id}.mp4"

        self.db.insert_event(
            event_id=event_id, start_time=now,
            clean_clip=clean_clip, annotated_clip=annotated_clip,
        )

    def _end_event(self):
        if not self._current_event_id:
            return

        now = datetime.now()
        duration = (now - self._event_start).total_seconds() if self._event_start else 0

        if self.clean and self.clean.is_recording:
            self.clean.stop_event()

        top_label = ''
        total_detections = sum(self._detection_counts.values())
        if self._detection_counts:
            top_label = self._detection_counts.most_common(1)[0][0]

        self.db.update_event_end(
            event_id=self._current_event_id, end_time=now,
            duration=round(duration, 1),
            detection_count=total_detections, top_label=top_label,
        )

        if self._best_thumb is not None:
            thumb_rel = f"thumbnails/event_{self._current_event_id}.jpg"
            thumb_path = self.output_dir / thumb_rel
            cv2.imwrite(str(thumb_path), self._best_thumb,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.db.update_event_thumbnail(self._current_event_id, thumb_rel)

        self._current_event_id = None
        self._event_start = None
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_thumb_count = 0
        self._enforce_limits()

    def _enforce_limits(self):
        max_keep = self.config.retention.max_per_label
        if max_keep <= 0:
            return

        priority = set(self.config.retention.priority_labels)
        counts = self.db.get_label_counts()

        non_priority = [l for l in counts if l not in priority and counts[l] > max_keep]
        priority_over = [l for l in counts if l in priority and counts[l] > max_keep]

        for label in non_priority + priority_over:
            overflow = self.db.get_overflow_events(label, max_keep)
            for ev in overflow:
                self._delete_event_files(ev)
                self.db.delete_event(ev['event_id'])

    def _delete_event_files(self, event: dict):
        out = self.output_dir
        for key in ('clean_clip', 'annotated_clip', 'thumbnail'):
            rel = event.get(key, '')
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else out / rel
            if p.exists():
                p.unlink()
            if key != 'thumbnail':
                meta = p.with_suffix('.json')
                if meta.exists():
                    meta.unlink()

    def _retention_loop(self):
        while not self._retention_stop.wait(self.config.retention.check_interval):
            cutoff = datetime.now() - timedelta(days=self.config.retention.days)
            for event in self.db.get_events_before(cutoff):
                self._delete_event_files(event)
                self.db.delete_event(event['event_id'])
            self._enforce_storage_limit()

    def _get_storage_bytes(self) -> int:
        if not self.output_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())

    def _enforce_storage_limit(self):
        max_gb = self.config.retention.max_storage_gb
        if max_gb <= 0:
            return
        max_bytes = int(max_gb * 1024 * 1024 * 1024)
        current = self._get_storage_bytes()
        if current <= max_bytes:
            return
        priority = self.config.retention.priority_labels
        events = self.db.get_events_by_delete_priority(priority)
        for event in events:
            if current <= max_bytes:
                break
            freed = self._delete_event_files_sized(event)
            self.db.delete_event(event['event_id'])
            current -= freed

    def _delete_event_files_sized(self, event: dict) -> int:
        """Delete event files and return bytes freed."""
        freed = 0
        out = self.output_dir
        for key in ('clean_clip', 'annotated_clip', 'thumbnail'):
            rel = event.get(key, '')
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else out / rel
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            if key != 'thumbnail':
                meta = p.with_suffix('.json')
                if meta.exists():
                    freed += meta.stat().st_size
                    meta.unlink()
        return freed

    def stop(self):
        if self._current_event_id:
            self._end_event()
        if self.annotated:
            self.annotated.release()
        if self.clean:
            self.clean.release()
        self._retention_stop.set()
        if self._retention_thread and self._retention_thread.is_alive():
            self._retention_thread.join(timeout=2)
