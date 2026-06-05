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

# Best-snapshot scoring (Frigate-style): prefer big, confident, non-edge boxes.
EDGE_MARGIN_FRAC = 0.02   # a box side within 2% of a border counts as "touching"
EDGE_PENALTY = 0.6        # multiplier per touched border (compounds)
AREA_WEIGHT = 0.5         # area_ratio**0.5 softens dominance of very large boxes


class RecordingManager:
    def __init__(
        self,
        config: RecordingConfig,
        rtsp_url: str = '',
        camera: str = '',
        output_dir: str | Path | None = None,
        db: RecordingDatabase | None = None,
    ):
        self.config = config
        self.camera = camera
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_dir)
        self._detection_counts: Counter = Counter()
        self._best_thumb: np.ndarray | None = None
        self._best_snapshot: np.ndarray | None = None
        self._best_score: float = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotated = EventRecorder(
            output_dir=str(self.output_dir / 'annotated'),
            cooldown=config.annotated.cooldown,
            fps=config.annotated.fps,
        ) if config.annotated.enabled else None

        self.clean = CleanRecorder(
            output_dir=self.output_dir / 'clean',
            rtsp_url=rtsp_url,
        ) if config.clean.enabled else None

        if db is not None:
            self.db = db
            self._owns_db = False
        else:
            self.db = RecordingDatabase(self.output_dir / 'visionbox.db')
            self._owns_db = True

        self._retention_stop = threading.Event()
        self._retention_thread: threading.Thread | None = None
        self._current_event_id: str | None = None
        self._event_start: datetime | None = None

    def start(self):
        (self.output_dir / 'thumbnails').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'snapshots').mkdir(parents=True, exist_ok=True)
        if self.config.retention.days > 0:
            self._retention_thread = threading.Thread(
                target=self._retention_loop, daemon=True,
                name=f'retention-{self.camera or "default"}',
            )
            self._retention_thread.start()

    @staticmethod
    def _frame_score(detections, w, h) -> float:
        """Best-object quality score for a frame: confidence x area, edge-penalized."""
        if not detections or w <= 0 or h <= 0:
            return 0.0
        mx, my = EDGE_MARGIN_FRAC * w, EDGE_MARGIN_FRAC * h
        area = float(w * h)
        best = 0.0
        for d in detections:
            box = d.get('box')
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area_ratio = min(1.0, (bw * bh) / area) if area else 0.0
            touches = (x1 <= mx) + (y1 <= my) + (x2 >= w - mx) + (y2 >= h - my)
            score = float(d.get('confidence', 0.0)) * (area_ratio ** AREA_WEIGHT) * (EDGE_PENALTY ** touches)
            if score > best:
                best = score
        return best

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
            score = self._frame_score(detections, frame.shape[1], frame.shape[0])
            if score > self._best_score:
                self._best_score = score
                self._best_snapshot = frame.copy()  # clean, unannotated
                self._best_thumb = (annotated_frame.copy()
                                    if annotated_frame is not None else frame.copy())
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

    def _event_key(self, ts: datetime) -> str:
        suffix = f'_{self.camera}' if self.camera else ''
        return ts.strftime('%Y%m%d_%H%M%S') + suffix

    def _start_event(self):
        now = datetime.now()
        event_id = self._event_key(now)
        self._current_event_id = event_id
        self._event_start = now
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_snapshot = None
        self._best_score = 0.0

        clean_clip = ''
        if self.clean:
            path = self.clean.start_event(event_id)
            if path:
                clean_clip = f'clean/event_{event_id}.mp4'

        annotated_clip = ''
        if self.annotated and self.annotated.event_id:
            annotated_clip = f'annotated/event_{self.annotated.event_id}.mp4'

        self.db.insert_event(
            event_id=event_id, start_time=now, camera=self.camera,
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
            thumb_rel = f'thumbnails/event_{self._current_event_id}.jpg'
            thumb_path = self.output_dir / thumb_rel
            cv2.imwrite(str(thumb_path), self._best_thumb,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.db.update_event_thumbnail(self._current_event_id, thumb_rel)

        if self._best_snapshot is not None:
            snap_rel = f'snapshots/event_{self._current_event_id}.jpg'
            cv2.imwrite(str(self.output_dir / snap_rel), self._best_snapshot,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.db.update_event_snapshot(self._current_event_id, snap_rel)

        self._current_event_id = None
        self._event_start = None
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_snapshot = None
        self._best_score = 0.0
        self._enforce_limits()

    def _enforce_limits(self):
        max_keep = self.config.retention.max_per_label
        if max_keep <= 0:
            return

        priority = set(self.config.retention.priority_labels)
        counts = self.db.get_label_counts(camera=self.camera or None)

        non_priority = [l for l in counts if l not in priority and counts[l] > max_keep]
        priority_over = [l for l in counts if l in priority and counts[l] > max_keep]

        for label in non_priority + priority_over:
            overflow = self.db.get_overflow_events(label, max_keep, camera=self.camera or None)
            for ev in overflow:
                self._delete_event_files(ev)
                self.db.delete_event(ev['event_id'])

    def _delete_event_files(self, event: dict):
        out = self.output_dir
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            rel = event.get(key, '')
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else out / rel
            if p.exists():
                p.unlink()
            if key not in ('thumbnail', 'snapshot'):
                meta = p.with_suffix('.json')
                if meta.exists():
                    meta.unlink()

    def _retention_loop(self):
        while not self._retention_stop.wait(self.config.retention.check_interval):
            cutoff = datetime.now() - timedelta(days=self.config.retention.days)
            # Filter to this camera's events only
            events = [
                e for e in self.db.get_events_before(cutoff)
                if (e.get('camera') or '') == self.camera
            ]
            for event in events:
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
        events = [
            e for e in self.db.get_events_by_delete_priority(priority)
            if (e.get('camera') or '') == self.camera
        ]
        for event in events:
            if current <= max_bytes:
                break
            freed = self._delete_event_files_sized(event)
            self.db.delete_event(event['event_id'])
            current -= freed

    def _delete_event_files_sized(self, event: dict) -> int:
        freed = 0
        out = self.output_dir
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            rel = event.get(key, '')
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else out / rel
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            if key not in ('thumbnail', 'snapshot'):
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
        if self._owns_db:
            self.db.close()
