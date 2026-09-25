"""Orchestrates dual recording (clean + annotated), database, and retention."""

import subprocess
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


def _upgrade_snapshot(clip_path: str, offset: float, dest: Path):
    """Replace a detect-stream snapshot with the same moment from the full-res clean clip."""
    tmp = dest.with_name(dest.stem + '.tmp.jpg')
    cmd = ['ffmpeg', '-ss', f'{offset:.2f}', '-i', clip_path,
           '-frames:v', '1', '-q:v', '2', '-y', str(tmp)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=60, check=False)
        if proc.returncode == 0 and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    tmp.unlink(missing_ok=True)


class RecordingManager:
    def __init__(
        self,
        config: RecordingConfig,
        rtsp_url: str = '',
        camera: str = '',
        output_dir: str | Path | None = None,
        db: RecordingDatabase | None = None,
        snapshot_from_clean: bool = False,
    ):
        self.config = config
        self.camera = camera
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_dir)
        self.snapshot_from_clean = snapshot_from_clean
        self._detection_counts: Counter = Counter()
        self._best_thumb: np.ndarray | None = None
        self._best_snapshot: np.ndarray | None = None
        self._best_score: float = 0.0
        self._best_ts: float = 0.0
        self._clean_start_ts: float = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotated = EventRecorder(
            output_dir=str(self.output_dir / 'annotated'),
            cooldown=config.annotated.cooldown,
            fps=config.annotated.fps,
            max_duration=config.annotated.max_duration,
        ) if config.annotated.enabled else None

        self.clean = CleanRecorder(
            output_dir=self.output_dir / 'clean',
            rtsp_url=rtsp_url,
            max_duration=config.clean.max_duration,
            min_valid_bytes=config.clean.min_valid_bytes,
            stop_grace=config.clean.stop_grace,
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
            best = max(best, score)
        return best

    def update(
        self,
        frame: np.ndarray,
        annotated_frame: np.ndarray | None,
        triggered: bool,
        detections: list[dict] | None = None,
        trigger_detections: list[dict] | None = None,
        count_now: bool = True,
    ):
        was_recording = self._current_event_id is not None

        if self.annotated:
            self.annotated.update(
                annotated_frame if annotated_frame is not None else frame,
                triggered, detections,
            )

        if not was_recording:
            started = self.annotated.is_recording if self.annotated else triggered
            if started:
                self._start_event()

        # Score the snapshot and tally labels from the trigger subset (confirmed, moving objects)
        # when given, so parked cars do not dominate top_label / detection_count / snapshot.
        score_dets = trigger_detections if trigger_detections is not None else detections
        if self._current_event_id and score_dets:
            score = self._frame_score(score_dets, frame.shape[1], frame.shape[0])
            if score > self._best_score:
                self._best_score = score
                self._best_snapshot = frame.copy()  # clean, unannotated
                self._best_ts = time.time()
                self._best_thumb = (annotated_frame.copy()
                                    if annotated_frame is not None else frame.copy())
            if count_now:
                for d in score_dets:
                    self._detection_counts[d.get('class_name', 'unknown')] += 1

        if was_recording and (self.annotated is None or self.annotated.state == RecorderState.IDLE):
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

    def _reset_event_state(self):
        self._current_event_id = None
        self._event_start = None
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_snapshot = None
        self._best_score = 0.0
        self._best_ts = 0.0

    def _start_event(self):
        self._reset_event_state()
        now = datetime.now()
        event_id = self._event_key(now)
        self._current_event_id = event_id
        self._event_start = now

        clean_clip = ''
        if self.clean:
            path = self.clean.start_event(event_id)
            if path:
                clean_clip = f'clean/event_{event_id}.mp4'
                self._clean_start_ts = time.time()

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

        clean_path = None
        if self.clean:
            if self.clean.is_recording:
                clean_path = self.clean.stop_event()
                if clean_path is None:
                    # broken/empty clean clip was discarded — clear the dangling DB pointer
                    self.db.update_event_clean_clip(self._current_event_id, '')
            else:
                # ffmpeg already exited (-t cap); the clip may still exist on disk
                p = self.output_dir / 'clean' / f'event_{self._current_event_id}.mp4'
                clean_path = str(p) if p.exists() else None

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
            snap_path = self.output_dir / snap_rel
            cv2.imwrite(str(snap_path), self._best_snapshot,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.db.update_event_snapshot(self._current_event_id, snap_rel)
            if self.snapshot_from_clean and clean_path and self._best_ts:
                # Detect-stream frame is low-res; swap in the same moment from the
                # full-res clean clip. The file above stays as the fallback.
                offset = min(max(0.0, self._best_ts - self._clean_start_ts),
                             max(0.0, self.config.clean.max_duration - 1.0))
                threading.Thread(
                    target=_upgrade_snapshot, args=(clean_path, offset, snap_path),
                    daemon=True, name=f'snapshot-{self.camera or "default"}',
                ).start()

        self._reset_event_state()
        self._enforce_limits()

    def _enforce_limits(self):
        max_keep = self.config.retention.max_per_label
        if max_keep <= 0:
            return

        priority = set(self.config.retention.priority_labels)
        counts = self.db.get_label_counts(camera=self.camera or None)

        non_priority = [label for label, n in counts.items() if label not in priority and n > max_keep]
        priority_over = [label for label, n in counts.items() if label in priority and n > max_keep]

        for label in non_priority + priority_over:
            overflow = self.db.get_overflow_events(label, max_keep, camera=self.camera or None)
            for ev in overflow:
                self._delete_event_files(ev)
                self.db.delete_event(ev['event_id'])

    def _delete_event_files(self, event: dict) -> int:
        """Remove an event's clips, sidecar metadata, thumbnail and snapshot; returns bytes freed."""
        freed = 0
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            rel = event.get(key, '')
            if not rel:
                continue
            path = Path(rel) if Path(rel).is_absolute() else self.output_dir / rel
            targets = [path] if key in ('thumbnail', 'snapshot') else [path, path.with_suffix('.json')]
            for p in targets:
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
        return freed

    def _retention_loop(self):
        while not self._retention_stop.wait(self.config.retention.check_interval):
            cutoff = datetime.now() - timedelta(days=self.config.retention.days)
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
            current -= self._delete_event_files(event)
            self.db.delete_event(event['event_id'])

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
