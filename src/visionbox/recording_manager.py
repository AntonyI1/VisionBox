"""Orchestrates dual recording (clean + annotated), database, and retention."""

import logging
import subprocess
import threading
import time
from collections import Counter
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .clean_recorder import CleanRecorder
from .config import RecordingConfig
from .database import RecordingDatabase
from .recorder import EventRecorder, RecorderState

logger = logging.getLogger(__name__)

# The storage budget covers the whole recordings tree, so only one camera's
# retention thread may measure and trim it at a time.
_storage_budget_lock = threading.Lock()

# An annotated clip smaller than this holds no frames: ffmpeg wrote its headers and died.
ANNOTATED_MIN_BYTES = 4096

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
        # Event timing when there is no annotated recorder to follow.
        self._clock = time.monotonic
        self._event_start_mono = 0.0
        self._last_trigger_mono = 0.0
        self._timer_state = RecorderState.IDLE

    def start(self):
        (self.output_dir / 'thumbnails').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'snapshots').mkdir(parents=True, exist_ok=True)
        self._recover_open_events()
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
            recording = self.annotated.is_recording
        else:
            recording = self._time_event(triggered)

        if recording and not was_recording:
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

        if was_recording and not recording:
            self._end_event()

    def _time_event(self, triggered: bool) -> bool:
        """Same IDLE -> RECORDING -> COOLDOWN cycle as EventRecorder, without a clip to write."""
        now = self._clock()
        timing = self.config.annotated
        if self._current_event_id is None:
            if not triggered:
                self._timer_state = RecorderState.IDLE
                return False
            self._event_start_mono = self._last_trigger_mono = now
            self._timer_state = RecorderState.RECORDING
            return True
        if timing.max_duration > 0 and now - self._event_start_mono >= timing.max_duration:
            self._timer_state = RecorderState.IDLE
            return False
        if triggered:
            self._last_trigger_mono = now
            self._timer_state = RecorderState.RECORDING
            return True
        if now - self._last_trigger_mono < timing.cooldown:
            self._timer_state = RecorderState.COOLDOWN
            return True
        self._timer_state = RecorderState.IDLE
        return False

    @property
    def is_recording(self) -> bool:
        return self._current_event_id is not None

    @property
    def state(self) -> RecorderState:
        if self.annotated:
            return self.annotated.state
        return self._timer_state

    @property
    def event_id(self) -> str | None:
        return self._current_event_id

    def _event_key(self, ts: datetime) -> str:
        suffix = f'_{self.camera}' if self.camera else ''
        return ts.strftime('%Y%m%d_%H%M%S') + suffix

    def _reset_event_state(self):
        self._current_event_id = None
        self._event_start = None
        self._annotated_rel = ''
        self._detection_counts.clear()
        self._best_thumb = None
        self._best_snapshot = None
        self._best_score = 0.0
        self._best_ts = 0.0

    def _start_event(self):
        self._reset_event_state()
        now = datetime.now()
        event_id = base_id = self._event_key(now)
        n = 1
        while self.db.get_event(event_id) is not None:   # two events within one second
            n += 1
            event_id = f'{base_id}_{n}'
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
        self._annotated_rel = annotated_clip

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
        if self.clean and self.clean.event_id == self._current_event_id:
            # Finalise whether ffmpeg is still running, hit its -t cap or died early:
            # the size floor, the sidecar and the DB pointer are decided in one place.
            clean_path = self.clean.stop_event()
            if clean_path is None:
                self.db.update_event_clean_clip(self._current_event_id, '')

        if self._annotated_rel:
            # The annotated recorder has already closed its ffmpeg; a clip that never
            # grew past its headers (or never appeared) would 404 in the dashboard.
            clip = self.output_dir / self._annotated_rel
            size = clip.stat().st_size if clip.exists() else -1
            if size < ANNOTATED_MIN_BYTES:
                logger.warning('[%s] annotated clip for %s is %s; dropping it from the event',
                               self.camera or 'default', self._current_event_id,
                               'missing' if size < 0 else f'{size} bytes')
                with suppress(OSError):
                    clip.unlink(missing_ok=True)
                    clip.with_suffix('.json').unlink(missing_ok=True)
                self.db.update_event_annotated_clip(self._current_event_id, '')

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

    def _recover_open_events(self):
        """Settle rows a crash left without an end time: keep a usable clean clip, drop the rest.

        Retention only ever looks at finished events, so without this an OOM kill or a
        watchdog restart leaves the row and its clips on disk for good.
        """
        for event in self.db.get_open_events(camera=self.camera):
            event_id = event['event_id']
            clip = self._media_path(event, 'clean_clip')
            usable = clip is not None and clip.exists() and clip.stat().st_size >= self.config.clean.min_valid_bytes
            if not usable:
                self._delete_event_files(event)
                self.db.delete_event(event_id)
                logger.info('[%s] dropped unfinished event %s: no usable clip', self.camera or 'default', event_id)
                continue
            ended = datetime.fromtimestamp(clip.stat().st_mtime)
            started = datetime.fromisoformat(event['start_time'])
            duration = round(max(0.0, (ended - started).total_seconds()), 1)
            self.db.update_event_end(event_id, ended, duration)
            annotated = self._media_path(event, 'annotated_clip')
            if annotated is not None and not annotated.exists():
                self.db.update_event_annotated_clip(event_id, '')
            logger.info('[%s] recovered event %s left open by a restart (%.1fs)',
                        self.camera or 'default', event_id, duration)

    def _media_path(self, event: dict, key: str) -> Path | None:
        rel = event.get(key) or ''
        if not rel:
            return None
        path = Path(rel)
        return path if path.is_absolute() else self._event_dir(event) / path

    def _storage_root(self) -> Path:
        """The shared recordings tree: cameras record into sibling subdirectories of it."""
        return self.output_dir.parent if self.camera else self.output_dir

    def _event_dir(self, event: dict) -> Path:
        camera = event.get('camera') or ''
        if not self.camera or camera == self.camera:
            return self.output_dir
        return self._storage_root() / camera

    def _delete_event_files(self, event: dict) -> int:
        """Remove an event's clips, sidecar metadata, thumbnail and snapshot; returns bytes freed."""
        freed = 0
        base = self._event_dir(event)
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            rel = event.get(key, '')
            if not rel:
                continue
            # quarantine_false_positives.py parks a flagged event's files under _quarantine/
            rel_path = Path(rel)
            paths = [rel_path] if rel_path.is_absolute() else [base / rel_path, base / '_quarantine' / rel_path]
            if key.endswith('_clip'):
                paths += [p.with_suffix('.json') for p in paths]
            for p in paths:
                # The API's delete route and the per-label cap race this thread for the
                # same files; a file that vanished in between is not an error.
                try:
                    size = p.stat().st_size
                    p.unlink()
                except FileNotFoundError:
                    continue
                freed += size
        return freed

    def _retention_loop(self):
        while not self._retention_stop.wait(self.config.retention.check_interval):
            try:
                self._retention_pass()
            except Exception:
                logger.exception('[%s] retention pass failed; retrying next interval',
                                 self.camera or 'default')

    def _retention_pass(self):
        cutoff = datetime.now() - timedelta(days=self.config.retention.days)
        for event in self.db.get_events_before(cutoff):
            if (event.get('camera') or '') != self.camera:
                continue
            self._delete_event_files(event)
            self.db.delete_event(event['event_id'])
        self._enforce_storage_limit()

    @staticmethod
    def _tree_bytes(root: Path) -> int:
        if not root.exists():
            return 0
        total = 0
        for f in root.rglob('*'):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _enforce_storage_limit(self):
        """Trim the whole recordings tree to max_storage_gb, oldest non-priority events first.

        Measured and trimmed across cameras: a per-camera budget would let four cameras
        fill four times the configured limit while the dashboard reports one.
        """
        max_gb = self.config.retention.max_storage_gb
        if max_gb <= 0:
            return
        max_bytes = int(max_gb * 1024 * 1024 * 1024)
        with _storage_budget_lock:
            current = self._tree_bytes(self._storage_root())
            if current <= max_bytes:
                return
            for event in self.db.get_events_by_delete_priority(self.config.retention.priority_labels):
                if current <= max_bytes:
                    break
                current -= self._delete_event_files(event)
                self.db.delete_event(event['event_id'])

    def abort(self):
        """Close the open event when the stream drops.

        Releasing the annotated recorder matters: left mid-recording, its state would
        restart an event on the first frame after reconnect, with a row pointing at the
        previous event's clip.
        """
        if self.annotated:
            self.annotated.release()
        if self._current_event_id:
            self._end_event()
        self._timer_state = RecorderState.IDLE

    def stop(self):
        self.abort()
        if self.clean:
            self.clean.release()
        self._retention_stop.set()
        if self._retention_thread and self._retention_thread.is_alive():
            self._retention_thread.join(timeout=2)
        if self._owns_db:
            self.db.close()
