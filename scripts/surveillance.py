#!/usr/bin/env python3
"""VisionBox surveillance pipeline. Multi-camera, single process."""

import argparse
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

# Time out on stalled RTSP streams so the reconnect logic fires (must precede cv2 use).
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp|timeout;5000000')

import cv2
import numpy as np

from visionbox import (
    Tracker,
    CLASS_PRESETS_V2,
    MotionDetector,
    merge_overlapping_regions,
)
from visionbox.detector_v2 import MultiModelDetector, ModelConfig
from visionbox.config import load_config, CameraConfig, VisionBoxConfig
from visionbox.database import RecordingDatabase
from visionbox.recording_manager import RecordingManager
from visionbox.api import (
    CamerasState, CameraView, start_api_server,
)
from visionbox.zones import ZoneFilter


np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]


def _redact_url(url: str) -> str:
    """Strip credentials from an RTSP URL before logging."""
    return re.sub(r'://[^/@]*@', '://***@', url)


def _rtsp_ready(url: str, timeout: float = 4.0) -> bool:
    """Liveness probe before the heavy cv2 open. A wedged source (TCP accepts, data
    never comes) makes every VideoCapture retry churn ffmpeg allocations — days of
    that fragments the heap until av_frame_get_buffer fails for ALL cameras. Any
    RTSP status line counts as alive (401/454 still prove the server talks);
    timeout, refusal, or an empty reply means don't bother opening."""
    if not url.startswith('rtsp://'):
        return True  # file/test inputs
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 554),
                                      timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\n"
                          "Accept: application/sdp\r\nUser-Agent: VisionBox\r\n\r\n").encode())
            return sock.recv(64).startswith(b'RTSP/1.0 ')
    except OSError:
        return False


class CameraStream:
    """Threaded RTSP reader — always returns the latest frame."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        self.stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name='rtsp-reader')
        self._thread.start()

    def _reader(self):
        # The cap is released HERE, in the same thread that reads it. Releasing a
        # VideoCapture from another thread while this one is inside cap.read() is a
        # use-after-free in ffmpeg/OpenCV and segfaults the whole process — which is
        # what happened on every flaky-WiFi reconnect.
        try:
            while not self.stopped:
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                if ret and frame is not None:
                    self.last_frame_time = time.time()
                else:
                    time.sleep(0.1)
        finally:
            self.cap.release()

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        # Signal the reader to stop and let IT release the cap; never release across
        # threads. join() waits out an in-flight read (RTSP timeout ~5s); if it can't,
        # the daemon reader still releases the cap itself when read() finally returns.
        self.stopped = True
        if self._thread is not None:
            self._thread.join(timeout=6)
            self._thread = None


def start_rss_watchdog(max_rss_mb: int):
    """Self-restart valve for slow leaks: a bloated resident set starves frame
    buffers for every camera long before the kernel OOM-killer cares. Three
    strikes, then SIGTERM ourselves so open events finalize and systemd respawns
    us; hard-exit only if that graceful shutdown wedges."""
    def _watch():
        strikes = 0
        while True:
            time.sleep(60)
            try:
                with open('/proc/self/status') as f:
                    rss_kb = next(int(line.split()[1]) for line in f
                                  if line.startswith('VmRSS'))
            except (OSError, StopIteration, ValueError):
                continue
            strikes = strikes + 1 if rss_kb > max_rss_mb * 1024 else 0
            if strikes >= 3:
                print(f"[watchdog] RSS {rss_kb // 1024}MB above {max_rss_mb}MB cap "
                      f"for 3 checks; restarting for a clean slate", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(60)
                os._exit(42)
    threading.Thread(target=_watch, daemon=True, name='rss-watchdog').start()


def open_browser(url: str):
    import shutil
    for cmd in ['xdg-open', 'cmd.exe']:
        if shutil.which(cmd):
            try:
                args = [cmd, url] if cmd != 'cmd.exe' else [cmd, '/c', 'start', url.replace('&', '^&')]
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                pass


def draw_tracks(image, tracks, track_info, class_names):
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        info = track_info.get(track_id)
        if info:
            class_id, class_name, conf = info['class_id'], info['class_name'], info['confidence']
        else:
            class_id, class_name, conf = 0, class_names.get(0, 'unknown'), 0
        color = COLORS[track_id % len(COLORS)]
        if class_id == 80:
            color = (0, 255, 255)
            label = f"PLATE #{track_id} {conf:.0%}"
        else:
            label = f"{class_name} #{track_id} {conf:.0%}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _overlaps_motion(det_box, motion_boxes):
    dx1, dy1, dx2, dy2 = det_box
    for mx1, my1, mx2, my2 in motion_boxes:
        if dx1 < mx2 and dx2 > mx1 and dy1 < my2 and dy2 > my1:
            return True
    return False


class CameraPipeline:
    """Per-camera worker: capture → motion → detect → track → record."""

    def __init__(
        self,
        cam_cfg: CameraConfig,
        global_cfg: VisionBoxConfig,
        detector: MultiModelDetector,
        detector_lock: threading.Lock,
        db: RecordingDatabase,
    ):
        self.name = cam_cfg.name
        self.url = cam_cfg.url
        self.detect_url = cam_cfg.detect_url or cam_cfg.url
        self.test_input = cam_cfg.test_input
        self.cfg = global_cfg
        self.detector = detector
        self.detector_lock = detector_lock
        self.db = db

        self.recordings_dir = global_cfg.camera_recordings_dir(self.name)
        self.crops_dir = global_cfg.camera_crops_dir(self.name)
        self.dataset_dir = global_cfg.camera_dataset_dir(self.name)
        self.review_dir = global_cfg.camera_review_dir(self.name)
        zones_path = global_cfg.camera_zones_path(self.name)

        for d in [
            self.recordings_dir, self.crops_dir, self.review_dir,
            self.dataset_dir / 'images', self.dataset_dir / 'labels',
            zones_path.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.zone_filter = ZoneFilter(str(zones_path))
        self.motion = MotionDetector(
            min_area=global_cfg.motion.min_area,
            min_area_frac=global_cfg.motion.min_area_frac,
            var_threshold=global_cfg.motion.var_threshold,
            history=global_cfg.motion.history,
            detect_shadows=global_cfg.motion.detect_shadows,
            learning_rate=global_cfg.motion.learning_rate,
        )
        self.tracker = Tracker(
            max_age=global_cfg.tracker.max_age,
            min_hits=global_cfg.tracker.min_hits,
            iou_threshold=global_cfg.tracker.iou_threshold,
            max_coast=global_cfg.tracker.max_coast,
        )
        self.recording_mgr = RecordingManager(
            global_cfg.recording,
            rtsp_url=self.url,
            camera=self.name,
            output_dir=self.recordings_dir,
            db=db,
            # Detection runs on the sub-stream; pull event snapshots from the
            # full-res clean clip instead of the small detect frame.
            snapshot_from_clean=self.detect_url != self.url,
        )

        self.view = CameraView(name=self.name)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._track_info: dict[int, dict] = {}
        self._track_last_capture: dict[int, float] = {}
        self._track_positions: dict[int, deque] = {}
        self._track_scores: dict[int, deque] = {}
        self._capture_count = 0
        self._event_count = 0
        self._class_counts: dict[str, int] = {}
        self._classes_seen: dict[int, str] = {}
        self._last_uncertain_save = 0.0
        self._frame_count = 0
        self._last_real_obj_frame = -10**9   # frame index a person/car/etc. was last seen (illumination-guard override)

    def start(self):
        self.recording_mgr.start()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f'cam-{self.name}',
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.recording_mgr.stop()
        if self._classes_seen:
            with open(self.dataset_dir / 'classes.txt', 'w') as f:
                for cid in sorted(self._classes_seen):
                    f.write(f"{cid}: {self._classes_seen[cid]}\n")

    def _run(self):
        """Top-level loop: handles reconnects."""
        backoff = 2
        while not self._stop.is_set():
            try:
                if self._connect_and_process():
                    backoff = 2
            except Exception as exc:
                self.view.connected = False
                self.view.last_error = f'{type(exc).__name__}: {exc}'
                print(f"[{self.name}] error: {self.view.last_error}", flush=True)
            if self._stop.is_set():
                break
            for _ in range(int(backoff * 10)):
                if self._stop.is_set():
                    return
                time.sleep(0.1)
            backoff = min(backoff * 2, 30)

    def _connect_and_process(self) -> bool:
        stream_url = self.test_input or self.detect_url
        if not _rtsp_ready(stream_url):
            self.view.connected = False
            if self.view.last_error != 'source unreachable':
                print(f"[{self.name}] source unreachable, retrying quietly until it answers",
                      flush=True)
            self.view.last_error = 'source unreachable'
            return False
        print(f"[{self.name}] connecting to {_redact_url(stream_url)}", flush=True)
        cap = CameraStream(stream_url)
        if not cap.isOpened():
            self.view.connected = False
            self.view.last_error = 'open failed'
            cap.release()
            return False
        time.sleep(1)
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            self.view.connected = False
            self.view.last_error = 'no frames'
            cap.release()
            return False
        self.view.connected = True
        self.view.last_error = ''
        print(f"[{self.name}] connected ({test_frame.shape[1]}x{test_frame.shape[0]})", flush=True)
        try:
            self._main_loop(cap)
        finally:
            # Finalize any open event on every exit path (stall/lost/exception) so a reconnect
            # or restart can't leave a NULL-end orphan with a runaway clean ffmpeg behind it.
            if self.recording_mgr.is_recording:
                self.recording_mgr._end_event()
            cap.release()
        return True

    def _main_loop(self, cap: CameraStream):
        cfg = self.cfg
        class_filter = CLASS_PRESETS_V2[cfg.detection.mode]

        prev_event_id = self.recording_mgr.event_id
        min_frame_time = 1.0 / cfg.display.max_fps
        last_loop_time = time.time()
        detect_interval = 1.0 / cfg.detection.detect_fps
        last_detect_time = 0.0
        det_array = np.empty((0, 6))
        detections: list[dict] = []
        merged_full: list[tuple] = []
        frame_times: list[float] = []

        consecutive_failures = 0

        while not self._stop.is_set():
            if time.time() - cap.last_frame_time > 10:
                self.view.connected = False
                self.view.last_error = 'stream stalled'
                print(f"[{self.name}] stream stalled, reconnecting", flush=True)
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 100:  # ~10s of no frames
                    self.view.connected = False
                    self.view.last_error = 'lost stream'
                    print(f"[{self.name}] lost stream, reconnecting", flush=True)
                    return
                time.sleep(0.1)
                continue
            consecutive_failures = 0

            start = time.time()
            self._frame_count += 1
            full_h, full_w = frame.shape[:2]

            # Motion on downscaled frame
            proc_w = 640
            scale = proc_w / full_w
            proc_frame = cv2.resize(frame, (proc_w, int(full_h * scale)))
            motion_regions = self.motion.detect(proc_frame)
            # Whole-frame illumination guard: an IR-cut / auto-exposure / dawn step lights up the
            # ENTIRE frame (high coverage AND every quadrant uniformly foreground). Treat that as a
            # background step, not motion -- unless a real object was seen in the last N frames.
            if (cfg.motion.global_change_max_fraction > 0
                    and self.motion.last_coverage >= cfg.motion.global_change_max_fraction
                    and self.motion.quadrant_min >= cfg.motion.global_change_min_quadrant
                    and not self._recent_real_object()):
                self.motion.relearn(proc_frame, -1)   # fold the new lighting into the model
                motion_regions = []
            merged = merge_overlapping_regions(motion_regions, padding=cfg.motion.merge_padding)

            inv_scale = full_w / proc_w
            merged_full = [
                (int(x1 * inv_scale), int(y1 * inv_scale),
                 int(x2 * inv_scale), int(y2 * inv_scale))
                for x1, y1, x2, y2 in merged
            ]
            if self.zone_filter:
                merged_full = self.zone_filter.filter_motion_regions(
                    merged_full, (full_h, full_w)
                )
            has_motion = len(merged_full) > 0

            # Detection (motion-gated, FPS-capped, shared detector)
            now = time.time()
            ran_detection = False
            if has_motion and (now - last_detect_time >= detect_interval):
                with self.detector_lock:
                    det_array = self.detector.detect_array(
                        frame, conf_threshold=cfg.detection.confidence,
                        classes=class_filter,
                    )
                detections = [
                    {'box': det[:4].tolist(), 'confidence': float(det[4]),
                     'class_id': int(det[5]),
                     'class_name': self.detector.class_names.get(int(det[5]), 'unknown')}
                    for det in det_array
                ]
                last_detect_time = now
                ran_detection = True
                if self.zone_filter and detections:
                    detections = self.zone_filter.filter_detections(detections, frame.shape)
                    det_array = np.array([
                        [*d['box'], d['confidence'], d['class_id']]
                        for d in detections
                    ]) if detections else np.empty((0, 6))

            tracks = self.tracker.update(det_array if ran_detection else np.empty((0, 6)))

            # Maintain per-track position history for the stationary filter. Only log positions on
            # frames where the track was actually measured (detected) so SORT's Kalman coast can't
            # synthesise displacement and make a parked car look like it's moving.
            now_pos = time.time()
            measured_ids = (
                {t.id for t in self.tracker.tracks if t.time_since_update == 0}
                if ran_detection else set()
            )
            for row in tracks:
                tid = int(row[4])
                if cfg.tracker.positions_measured_only and tid not in measured_ids:
                    continue
                cx = (row[0] + row[2]) * 0.5
                cy = (row[1] + row[3]) * 0.5
                hist = self._track_positions.setdefault(tid, deque(maxlen=240))
                hist.append((cx, cy, now_pos))

            moving_track_ids = {int(t[4]) for t in tracks if self._is_track_moving(int(t[4]))}

            if ran_detection:
                for t in self.tracker.tracks:
                    if t.time_since_update == 0 and detections:
                        track_box = t.get_state().flatten()
                        best_iou, best_det = 0, None
                        for det in detections:
                            iou = _box_iou(track_box, det['box'])
                            if iou > best_iou:
                                best_iou, best_det = iou, det
                        if best_det and best_iou > 0.3:
                            self._track_info[t.id] = {
                                'class_id': best_det['class_id'],
                                'class_name': best_det['class_name'],
                                'confidence': best_det['confidence'],
                                'box': best_det['box'],
                            }
                            self._track_scores.setdefault(
                                t.id, deque(maxlen=cfg.detection.confirm_window)
                            ).append(best_det['confidence'])

            confirmed_ids = {tid for tid in moving_track_ids if self._is_track_confirmed(tid)}
            # Spatially correlate the trigger with LIVE motion: a confirmed+moving track must
            # overlap a (dilated) motion region using its current Kalman box. Kills tree-only
            # triggers and parked cars that jitter but sit outside any fresh motion. The looser
            # has_moving_objects gate still sustains an in-progress recording.
            live_box = {int(r[4]): (r[0], r[1], r[2], r[3]) for r in tracks}
            if cfg.motion.require_motion_overlap and merged_full:
                pad = cfg.motion.overlap_dilate_px
                dilated = [(x1 - pad, y1 - pad, x2 + pad, y2 + pad)
                           for x1, y1, x2, y2 in merged_full]
                confirmed_ids = {tid for tid in confirmed_ids
                                 if tid in live_box and _overlaps_motion(live_box[tid], dilated)}
            has_moving_objects = len(confirmed_ids) > 0
            # Note when a real object was last seen, for the illumination-guard override.
            if confirmed_ids or (detections and any(d['class_id'] in (0, 2, 5, 7) for d in detections)):
                self._last_real_obj_frame = self._frame_count
            in_required_zone = (
                not self.zone_filter
                or self.zone_filter.check_required_zones(detections, frame.shape)
            ) if ran_detection else False

            display = frame.copy()
            for bx1, by1, bx2, by2 in merged_full:
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
            display = draw_tracks(display, tracks, self._track_info, self.detector.class_names)

            triggered = has_motion and has_moving_objects and in_required_zone
            sustain = has_motion and self.recording_mgr.is_recording and has_moving_objects
            # Count/score only the confirmed+moving objects, only on detection frames, so the
            # snapshot and top_label reflect what triggered the event, not parked cars.
            trigger_dets = [self._track_info[tid] for tid in confirmed_ids
                            if self._track_info.get(tid)]
            self.recording_mgr.update(frame, display, triggered or sustain, detections,
                                      trigger_detections=trigger_dets, count_now=ran_detection)

            current_event_id = self.recording_mgr.event_id
            if current_event_id != prev_event_id:
                ts = datetime.now().strftime('%H:%M:%S')
                if current_event_id and not prev_event_id:
                    self._event_count += 1
                    print(f"[{self.name}] [{ts}] recording (event #{self._event_count})", flush=True)
                elif prev_event_id and not current_event_id:
                    print(f"[{self.name}] [{ts}] saved", flush=True)
                prev_event_id = current_event_id

            # Capture crops for moving tracked objects (skip stationary)
            now = time.time()
            frame_captures = []
            for row in tracks:
                x1, y1, x2, y2, track_id = row
                track_id = int(track_id)
                info = self._track_info.get(track_id)
                if info is None:
                    continue
                if track_id not in confirmed_ids:
                    continue
                if not _overlaps_motion(info['box'], merged_full):
                    continue
                if now - self._track_last_capture.get(track_id, 0) < cfg.capture.interval:
                    continue
                if self._save_crop(frame, info['box'], track_id,
                                   info['class_name'], info['confidence']):
                    self._track_last_capture[track_id] = now
                    self._capture_count += 1
                    self._class_counts[info['class_name']] = (
                        self._class_counts.get(info['class_name'], 0) + 1
                    )
                    self._classes_seen[info['class_id']] = info['class_name']
                    frame_captures.append(info)

            if frame_captures:
                self._save_frame_with_labels(frame, frame_captures)

            # Uncertain detection capture
            if now - self._last_uncertain_save >= cfg.capture.uncertain_interval:
                uncertain = [
                    d for d in detections
                    if cfg.capture.uncertain_low <= d['confidence'] <= cfg.capture.uncertain_high
                ]
                if uncertain:
                    self._save_uncertain(frame, uncertain,
                                         datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    self._last_uncertain_save = now

            # Update web view (clean frame unless live_overlays; capped width to keep MJPEG light)
            now_t = time.time()
            frame_times.append(now_t - last_loop_time)
            last_loop_time = now_t
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0

            web_src = display if cfg.display.live_overlays else frame
            target_w = 1920
            if web_src.shape[1] > target_w:
                ws = target_w / web_src.shape[1]
                web_frame = cv2.resize(
                    web_src, (target_w, int(web_src.shape[0] * ws)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                web_frame = web_src
            with self.view.frame_cond:
                self.view.frame = web_frame
                self.view.frame_version += 1
                if self.view.stream_clients:
                    self.view.frame_cond.notify_all()
            self.view.fps = round(fps, 1)
            self.view.frame_count = self._frame_count
            self.view.event_count = self._event_count
            self.view.recording = self.recording_mgr.is_recording
            self.view.state = self.recording_mgr.state.value
            self.view.last_update = now_t

            # Periodic track-info cleanup
            if self._frame_count % 500 == 0:
                active_ids = {t.id for t in self.tracker.tracks}
                for sid in set(self._track_info) - active_ids:
                    self._track_info.pop(sid, None)
                    self._track_last_capture.pop(sid, None)
                    self._track_positions.pop(sid, None)
                    self._track_scores.pop(sid, None)

            elapsed = time.time() - start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

    def _is_track_moving(self, tid: int) -> bool:
        """True if the track's smoothed centroid shifted past a box-relative threshold over the
        recent window. Jitter-robust: needs several MEASURED samples and compares the mean of the
        first third vs the last third, so a parked car's detection jitter never reads as motion."""
        tcfg = self.cfg.tracker
        hist = self._track_positions.get(tid)
        if not hist or len(hist) < tcfg.stationary_min_samples:
            return False  # not enough measured evidence yet — don't trigger on a brand-new track
        now = hist[-1][2]
        recent = [p for p in hist if now - p[2] <= tcfg.stationary_window]
        if len(recent) < tcfg.stationary_min_samples:
            return False
        # Compare the mean of the first third vs the last third to smooth out box jitter.
        k = max(1, len(recent) // 3)
        sx0 = sum(p[0] for p in recent[:k]) / k
        sy0 = sum(p[1] for p in recent[:k]) / k
        sx1 = sum(p[0] for p in recent[-k:]) / k
        sy1 = sum(p[1] for p in recent[-k:]) / k
        disp = math.hypot(sx1 - sx0, sy1 - sy0)
        # Threshold scales with box size so a large parked car's jitter doesn't read as motion,
        # while a small distant walker still clears the absolute floor.
        min_disp = tcfg.stationary_min_displacement
        info = self._track_info.get(tid)
        if info and info.get('box'):
            bx1, by1, bx2, by2 = info['box']
            diag = math.hypot(bx2 - bx1, by2 - by1)
            min_disp = max(min_disp, tcfg.stationary_disp_box_frac * diag)
        return disp >= min_disp

    def _is_track_confirmed(self, tid: int) -> bool:
        """True once a track has at least confirm_min_count detections whose median
        score crosses the threshold — kills single-frame flicker without penalising
        short or mid-confidence tracks (Frigate-style)."""
        scores = self._track_scores.get(tid)
        if not scores or len(scores) < self.cfg.detection.confirm_min_count:
            return False
        threshold = self.cfg.detection.confirm_threshold
        info = self._track_info.get(tid)
        if info:
            threshold = self.cfg.detection.confirm_threshold_by_class.get(
                info['class_id'], threshold)
        return statistics.median(scores) >= threshold

    def _recent_real_object(self) -> bool:
        """True if a person/car/truck/bus was detected or a track confirmed within the last
        recent_real_object_frames frames. Vetoes the illumination guard so a genuine arrival that
        happens to light up the frame (e.g. headlights at night) is never suppressed."""
        return (self._frame_count - self._last_real_obj_frame
                <= self.cfg.motion.recent_real_object_frames)

    def _save_crop(self, frame, box, track_id, class_name, confidence, padding=20):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
        cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False
        class_dir = self.crops_dir / class_name.replace(' ', '_')
        class_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cv2.imwrite(str(class_dir / f"track{track_id}_{ts}_{confidence:.2f}.jpg"), crop)
        return True

    def _save_frame_with_labels(self, frame, capture_detections):
        h, w = frame.shape[:2]
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cv2.imwrite(str(self.dataset_dir / 'images' / f"{ts}.jpg"), frame)
        with open(self.dataset_dir / 'labels' / f"{ts}.txt", 'w') as f:
            for det in capture_detections:
                x1, y1, x2, y2 = det['box']
                cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _save_uncertain(self, frame, detections, timestamp_str):
        img_path = self.review_dir / f"{timestamp_str}.jpg"
        cv2.imwrite(str(img_path), frame)
        meta = {
            'timestamp': timestamp_str,
            'camera': self.name,
            'source': 'surveillance_auto',
            'detections': [
                {'class': d['class_name'], 'class_id': d['class_id'],
                 'confidence': round(d['confidence'], 3),
                 'box': [int(x) for x in d['box']]}
                for d in detections
            ],
        }
        with open(self.review_dir / f"{timestamp_str}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        h, w = frame.shape[:2]
        with open(self.review_dir / f"{timestamp_str}.txt", 'w') as f:
            for d in detections:
                x1, y1, x2, y2 = d['box']
                cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"{d['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def init_global_storage(cfg: VisionBoxConfig):
    """Ensure shared roots exist (the per-camera subdirs are created later)."""
    for d in [
        Path(cfg.recording.output_dir),
        Path(cfg.storage.crops),
        Path(cfg.storage.dataset),
        Path(cfg.storage.review),
        Path(cfg.storage.training),
        Path(cfg.storage.zones_dir),
    ]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='VisionBox multi-camera surveillance')
    parser.add_argument('--config', default='config.yml', help='Config file path')
    parser.add_argument('--ui-only', action='store_true',
                        help='Start web UI only (no cameras, browse past events)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open the browser')
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_global_storage(cfg)

    output_dir = Path(cfg.recording.output_dir)
    db = RecordingDatabase(output_dir / 'visionbox.db')

    cameras = cfg.enabled_cameras()

    state = CamerasState(
        config=cfg,
        db=db,
        output_dir=output_dir,
        crops_dir=Path(cfg.storage.crops),
        training_dir=Path(cfg.storage.training),
        zones_dir=Path(cfg.storage.zones_dir),
    )

    if args.ui_only or not cameras:
        if not cameras:
            print("No cameras configured; running UI-only.")
        state.offline = True
        port = cfg.display.web_port
        start_api_server(state, port, cfg.display.bind_host)
        print(f"VisionBox UI-only at http://0.0.0.0:{port}")
        if not args.no_browser:
            open_browser(f'http://localhost:{port}')
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda s, f: stop.set())
        signal.signal(signal.SIGTERM, lambda s, f: stop.set())
        stop.wait()
        db.close()
        return

    print(f"Loading model ({cfg.detection.model})...")
    detector = MultiModelDetector(
        [ModelConfig(cfg.detection.model, class_conf=cfg.detection.class_conf)],
        device=cfg.detection.device, imgsz=cfg.detection.imgsz,
    )
    detector_lock = threading.Lock()
    state.detector = detector
    state.detector_lock = detector_lock
    print(f"Model loaded ({detector.effective_device})")

    pipelines = []
    for name, cam_cfg in cameras.items():
        pipeline = CameraPipeline(cam_cfg, cfg, detector, detector_lock, db)
        pipelines.append(pipeline)
        state.add_camera(pipeline.view, pipeline.zone_filter, pipeline.recording_mgr)

    port = cfg.display.web_port
    start_api_server(state, port, cfg.display.bind_host)
    print(f"\nVisionBox running")
    print(f"  Cameras: {', '.join(p.name for p in pipelines)}")
    print(f"  Storage: {output_dir}")
    print(f"  Web UI:  http://0.0.0.0:{port}")
    print(f"  Press Ctrl+C to stop\n")

    for p in pipelines:
        p.start()

    if not args.no_browser:
        open_browser(f'http://localhost:{port}')

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop.set())

    def _reload_model(signum, frame):
        # SIGHUP -> hot-swap a freshly-promoted model. reload() builds + warms the new
        # model unlocked and takes detector_lock only for the sub-ms pointer swap, so
        # the cameras are never blind for more than a frame.
        def _do():
            try:
                detector.reload(swap_lock=detector_lock)
                print("[reload] model reloaded via SIGHUP", flush=True)
            except Exception as exc:
                print(f"[reload] failed: {type(exc).__name__}: {exc}", flush=True)
        threading.Thread(target=_do, daemon=True, name='model-reload').start()
    signal.signal(signal.SIGHUP, _reload_model)

    start_rss_watchdog(int(os.environ.get('VISIONBOX_MAX_RSS_MB', '4096')))

    try:
        stop.wait()
    finally:
        print("\nShutting down...")
        for p in pipelines:
            p.stop()
        db.close()
        print("Done.")


if __name__ == "__main__":
    main()
