#!/usr/bin/env python3
"""VisionBox surveillance pipeline. Config via config.yml."""

import argparse
import os
import signal
import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

import cv2
import time
import json
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from visionbox import (
    Tracker,
    CLASS_PRESETS_V2,
    MotionDetector,
    merge_overlapping_regions,
)
from visionbox.detector_v2 import MultiModelDetector, ModelConfig
from visionbox.config import load_config
from visionbox.database import RecordingDatabase
from visionbox.recording_manager import RecordingManager
from visionbox.api import PipelineState, start_api_server
from visionbox.zones import ZoneFilter


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


np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]

REVIEW_DIR = Path('/mnt/storage/visionbox/datasets/review')
CAPTURES_DIR = Path('/mnt/storage/visionbox/captures/crops')
DATASET_DIR = Path('/mnt/storage/visionbox/captures/dataset')

for d in [REVIEW_DIR, CAPTURES_DIR, DATASET_DIR / 'images', DATASET_DIR / 'labels']:
    d.mkdir(parents=True, exist_ok=True)


class CameraStream:
    """Threaded RTSP reader — always returns the latest frame."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.cap.release()



def save_uncertain(frame, detections, timestamp_str):
    img_path = REVIEW_DIR / f"{timestamp_str}.jpg"
    cv2.imwrite(str(img_path), frame)

    meta = {
        'timestamp': timestamp_str,
        'source': 'surveillance_auto',
        'detections': [
            {'class': d['class_name'], 'class_id': d['class_id'],
             'confidence': round(d['confidence'], 3),
             'box': [int(x) for x in d['box']]}
            for d in detections
        ]
    }
    with open(REVIEW_DIR / f"{timestamp_str}.json", 'w') as f:
        json.dump(meta, f, indent=2)

    h, w = frame.shape[:2]
    with open(REVIEW_DIR / f"{timestamp_str}.txt", 'w') as f:
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            f.write(f"{d['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def save_crop(frame, box, track_id, class_name, confidence, padding=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
    cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False

    class_dir = CAPTURES_DIR / class_name.replace(' ', '_')
    class_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv2.imwrite(str(class_dir / f"track{track_id}_{ts}_{confidence:.2f}.jpg"), crop)
    return True


def save_frame_with_labels(frame, capture_detections):
    h, w = frame.shape[:2]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv2.imwrite(str(DATASET_DIR / 'images' / f"{ts}.jpg"), frame)

    with open(DATASET_DIR / 'labels' / f"{ts}.txt", 'w') as f:
        for det in capture_detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


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


def main():
    parser = argparse.ArgumentParser(description='VisionBox Surveillance')
    parser.add_argument('url', nargs='?', help='Camera URL (overrides config)')
    parser.add_argument('--config', default='config.yml', help='Config file path')
    parser.add_argument('--ui-only', action='store_true',
                        help='Start web UI only (no camera/detection)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.ui_only:
        output_dir = Path(cfg.recording.output_dir)
        db_path = output_dir / 'visionbox.db'
        db = RecordingDatabase(db_path)
        zone_filter = ZoneFilter('zones.json')

        api_state = PipelineState(
            zone_filter=zone_filter,
            config=cfg,
            offline=True,
            db=db,
            output_dir=output_dir,
        )

        port = cfg.display.web_port
        start_api_server(api_state, port)
        print(f"VisionBox UI-only mode")
        print(f"  Web UI: http://localhost:{port}")
        print(f"  Database: {db_path}")
        print(f"  Press Ctrl+C to stop")
        open_browser(f'http://localhost:{port}')

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda s, f: stop.set())
        signal.signal(signal.SIGTERM, lambda s, f: stop.set())
        stop.wait()
        db.close()
        return

    stream_url = args.url or cfg.camera.url or os.environ.get('CAMERA_URL', '')
    if cfg.camera.test_input:
        stream_url = cfg.camera.test_input
    elif not stream_url:
        print("Usage: python scripts/surveillance.py <camera_url>")
        print("  Or set CAMERA_URL in .env or config.yml")
        sys.exit(1)

    no_display = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    class_filter = CLASS_PRESETS_V2[cfg.detection.mode]

    print("Loading model...")
    detector = MultiModelDetector(
        [ModelConfig(cfg.detection.model, class_conf=cfg.detection.class_conf)],
        device='auto', imgsz=cfg.detection.imgsz
    )
    tracker = Tracker(
        max_age=cfg.tracker.max_age, min_hits=cfg.tracker.min_hits,
        iou_threshold=cfg.tracker.iou_threshold, max_coast=cfg.tracker.max_coast,
    )
    motion = MotionDetector(min_area=cfg.motion.min_area)
    recording_mgr = RecordingManager(cfg.recording, rtsp_url=stream_url)
    recording_mgr.start()
    zone_filter = ZoneFilter('zones.json')
    print(f"Model loaded ({detector.device})")

    clean_status = "enabled" if recording_mgr.clean and recording_mgr.clean.available else "disabled"
    annotated_status = "enabled" if recording_mgr.annotated else "disabled"
    print(f"  Clean recording: {clean_status}")
    print(f"  Annotated recording: {annotated_status}")

    api_state = PipelineState(
        recording_mgr=recording_mgr,
        zone_filter=zone_filter,
        config=cfg,
    )

    if cfg.display.web:
        start_api_server(api_state, cfg.display.web_port)
        print(f"\n  Web UI: http://localhost:{cfg.display.web_port}")
        open_browser(f'http://localhost:{cfg.display.web_port}')

    print(f"\nSurveillance settings:")
    print(f"  Mode: {cfg.detection.mode} | Confidence: {cfg.detection.confidence}")
    print(f"  Detection FPS: {cfg.detection.detect_fps} | Loop FPS: {cfg.display.max_fps}")
    print(f"  Recording cooldown: {cfg.recording.annotated.cooldown}s")
    print(f"  Outputs: {cfg.recording.output_dir}/ | captures/ | datasets/review/")
    print(f"\n{'Controls: q quit | r reset' if not no_display else 'Press Ctrl+C to stop'}")

    cap = CameraStream(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    time.sleep(1)

    track_info = {}
    track_last_capture = {}
    frame_times = []
    last_uncertain_save = 0
    uncertain_count = 0
    event_count = 0
    capture_count = 0
    class_counts = {}
    classes_seen = {}
    frame_count = 0
    prev_recorder_state = recording_mgr.state
    min_frame_time = 1.0 / cfg.display.max_fps
    last_loop_time = time.time()
    detect_interval = 1.0 / cfg.detection.detect_fps
    last_detect_time = 0
    det_array = np.empty((0, 6))
    detections = []
    merged_full = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            start = time.time()
            frame_count += 1
            full_h, full_w = frame.shape[:2]

            # Motion detection on downscaled frame
            proc_w = 640
            scale = proc_w / full_w
            proc_frame = cv2.resize(frame, (proc_w, int(full_h * scale)))

            motion_regions = motion.detect(proc_frame)
            merged = merge_overlapping_regions(motion_regions, padding=30)
            has_motion = len(merged) > 0

            inv_scale = full_w / proc_w
            merged_full = [
                (int(x1 * inv_scale), int(y1 * inv_scale),
                 int(x2 * inv_scale), int(y2 * inv_scale))
                for x1, y1, x2, y2 in merged
            ]

            # Exclude zones act as motion masks — drop motion in masked areas
            if zone_filter:
                merged_full = zone_filter.filter_motion_regions(
                    merged_full, (full_h, full_w)
                )
            has_motion = len(merged_full) > 0

            # Object detection (only on unmasked motion, capped at detect_fps)
            now = time.time()
            ran_detection = False
            if has_motion and (now - last_detect_time >= detect_interval):
                det_array = detector.detect_array(
                    frame, conf_threshold=cfg.detection.confidence, classes=class_filter
                )
                detections = [
                    {'box': det[:4].tolist(), 'confidence': float(det[4]),
                     'class_id': int(det[5]),
                     'class_name': detector.class_names.get(int(det[5]), 'unknown')}
                    for det in det_array
                ]
                last_detect_time = now
                ran_detection = True

                # Exclude zones — drop detections in masked areas
                if zone_filter and detections:
                    detections = zone_filter.filter_detections(detections, frame.shape)
                    det_array = np.array([
                        [*d['box'], d['confidence'], d['class_id']]
                        for d in detections
                    ]) if detections else np.empty((0, 6))

            # Tracking
            tracks = tracker.update(det_array if ran_detection else np.empty((0, 6)))

            if ran_detection:
                for t in tracker.tracks:
                    if t.time_since_update == 0 and len(detections) > 0:
                        track_box = t.get_state().flatten()
                        best_iou, best_det = 0, None
                        for det in detections:
                            iou = _box_iou(track_box, det['box'])
                            if iou > best_iou:
                                best_iou, best_det = iou, det
                        if best_det and best_iou > 0.3:
                            track_info[t.id] = {
                                'class_id': best_det['class_id'],
                                'class_name': best_det['class_name'],
                                'confidence': best_det['confidence'],
                                'box': best_det['box'],
                            }

            # Build annotated frame, then record
            has_objects = len(tracks) > 0
            in_required_zone = (
                not zone_filter
                or zone_filter.check_required_zones(detections, frame.shape)
            ) if ran_detection else False
            display = frame.copy()
            for bx1, by1, bx2, by2 in merged_full:
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
            display = draw_tracks(display, tracks, track_info, detector.class_names)

            triggered = has_motion and has_objects and in_required_zone
            sustain = has_motion and recording_mgr.is_recording and has_objects
            recording_mgr.update(frame, display, triggered or sustain, detections)

            if recording_mgr.state != prev_recorder_state:
                ts = datetime.now().strftime('%H:%M:%S')
                if recording_mgr.state.value == 'recording':
                    event_count += 1
                    print(f"  [{ts}] Recording started (event #{event_count})")
                elif recording_mgr.state.value == 'cooldown':
                    print(f"  [{ts}] Motion stopped, cooldown...")
                elif recording_mgr.state.value == 'idle':
                    print(f"  [{ts}] Recording saved")
                prev_recorder_state = recording_mgr.state

            # Capture crops for moving objects only
            now = time.time()
            frame_captures = []
            for row in tracks:
                x1, y1, x2, y2, track_id = row
                track_id = int(track_id)
                info = track_info.get(track_id)
                if info is None:
                    continue
                if not _overlaps_motion(info['box'], merged_full):
                    continue
                if now - track_last_capture.get(track_id, 0) < cfg.capture.interval:
                    continue
                if save_crop(frame, info['box'], track_id, info['class_name'], info['confidence']):
                    track_last_capture[track_id] = now
                    capture_count += 1
                    class_counts[info['class_name']] = class_counts.get(info['class_name'], 0) + 1
                    classes_seen[info['class_id']] = info['class_name']
                    frame_captures.append(info)

            if frame_captures:
                save_frame_with_labels(frame, frame_captures)
                names = [c['class_name'] for c in frame_captures]
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Captured {len(frame_captures)}: {', '.join(names)}")

            # Uncertain detection capture
            if now - last_uncertain_save >= cfg.capture.uncertain_interval:
                uncertain = [
                    d for d in detections
                    if cfg.capture.uncertain_low <= d['confidence'] <= cfg.capture.uncertain_high
                ]
                if uncertain:
                    save_uncertain(frame, uncertain, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                    uncertain_count += 1
                    last_uncertain_save = now

            # Display / Web stream
            if not no_display or cfg.display.web:
                now_t = time.time()
                frame_times.append(now_t - last_loop_time)
                last_loop_time = now_t
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = len(frame_times) / sum(frame_times)

                rec_color = {'idle': (200, 200, 200), 'recording': (0, 0, 255),
                             'cooldown': (0, 165, 255)}[recording_mgr.state.value]
                cv2.putText(display, f"FPS: {fps:.1f} | {recording_mgr.state.value.upper()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
                cv2.putText(display,
                            f"Tracks: {len(tracks)} | Captures: {capture_count} | Events: {event_count}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if recording_mgr.is_recording:
                    cv2.circle(display, (display.shape[1] - 30, 30), 12, (0, 0, 255), -1)

                if cfg.display.web:
                    ws = 1920 / display.shape[1]
                    web_frame = cv2.resize(display, (1920, int(display.shape[0] * ws)))
                    with api_state.frame_lock:
                        api_state.frame = web_frame
                    api_state.fps = fps
                    api_state.frame_count = frame_count
                    api_state.event_count = event_count

                if not no_display:
                    cv2.imshow("VisionBox Surveillance", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        tracker.reset()
                        motion.reset()
                        track_info.clear()
                        track_last_capture.clear()
                        print("Reset")

            elapsed = time.time() - start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

            if frame_count % 500 == 0:
                active_ids = {t.id for t in tracker.tracks}
                for sid in set(track_info) - active_ids:
                    track_info.pop(sid, None)
                    track_last_capture.pop(sid, None)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recording_mgr.stop()
        cap.release()
        cv2.destroyAllWindows()
        if classes_seen:
            with open(DATASET_DIR / 'classes.txt', 'w') as f:
                for cid in sorted(classes_seen):
                    f.write(f"{cid}: {classes_seen[cid]}\n")

        print(f"\n{'='*50}")
        print(f"Session summary:")
        print(f"  Frames: {frame_count} | Events: {event_count} | Captures: {capture_count}")
        if class_counts:
            for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count}")
        print(f"  Uncertain: {uncertain_count}")
        print(f"  Outputs: {cfg.recording.output_dir}/ | captures/ | datasets/review/")


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


if __name__ == "__main__":
    main()
