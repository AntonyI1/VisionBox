#!/usr/bin/env python3
"""
Full surveillance pipeline combining all VisionBox components.

Architecture:
1. Motion detection (CPU, ~1ms) → decides if GPU work is needed
2. Object detection (GPU, ~20ms) → only when motion present
3. Tracking → persistent IDs across frames
4. Event recording → clips saved when activity occurs
5. Detection capture → crops saved by class for review/training
6. Uncertain capture → low-confidence frames saved for review

Use --web to view the processed feed in a browser (works in WSL2).
"""

import argparse
import os
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
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from visionbox import (
    Tracker,
    CLASS_PRESETS_V2,
    MotionDetector,
    merge_overlapping_regions,
)
from visionbox.detector_v2 import MultiModelDetector, ModelConfig
from visionbox.recorder import EventRecorder

def open_browser(url: str):
    """Open URL in Windows browser from WSL2."""
    try:
        subprocess.Popen(
            ['cmd.exe', '/c', 'start', url.replace('&', '^&')],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass

np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]

# Output directories
REVIEW_DIR = Path('datasets/review')
CAPTURES_DIR = Path('captures/crops')
DATASET_DIR = Path('captures/dataset')

for d in [REVIEW_DIR, CAPTURES_DIR, DATASET_DIR / 'images', DATASET_DIR / 'labels']:
    d.mkdir(parents=True, exist_ok=True)


# --- Threaded camera reader (eliminates RTSP buffer lag) ---

class CameraStream:
    """Reads frames in a background thread so we always get the latest frame.

    Without this, OpenCV buffers RTSP frames and you end up processing
    video from 10-30 seconds ago.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        thread = threading.Thread(target=self._reader, daemon=True)
        thread.start()

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


# --- Web viewer (MJPEG stream) ---

class MJPEGHandler(BaseHTTPRequestHandler):
    latest_frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with MJPEGHandler.frame_lock:
                    frame = MJPEGHandler.latest_frame
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                time.sleep(0.067)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def start_web_viewer(port):
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def update_web_frame(frame):
    with MJPEGHandler.frame_lock:
        MJPEGHandler.latest_frame = frame


# --- Data saving ---

def save_uncertain(frame, detections, timestamp_str):
    """Save uncertain detections for human review (data flywheel)."""
    img_path = REVIEW_DIR / f"{timestamp_str}.jpg"
    cv2.imwrite(str(img_path), frame)

    meta_path = REVIEW_DIR / f"{timestamp_str}.json"
    meta = {
        'timestamp': timestamp_str,
        'source': 'surveillance_auto',
        'detections': [
            {
                'class': d['class_name'],
                'class_id': d['class_id'],
                'confidence': round(d['confidence'], 3),
                'box': [int(x) for x in d['box']],
            }
            for d in detections
        ]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    label_path = REVIEW_DIR / f"{timestamp_str}.txt"
    h, w = frame.shape[:2]
    with open(label_path, 'w') as f:
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{d['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def save_crop(frame, box, track_id, class_name, confidence, padding=20):
    """Save a cropped detection organized by class."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w, x2 + padding)
    cy2 = min(h, y2 + padding)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False

    folder = class_name.replace(' ', '_')
    class_dir = CAPTURES_DIR / folder
    class_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"track{track_id}_{ts}_{confidence:.2f}.jpg"
    cv2.imwrite(str(class_dir / filename), crop)
    return True


def save_frame_with_labels(frame, capture_detections):
    """Save full frame + YOLO labels for training."""
    h, w = frame.shape[:2]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv2.imwrite(str(DATASET_DIR / 'images' / f"{ts}.jpg"), frame)

    with open(DATASET_DIR / 'labels' / f"{ts}.txt", 'w') as f:
        for det in capture_detections:
            x1, y1, x2, y2 = det['box']
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# --- Drawing ---

def draw_tracks(image, tracks, track_info, class_names):
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        info = track_info.get(track_id)
        if info:
            class_id = info['class_id']
            class_name = info['class_name']
            conf = info['confidence']
        else:
            class_id = 0
            class_name = class_names.get(0, 'unknown')
            conf = 0

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


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='VisionBox Surveillance')
    parser.add_argument('url', nargs='?', help='Camera URL')
    parser.add_argument('--mode', choices=['outdoor', 'indoor', 'vehicles', 'all'],
                        default='outdoor')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--cooldown', type=float, default=10.0,
                        help='Recording cooldown after motion stops (seconds)')
    parser.add_argument('--min-area', type=int, default=2000,
                        help='Minimum motion area (default: 2000, increase for noisy feeds)')
    parser.add_argument('--capture-interval', type=float, default=10.0,
                        help='Seconds between captures of same tracked object (default: 10)')
    parser.add_argument('--uncertain-low', type=float, default=0.3,
                        help='Lower bound for uncertain detections')
    parser.add_argument('--uncertain-high', type=float, default=0.6,
                        help='Upper bound for uncertain detections')
    parser.add_argument('--uncertain-interval', type=float, default=5.0,
                        help='Min seconds between uncertain captures')
    parser.add_argument('--no-display', action='store_true', default=True,
                        help='Run headless (no window)')
    parser.add_argument('--web', action='store_true', default=True,
                        help='Stream processed feed to browser (http://localhost:8085)')
    parser.add_argument('--web-port', type=int, default=8085,
                        help='Web viewer port (default: 8085)')
    parser.add_argument('--max-fps', type=int, default=15,
                        help='Max loop FPS (default: 15)')
    parser.add_argument('--detect-fps', type=int, default=5,
                        help='Object detection FPS (default: 5, Frigate standard)')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        print("Usage: python scripts/surveillance.py <camera_url>")
        print("  Or set CAMERA_URL in .env")
        sys.exit(1)

    # Auto-detect: no display on WSL2/headless Linux
    if not args.no_display and sys.platform == 'linux':
        display_env = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
        if not display_env:
            args.no_display = True
            if not args.web:
                print("No display detected. Enabling --web for browser viewing.")
                args.web = True

    class_filter = CLASS_PRESETS_V2[args.mode]

    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Per-class confidence thresholds for noisy COCO classes.
    # Higher thresholds for classes that false-trigger on outdoor scenes.
    class_conf = {
        0: 0.35,   # person — most important, moderate threshold
        2: 0.45,   # car
        5: 0.50,   # bus — often confused with trucks/buildings
        7: 0.50,   # truck
        14: 0.50,  # bird — small, noisy
        15: 0.40,  # cat
        16: 0.40,  # dog
    }
    detector = MultiModelDetector(
        [ModelConfig('yolov8s.pt', class_conf=class_conf)],
        device=device, imgsz=1280
    )
    # Tuned for 5 FPS detection on 15 FPS loop:
    # - max_age=75: keep lost tracks 5 seconds (15fps * 5s)
    # - min_hits=2: confirm after 2 detections (400ms at 5 FPS detect)
    # - iou_threshold=0.2: looser matching compensates for Kalman drift
    # - max_coast=15: show Kalman predictions for 1 second between detections
    tracker = Tracker(max_age=75, min_hits=2, iou_threshold=0.2, max_coast=15)
    motion = MotionDetector(min_area=args.min_area)
    recorder = EventRecorder(cooldown=args.cooldown)
    print(f"Model loaded ({device})")

    streamlit_proc = None

    # Start web viewer
    if args.web:
        start_web_viewer(args.web_port)
        print(f"\n  Web viewer: http://localhost:{args.web_port}")
        open_browser(f'http://localhost:{args.web_port}')

    # Start Streamlit review app (only if web mode enabled)
    if args.web:
        streamlit_proc = subprocess.Popen(
            ['streamlit', 'run', 'scripts/review_app.py', '--server.headless', 'true'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"  Review app: http://localhost:8501")
        open_browser('http://localhost:8501')

    print(f"\nSurveillance settings:")
    print(f"  Mode: {args.mode}")
    print(f"  Confidence: {args.conf}")
    print(f"  Detection FPS: {args.detect_fps} (loop: {args.max_fps})")
    print(f"  Recording cooldown: {args.cooldown}s")
    print(f"  Capture interval: {args.capture_interval}s per object")
    print(f"  Uncertain range: {args.uncertain_low}-{args.uncertain_high}")
    print(f"  Outputs:")
    print(f"    Events:  recordings/events/")
    print(f"    Crops:   captures/crops/")
    print(f"    Dataset: captures/dataset/")
    print(f"    Review:  datasets/review/")
    if not args.no_display:
        print(f"\nControls: 'q' quit | 'r' reset")
    else:
        print(f"\nPress Ctrl+C to stop")

    cap = CameraStream(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    time.sleep(1)  # Let the reader thread grab the first frame

    # State
    track_info = {}             # track_id -> {class_id, class_name, confidence, box}
    track_last_capture = {}     # track_id -> last capture time
    frame_times = []
    last_uncertain_save = 0
    uncertain_count = 0
    event_count = 0
    capture_count = 0
    class_counts = {}           # class_name -> count
    classes_seen = {}           # class_id -> class_name
    frame_count = 0
    prev_recorder_state = recorder.state
    min_frame_time = 1.0 / args.max_fps
    last_loop_time = time.time()
    detect_interval = 1.0 / args.detect_fps
    last_detect_time = 0

    # Persistent state across frames (detections carry over between detect frames)
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

            # Downscale for motion detection (CPU, cheap, every frame)
            proc_w = 640
            scale = proc_w / full_w
            proc_h = int(full_h * scale)
            proc_frame = cv2.resize(frame, (proc_w, proc_h))

            # 1. Motion detection (every frame, ~1-2ms on downscaled)
            motion_regions = motion.detect(proc_frame)
            merged = merge_overlapping_regions(motion_regions, padding=30)
            has_motion = len(merged) > 0

            inv_scale = full_w / proc_w
            merged_full = [
                (int(x1 * inv_scale), int(y1 * inv_scale),
                 int(x2 * inv_scale), int(y2 * inv_scale))
                for x1, y1, x2, y2 in merged
            ]

            # 2. Object detection (GPU, only on motion, capped at detect_fps)
            # Frigate architecture: detect at 5 FPS, not every frame.
            # Between detections, the Kalman tracker predicts object positions.
            now = time.time()
            ran_detection = False
            if has_motion and (now - last_detect_time >= detect_interval):
                det_array = detector.detect_array(
                    frame, conf_threshold=args.conf, classes=class_filter
                )
                detections = [
                    {
                        'box': det[:4].tolist(),
                        'confidence': float(det[4]),
                        'class_id': int(det[5]),
                        'class_name': detector.class_names.get(int(det[5]), 'unknown'),
                    }
                    for det in det_array
                ]
                last_detect_time = now
                ran_detection = True

            # 3. Tracking (every frame — Kalman predicts between detections)
            if ran_detection:
                tracks = tracker.update(det_array)
            else:
                # No new detections — tracker predicts from velocity
                tracks = tracker.update(np.empty((0, 6)))

            # Match tracks to detections for class/confidence info
            if ran_detection:
                for t in tracker.tracks:
                    if t.time_since_update == 0 and len(detections) > 0:
                        track_box = t.get_state().flatten()
                        best_iou = 0
                        best_det = None
                        for det in detections:
                            iou = _box_iou(track_box, det['box'])
                            if iou > best_iou:
                                best_iou = iou
                                best_det = det
                        if best_det and best_iou > 0.3:
                            track_info[t.id] = {
                                'class_id': best_det['class_id'],
                                'class_name': best_det['class_name'],
                                'confidence': best_det['confidence'],
                                'box': best_det['box'],
                            }

            # 4. Event recording
            recorder.update(frame, has_motion, detections)
            h, w = full_h, full_w

            if recorder.state != prev_recorder_state:
                if recorder.state.value == 'recording':
                    event_count += 1
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Recording started (event #{event_count})")
                elif recorder.state.value == 'cooldown':
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Motion stopped, cooldown...")
                elif recorder.state.value == 'idle':
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Recording saved")
                prev_recorder_state = recorder.state

            # 5. Detection capture (crops by class + full frames)
            # Only capture objects that overlap with motion regions.
            # This filters out static objects (furniture, etc.) that the
            # detector sees but aren't actually moving.
            now = time.time()
            frame_captures = []

            for row in tracks:
                x1, y1, x2, y2, track_id = row
                track_id = int(track_id)

                info = track_info.get(track_id)
                if info is None:
                    continue

                # Skip static objects — only capture things that are moving
                det_box = info['box']
                if not _overlaps_motion(det_box, merged_full):
                    continue

                last = track_last_capture.get(track_id, 0)
                if now - last < args.capture_interval:
                    continue

                if save_crop(frame, det_box, track_id,
                             info['class_name'], info['confidence']):
                    track_last_capture[track_id] = now
                    capture_count += 1
                    class_counts[info['class_name']] = \
                        class_counts.get(info['class_name'], 0) + 1
                    classes_seen[info['class_id']] = info['class_name']
                    frame_captures.append(info)

            if frame_captures:
                save_frame_with_labels(frame, frame_captures)
                names = [c['class_name'] for c in frame_captures]
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Captured {len(frame_captures)}: {', '.join(names)}")

            # 6. Uncertain detection capture (data flywheel)
            if now - last_uncertain_save >= args.uncertain_interval:
                uncertain = [
                    d for d in detections
                    if args.uncertain_low <= d['confidence'] <= args.uncertain_high
                ]
                if uncertain:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_uncertain(frame, uncertain, ts)
                    uncertain_count += 1
                    last_uncertain_save = now

            # 7. Display / Web stream
            show_display = not args.no_display or args.web
            if show_display:
                display = frame.copy()

                for box in merged_full:
                    bx1, by1, bx2, by2 = box
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 255), 1)

                display = draw_tracks(display, tracks, track_info,
                                      detector.class_names)

                # Measure actual frame-to-frame time (includes sleep)
                now_t = time.time()
                frame_times.append(now_t - last_loop_time)
                last_loop_time = now_t
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = len(frame_times) / sum(frame_times)

                rec_status = recorder.state.value.upper()
                rec_color = {
                    'idle': (200, 200, 200),
                    'recording': (0, 0, 255),
                    'cooldown': (0, 165, 255),
                }[recorder.state.value]

                cv2.putText(display, f"FPS: {fps:.1f} | {rec_status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
                cv2.putText(display,
                            f"Tracks: {len(tracks)} | Captures: {capture_count} | "
                            f"Events: {event_count}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

                if recorder.is_recording:
                    cv2.circle(display, (display.shape[1] - 30, 30), 12,
                               (0, 0, 255), -1)

                if args.web:
                    # Scale to 1920px wide for browser
                    scale = 1920 / display.shape[1]
                    web_h = int(display.shape[0] * scale)
                    web_frame = cv2.resize(display, (1920, web_h))
                    update_web_frame(web_frame)

                if not args.no_display:
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

            # FPS limiter
            elapsed = time.time() - start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

            # Clean up stale track state
            if frame_count % 500 == 0:
                active_ids = {t.id for t in tracker.tracks}
                stale = set(track_info.keys()) - active_ids
                for sid in stale:
                    track_info.pop(sid, None)
                    track_last_capture.pop(sid, None)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.release()
        cap.release()
        cv2.destroyAllWindows()
        if streamlit_proc:
            streamlit_proc.terminate()

        if classes_seen:
            with open(DATASET_DIR / 'classes.txt', 'w') as f:
                for cid in sorted(classes_seen.keys()):
                    f.write(f"{cid}: {classes_seen[cid]}\n")

        print(f"\n{'='*50}")
        print(f"Session summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Events recorded:  {event_count}")
        print(f"  Detections saved: {capture_count}")
        if class_counts:
            print(f"  By class:")
            for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count}")
        print(f"  Uncertain:        {uncertain_count}")
        print(f"\n  Clips:   recordings/events/")
        print(f"  Crops:   captures/crops/")
        print(f"  Dataset: captures/dataset/")
        if capture_count > 0:
            print(f"\nRun 'streamlit run scripts/review_app.py' to review captures")


def _box_iou(a, b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _overlaps_motion(det_box, motion_boxes):
    """Check if a detection box overlaps any motion region."""
    dx1, dy1, dx2, dy2 = det_box
    for mx1, my1, mx2, my2 in motion_boxes:
        # Any overlap at all
        if dx1 < mx2 and dx2 > mx1 and dy1 < my2 and dy2 > my1:
            return True
    return False


if __name__ == "__main__":
    main()
