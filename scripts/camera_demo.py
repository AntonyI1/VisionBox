#!/usr/bin/env python3
"""Live detection + tracking demo for one camera stream (OpenCV window).

    python scripts/camera_demo.py rtsp://... [--mode outdoor] [--conf 0.25]

Falls back to CAMERA_URL from .env. Keys: q quit, r reset tracks.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv

load_dotenv(ROOT / '.env')

import cv2
import numpy as np

from visionbox import CLASS_PRESETS_V2, Tracker, create_surveillance_detector
from visionbox.viz import draw_tracks


def main():
    parser = argparse.ArgumentParser(description='VisionBox detection demo')
    parser.add_argument('url', nargs='?', help='Camera URL (default: CAMERA_URL from .env)')
    parser.add_argument('--mode', choices=['outdoor', 'indoor', 'vehicles', 'all'], default='all',
                        help='Class preset to detect')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        parser.error('camera URL required (argument or CAMERA_URL in .env)')

    class_filter = CLASS_PRESETS_V2[args.mode]
    mode_name = args.mode.upper()

    print(f"Loading models... (mode: {mode_name})")
    detector = create_surveillance_detector()
    class_names = detector.class_names
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    print("Models loaded")

    if class_filter:
        filtered_names = [class_names.get(c, f'class_{c}') for c in class_filter]
        print(f"Detecting: {', '.join(filtered_names)}")
    else:
        print(f"Detecting: all {len(class_names)} classes")

    print(f"Connecting to {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("Press 'q' to quit, 'r' to reset tracks")

    frame_times = []
    track_classes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()
        det_array = detector.detect_array(frame, conf_threshold=args.conf, classes=class_filter)
        tracks = tracker.update(det_array)

        for t in tracker.tracks:
            if t.time_since_update == 0 and len(det_array) > 0:
                track_box = t.get_state()
                for det in det_array:
                    if np.allclose(track_box, det[:4], atol=50):
                        track_classes[t.id] = int(det[5])
                        break

        frame = draw_tracks(frame, tracks, track_classes, class_names)

        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)

        cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {mode_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("VisionBox", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            track_classes.clear()
            print("Tracks reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
