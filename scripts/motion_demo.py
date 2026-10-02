#!/usr/bin/env python3
"""Motion-gated detection demo: YOLO runs only on frames with foreground motion.

    python scripts/motion_demo.py rtsp://... [--mode outdoor] [--conf 0.25] [--min-area 1000]

Keys: m toggle motion gating (compare the detection rate), v overlay the foreground mask,
r reset tracker + background model, q quit.
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

from visionbox import CLASS_PRESETS_V2, MotionDetector, Tracker, create_surveillance_detector, merge_overlapping_regions
from visionbox.viz import draw_motion_regions, draw_tracks


def main():
    parser = argparse.ArgumentParser(description='Motion-gated detection demo')
    parser.add_argument('url', nargs='?', help='Camera URL (default: CAMERA_URL from .env)')
    parser.add_argument('--mode', choices=['outdoor', 'indoor', 'vehicles', 'all'], default='all')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--min-area', type=int, default=1000, help='Minimum motion area to trigger detection')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        parser.error('camera URL required (argument or CAMERA_URL in .env)')

    class_filter = CLASS_PRESETS_V2[args.mode]

    print("Loading models...")
    detector = create_surveillance_detector()
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    motion = MotionDetector(min_area=args.min_area)
    print("Models loaded")

    print("\nControls:")
    print("  'm' - Toggle motion-gating")
    print("  'v' - Toggle motion mask overlay")
    print("  'r' - Reset tracker and background model")
    print("  'q' - Quit")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    motion_gating = True
    show_mask = False
    track_classes = {}
    frame_count = 0
    detections_run = 0
    detections_skipped = 0
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()
        frame_count += 1

        motion_regions = motion.detect(frame)
        merged_regions = merge_overlapping_regions(motion_regions, padding=50)
        has_motion = len(merged_regions) > 0

        if has_motion or not motion_gating:
            detections_run += 1
            det_array = detector.detect_array(frame, conf_threshold=args.conf, classes=class_filter)
        else:
            detections_skipped += 1
            det_array = np.empty((0, 6))

        tracks = tracker.update(det_array)

        for t in tracker.tracks:
            if t.time_since_update == 0 and len(det_array) > 0:
                track_box = t.get_state()
                for det in det_array:
                    if np.allclose(track_box, det[:4], atol=50):
                        track_classes[t.id] = int(det[5])
                        break

        if show_mask:
            mask_color = cv2.cvtColor(motion.get_mask(frame), cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
        if motion_gating:
            frame = draw_motion_regions(frame, merged_regions)
        frame = draw_tracks(frame, tracks, track_classes, detector.class_names)

        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)
        total = detections_run + detections_skipped
        skip_rate = (detections_skipped / total * 100) if total > 0 else 0

        status = "MOTION-GATING ON" if motion_gating else "MOTION-GATING OFF"
        status_color = (0, 255, 0) if motion_gating else (0, 165, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Tracks: {len(tracks)} | Motion: {'YES' if has_motion else 'NO'}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detections_run} run, {detections_skipped} skipped ({skip_rate:.0f}% saved)",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("VisionBox - Motion First", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            motion_gating = not motion_gating
            detections_run = detections_skipped = 0
            print(f"Motion gating: {'ON' if motion_gating else 'OFF'}")
        elif key == ord('v'):
            show_mask = not show_mask
            print(f"Mask visualization: {'ON' if show_mask else 'OFF'}")
        elif key == ord('r'):
            tracker.reset()
            motion.reset()
            track_classes.clear()
            detections_run = detections_skipped = 0
            print("Reset tracker and motion model")

    cap.release()
    cv2.destroyAllWindows()

    total = detections_run + detections_skipped
    print(f"\n{'=' * 50}")
    print("Session stats:")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections run: {detections_run}")
    print(f"  Detections skipped: {detections_skipped}")
    if total > 0:
        print(f"  Efficiency: {detections_skipped / total * 100:.1f}% saved")


if __name__ == "__main__":
    main()
