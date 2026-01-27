#!/usr/bin/env python3
"""
Motion-first detection demo.

This demonstrates the key efficiency trick used by Frigate and other smart cameras:
- Motion detection is CHEAP (runs on CPU, ~1ms per frame)
- Object detection is EXPENSIVE (runs on GPU, ~20-50ms per frame)

By only running detection when motion is present, we can:
1. Save GPU/CPU resources on static scenes
2. Reduce power consumption
3. Process more cameras with the same hardware

Press 'm' to toggle motion-gating on/off and see the difference!
"""

import argparse
import os
import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

import cv2
import time
import numpy as np
from visionbox import (
    Tracker,
    create_surveillance_detector,
    CLASS_PRESETS_V2,
    MotionDetector,
    merge_overlapping_regions
)

np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]


def draw_motion_regions(image: np.ndarray, regions: list, color=(0, 0, 255)):
    """Draw motion bounding boxes (red by default)."""
    for box in regions:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    return image


def draw_tracks(
    image: np.ndarray,
    tracks: np.ndarray,
    track_classes: dict[int, int],
    class_names: dict[int, str]
) -> np.ndarray:
    """Draw tracked bounding boxes with IDs."""
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        class_id = track_classes.get(track_id, 0)
        class_name = class_names.get(class_id, f'class_{class_id}')
        color = COLORS[track_id % len(COLORS)]

        if class_id == 80:
            color = (0, 255, 255)
            label = f"PLATE #{track_id}"
        else:
            label = f"{class_name} #{track_id}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='Motion-First Detection Demo')
    parser.add_argument('url', nargs='?', help='Camera URL')
    parser.add_argument('--mode', choices=['outdoor', 'indoor', 'vehicles', 'all'], default='all')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Minimum motion area to trigger detection')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        print("Usage: python scripts/motion_demo.py <camera_url>")
        sys.exit(1)

    class_filter = CLASS_PRESETS_V2[args.mode]

    print("Loading models...")
    detector = create_surveillance_detector(device='cuda')
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    motion = MotionDetector(min_area=args.min_area)
    print("Models loaded")

    print(f"\nControls:")
    print("  'm' - Toggle motion-gating (see efficiency difference)")
    print("  'v' - Toggle motion mask visualization")
    print("  'r' - Reset tracker and background model")
    print("  'q' - Quit")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    # State
    motion_gating = True  # Start with motion-gating ON
    show_mask = False
    track_classes = {}

    # Stats
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

        # Always run motion detection (it's cheap!)
        motion_regions = motion.detect(frame)
        merged_regions = merge_overlapping_regions(motion_regions, padding=50)
        has_motion = len(merged_regions) > 0

        # Decide whether to run detection
        run_detection = has_motion if motion_gating else True

        if run_detection:
            detections_run += 1
            det_array = detector.detect_array(frame, conf_threshold=args.conf, classes=class_filter)
        else:
            detections_skipped += 1
            det_array = np.empty((0, 6))

        # Update tracker (even with no detections - maintains predictions)
        tracks = tracker.update(det_array)

        # Update track->class mapping
        for t in tracker.tracks:
            if t.time_since_update == 0 and len(det_array) > 0:
                track_box = t.get_state()
                for det in det_array:
                    det_box = det[:4]
                    if np.allclose(track_box, det_box, atol=50):
                        track_classes[t.id] = int(det[5])
                        break

        # Visualization
        if show_mask:
            mask = motion.get_mask(frame)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # Draw motion regions (thin red boxes)
        if motion_gating:
            frame = draw_motion_regions(frame, merged_regions)

        # Draw tracks
        frame = draw_tracks(frame, tracks, track_classes, detector.class_names)

        # FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)

        # Calculate efficiency
        total = detections_run + detections_skipped
        skip_rate = (detections_skipped / total * 100) if total > 0 else 0

        # Status display
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
        elif key == ord('m'):
            motion_gating = not motion_gating
            detections_run = 0
            detections_skipped = 0
            print(f"Motion gating: {'ON' if motion_gating else 'OFF'}")
        elif key == ord('v'):
            show_mask = not show_mask
            print(f"Mask visualization: {'ON' if show_mask else 'OFF'}")
        elif key == ord('r'):
            tracker.reset()
            motion.reset()
            track_classes.clear()
            detections_run = 0
            detections_skipped = 0
            print("Reset tracker and motion model")

    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    print(f"\n{'='*50}")
    print(f"Session stats:")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections run: {detections_run}")
    print(f"  Detections skipped: {detections_skipped}")
    if detections_run + detections_skipped > 0:
        print(f"  Efficiency: {detections_skipped / (detections_run + detections_skipped) * 100:.1f}% saved")


if __name__ == "__main__":
    main()
