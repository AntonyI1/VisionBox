#!/usr/bin/env python3
"""
Detect objects and save captures organized by class.

Connects to camera, runs detection + tracking, saves:
- Cropped detections to captures/crops/{class_name}/
- Full frames + YOLO labels to captures/dataset/ (for retraining)

Uses tracking to avoid saving the same object every frame.
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
import numpy as np
from pathlib import Path
from datetime import datetime
from visionbox import create_surveillance_detector, Tracker


np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]


def box_iou(a, b):
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


def main():
    parser = argparse.ArgumentParser(description='Detect and capture objects by class')
    parser.add_argument('source', nargs='?', help='Video source (RTSP URL, file, or camera index)')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='Confidence threshold (default: 0.4)')
    parser.add_argument('--output', type=str, default='captures',
                        help='Output directory (default: captures)')
    parser.add_argument('--interval', type=float, default=10.0,
                        help='Seconds between captures of same tracked object (default: 10)')
    parser.add_argument('--padding', type=int, default=20,
                        help='Pixels of padding around crops (default: 20)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run headless (no window)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/cpu)')
    args = parser.parse_args()

    source = args.source or os.environ.get('CAMERA_URL')
    if not source:
        print("Usage: python scripts/detect_and_capture.py <source>")
        print("  source: RTSP URL, video file, or camera index")
        print("  Or set CAMERA_URL in .env")
        sys.exit(1)

    # Output directories
    output = Path(args.output)
    crops_dir = output / 'crops'
    images_dir = output / 'dataset' / 'images'
    labels_dir = output / 'dataset' / 'labels'
    for d in [crops_dir, images_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load detector + tracker
    print("Loading models...")
    device = args.device or 'cuda'
    detector = create_surveillance_detector(device=device)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    print("Ready\n")

    # Open source
    src = source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Could not open {source}")
        return

    # State
    track_class_info = {}   # track_id -> {class_id, class_name, confidence}
    track_last_capture = {} # track_id -> last capture timestamp
    capture_count = 0
    frame_count = 0
    class_counts = {}       # class_name -> count
    classes_seen = {}       # class_id -> class_name

    print(f"Output:   {output.resolve()}/")
    print(f"Confidence: {args.conf}")
    print(f"Interval: {args.interval}s per tracked object")
    if not args.no_display:
        print("Controls: 'q' quit | 's' force-save all current detections")
    print()

    frame_times = []
    force_save = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            start = time.time()
            h, w = frame.shape[:2]
            now = time.time()
            frame_count += 1

            # --- Detection ---
            detections = detector.detect(frame, conf_threshold=args.conf)

            # Convert to array for tracker (avoid double inference)
            if detections:
                det_array = np.array([
                    [*d['box'], d['confidence'], d['class_id']]
                    for d in detections
                ], dtype=np.float32)
            else:
                det_array = np.empty((0, 6), dtype=np.float32)

            # --- Tracking ---
            tracks = tracker.update(det_array)

            # Match tracks to detections (get class info via IoU)
            for t in tracker.tracks:
                if t.time_since_update == 0 and detections:
                    track_box = t.get_state().flatten()
                    best_iou = 0
                    best_det = None
                    for det in detections:
                        iou = box_iou(track_box, det['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_det = det
                    if best_det and best_iou > 0.3:
                        track_class_info[t.id] = {
                            'class_id': best_det['class_id'],
                            'class_name': best_det['class_name'],
                            'confidence': best_det['confidence'],
                        }

            # --- Process tracks, save captures ---
            display = frame.copy() if not args.no_display else None
            new_captures = []

            for row in tracks:
                x1, y1, x2, y2, track_id = row
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                info = track_class_info.get(track_id)
                if info is None:
                    continue

                class_name = info['class_name']
                class_id = info['class_id']
                confidence = info['confidence']

                # Draw on display
                if display is not None:
                    color = COLORS[track_id % len(COLORS)]
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} #{track_id} {confidence:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
                    cv2.putText(display, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Rate-limit captures per track
                last = track_last_capture.get(track_id, 0)
                if not force_save and (now - last < args.interval):
                    continue

                # Save crop (clean — no box drawn, usable for training)
                pad = args.padding
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                folder_name = class_name.replace(' ', '_')
                class_dir = crops_dir / folder_name
                class_dir.mkdir(exist_ok=True)

                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                crop_filename = f"track{track_id}_{ts}_{confidence:.2f}.jpg"
                cv2.imwrite(str(class_dir / crop_filename), crop)

                track_last_capture[track_id] = now
                capture_count += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                classes_seen[class_id] = class_name

                # Collect YOLO label for full frame save
                cx_norm = ((x1 + x2) / 2) / w
                cy_norm = ((y1 + y2) / 2) / h
                bw_norm = (x2 - x1) / w
                bh_norm = (y2 - y1) / h
                new_captures.append(
                    f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"
                )

            force_save = False

            # Save full frame + YOLO labels when we captured something
            if new_captures:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(str(images_dir / f"{ts}.jpg"), frame)
                with open(labels_dir / f"{ts}.txt", 'w') as f:
                    f.write('\n'.join(new_captures))

                names = [track_class_info[int(r[4])]['class_name']
                         for r in tracks if int(r[4]) in track_class_info]
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Captured {len(new_captures)} object(s): "
                      f"{', '.join(names[:5])}")

            # Clean up stale track state (memory management for long sessions)
            if frame_count % 1000 == 0:
                active_ids = {int(r[4]) for r in tracks}
                all_ids = {t.id for t in tracker.tracks}
                stale = set(track_class_info.keys()) - all_ids
                for sid in stale:
                    track_class_info.pop(sid, None)
                    track_last_capture.pop(sid, None)

            # Display
            if display is not None:
                frame_times.append(time.time() - start)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = len(frame_times) / sum(frame_times) if frame_times else 0

                status = f"FPS: {fps:.1f} | Captures: {capture_count} | Tracking: {len(tracks)}"
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('VisionBox - Detect & Capture', display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    force_save = True

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Write classes.txt for the dataset
        if classes_seen:
            classes_path = output / 'dataset' / 'classes.txt'
            with open(classes_path, 'w') as f:
                for cid in sorted(classes_seen.keys()):
                    f.write(f"{cid}: {classes_seen[cid]}\n")

        print(f"\n{'='*50}")
        print(f"Session summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total captures:   {capture_count}")
        if class_counts:
            print(f"  By class:")
            for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count}")
        print(f"\n  Crops:   {crops_dir.resolve()}/")
        print(f"  Dataset: {(output / 'dataset').resolve()}/")
        if classes_seen:
            print(f"  Classes: {output / 'dataset' / 'classes.txt'}")


if __name__ == '__main__':
    main()
