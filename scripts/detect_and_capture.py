#!/usr/bin/env python3
"""Detect, track and save training captures from a video source.

Writes clean crops to <output>/crops/<class>/ and full frames with YOLO labels to
<output>/dataset/{images,labels}/; tracking rate-limits captures per object.

    python scripts/detect_and_capture.py rtsp://... [--conf 0.4] [--output captures] [--interval 10]
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv

load_dotenv(ROOT / '.env')

import cv2
import numpy as np

from visionbox import Tracker, create_surveillance_detector
from visionbox.viz import COLORS, box_iou, draw_labeled_box


def main():
    parser = argparse.ArgumentParser(description='Detect and capture objects by class')
    parser.add_argument('source', nargs='?',
                        help='RTSP URL, video file or camera index (default: CAMERA_URL from .env)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--output', type=str, default='captures', help='Output directory (default: captures)')
    parser.add_argument('--interval', type=float, default=10.0,
                        help='Seconds between captures of the same tracked object (default: 10)')
    parser.add_argument('--padding', type=int, default=20, help='Pixels of padding around crops (default: 20)')
    parser.add_argument('--no-display', action='store_true', help='Run headless (no window)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Inference device: auto, cuda or cpu (default: auto)')
    args = parser.parse_args()

    source = args.source or os.environ.get('CAMERA_URL')
    if not source:
        parser.error('source required (argument or CAMERA_URL in .env)')

    output = Path(args.output)
    crops_dir = output / 'crops'
    images_dir = output / 'dataset' / 'images'
    labels_dir = output / 'dataset' / 'labels'
    for d in [crops_dir, images_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    detector = create_surveillance_detector(device=args.device)
    tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
    print("Ready\n")

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"ERROR: Could not open {source}")
        return

    track_class_info = {}
    track_last_capture = {}
    capture_count = 0
    frame_count = 0
    class_counts = {}
    classes_seen = {}

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

            detections = detector.detect(frame, conf_threshold=args.conf)
            if detections:
                det_array = np.array([[*d['box'], d['confidence'], d['class_id']] for d in detections],
                                     dtype=np.float32)
            else:
                det_array = np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(det_array)

            for t in tracker.tracks:
                if t.time_since_update == 0 and detections:
                    track_box = t.get_state().flatten()
                    best_iou, best_det = 0, None
                    for det in detections:
                        iou = box_iou(track_box, det['box'])
                        if iou > best_iou:
                            best_iou, best_det = iou, det
                    if best_det and best_iou > 0.3:
                        track_class_info[t.id] = {
                            'class_id': best_det['class_id'],
                            'class_name': best_det['class_name'],
                            'confidence': best_det['confidence'],
                        }

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

                if display is not None:
                    draw_labeled_box(display, (x1, y1, x2, y2), f"{class_name} #{track_id} {confidence:.0%}",
                                     COLORS[track_id % len(COLORS)])

                last = track_last_capture.get(track_id, 0)
                if not force_save and (now - last < args.interval):
                    continue

                # Crop from the clean frame so captures stay usable for training.
                pad = args.padding
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                class_dir = crops_dir / class_name.replace(' ', '_')
                class_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(str(class_dir / f"track{track_id}_{ts}_{confidence:.2f}.jpg"), crop)

                track_last_capture[track_id] = now
                capture_count += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                classes_seen[class_id] = class_name

                cx_norm = ((x1 + x2) / 2) / w
                cy_norm = ((y1 + y2) / 2) / h
                bw_norm = (x2 - x1) / w
                bh_norm = (y2 - y1) / h
                new_captures.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")

            force_save = False

            if new_captures:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(str(images_dir / f"{ts}.jpg"), frame)
                with open(labels_dir / f"{ts}.txt", 'w') as f:
                    f.write('\n'.join(new_captures))

                names = [track_class_info[int(r[4])]['class_name']
                         for r in tracks if int(r[4]) in track_class_info]
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Captured {len(new_captures)} object(s): {', '.join(names[:5])}")

            if frame_count % 1000 == 0:
                all_ids = {t.id for t in tracker.tracks}
                for sid in set(track_class_info) - all_ids:
                    track_class_info.pop(sid, None)
                    track_last_capture.pop(sid, None)

            if display is not None:
                frame_times.append(time.time() - start)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = len(frame_times) / sum(frame_times) if frame_times else 0
                status = f"FPS: {fps:.1f} | Captures: {capture_count} | Tracking: {len(tracks)}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('VisionBox - Detect & Capture', display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    force_save = True

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if classes_seen:
            with open(output / 'dataset' / 'classes.txt', 'w') as f:
                for cid in sorted(classes_seen):
                    f.write(f"{cid}: {classes_seen[cid]}\n")

        print(f"\n{'=' * 50}")
        print("Session summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total captures:   {capture_count}")
        if class_counts:
            print("  By class:")
            for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count}")
        print(f"\n  Crops:   {crops_dir.resolve()}/")
        print(f"  Dataset: {(output / 'dataset').resolve()}/")
        if classes_seen:
            print(f"  Classes: {output / 'dataset' / 'classes.txt'}")


if __name__ == '__main__':
    main()
