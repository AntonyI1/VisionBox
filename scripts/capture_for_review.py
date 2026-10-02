#!/usr/bin/env python3
"""Auto-capture confident detections of the custom bottle class for review and retraining.

Each saved frame gets a .json sidecar (detections) and a YOLO .txt label in datasets/review/.

    python scripts/capture_for_review.py [camera_url] [--conf 0.7] [--interval 2] [--class-filter 81]

Keys: q quit, s force-save the current frame.
"""

import argparse
import json
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

from visionbox import create_surveillance_detector

REVIEW_DIR = ROOT / 'datasets' / 'review'


def save_for_review(frame, detections, confidence_threshold=0.7):
    """Save the frame, its detections and a YOLO label; returns True if anything was saved."""
    high_conf = [d for d in detections if d['confidence'] >= confidence_threshold]
    if not high_conf:
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(str(REVIEW_DIR / f"{timestamp}.jpg"), frame)

    meta = {
        'timestamp': timestamp,
        'detections': [
            {'class': d['class_name'], 'class_id': d['class_id'],
             'confidence': round(d['confidence'], 3), 'box': d['box']}
            for d in high_conf
        ],
    }
    with open(REVIEW_DIR / f"{timestamp}.json", 'w') as f:
        json.dump(meta, f, indent=2)

    h, w = frame.shape[:2]
    with open(REVIEW_DIR / f"{timestamp}.txt", 'w') as f:
        for d in high_conf:
            x1, y1, x2, y2 = d['box']
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # The bottle fine-tune is single-class, so every label is class 0 whatever the detector id.
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Capture high-confidence detections for review')
    parser.add_argument('url', nargs='?', help='Camera URL (default: CAMERA_URL from .env)')
    parser.add_argument('--conf', type=float, default=0.7, help='Minimum confidence to auto-save (default: 0.7)')
    parser.add_argument('--interval', type=float, default=2.0, help='Minimum seconds between saves (default: 2)')
    parser.add_argument('--class-filter', type=int, nargs='+', default=[81],
                        help='Class IDs to capture (default: 81 = custom bottle)')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        parser.error('camera URL required (argument or CAMERA_URL in .env)')

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    detector = create_surveillance_detector()

    print("\nAuto-capture settings:")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Min interval: {args.interval}s")
    print(f"  Class filter: {args.class_filter}")
    print(f"  Saving to: {REVIEW_DIR}/")
    print("\nPress 'q' to quit, 's' to force-save current frame")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    last_save = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect(frame, conf_threshold=0.2)
        filtered = [d for d in detections if d['class_id'] in args.class_filter]

        now = time.time()
        if now - last_save > args.interval and save_for_review(frame, filtered, args.conf):
            save_count += 1
            last_save = now
            print(f"  Auto-saved #{save_count}")

        display = frame.copy()
        for d in filtered:
            x1, y1, x2, y2 = d['box']
            conf = d['confidence']
            color = (0, 255, 0) if conf >= args.conf else (0, 165, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{d['class_name']} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(display, f"Saved: {save_count} | Conf threshold: {args.conf}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture for Review", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and save_for_review(frame, filtered, 0.0):
            save_count += 1
            print(f"  Force-saved #{save_count}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 50}")
    print(f"Captured {save_count} frames for review")
    print(f"Location: {REVIEW_DIR}/")
    print("\nNext steps:")
    print(f"1. Review images in {REVIEW_DIR}/")
    print("2. Delete bad ones, fix labels if needed")
    print("3. Move good ones to datasets/bottles/images/ and labels/")
    print("4. Retrain: yolo train model=models/bottle-custom.pt data=datasets/bottles/data.yaml epochs=20")


if __name__ == '__main__':
    main()
