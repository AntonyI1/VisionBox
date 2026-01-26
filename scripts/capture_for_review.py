#!/usr/bin/env python3
"""
Auto-capture high-confidence detections for review and retraining.

Saves frames where the model is confident, so you can:
1. Review them
2. Correct any mistakes
3. Add to training data
4. Retrain for better accuracy

This is "active learning" - the model helps build its own training set.
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
from pathlib import Path
from datetime import datetime
import numpy as np
from visionbox import create_surveillance_detector, Tracker

# Where to save captures for review
REVIEW_DIR = Path('datasets/review')
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def save_for_review(frame, detections, confidence_threshold=0.7):
    """
    Save high-confidence detections for later review.

    Returns True if something was saved.
    """
    # Filter to high-confidence detections
    high_conf = [d for d in detections if d['confidence'] >= confidence_threshold]

    if not high_conf:
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Save image
    img_path = REVIEW_DIR / f"{timestamp}.jpg"
    cv2.imwrite(str(img_path), frame)

    # Save detections as JSON (for review)
    meta_path = REVIEW_DIR / f"{timestamp}.json"
    meta = {
        'timestamp': timestamp,
        'detections': [
            {
                'class': d['class_name'],
                'class_id': d['class_id'],
                'confidence': round(d['confidence'], 3),
                'box': d['box']
            }
            for d in high_conf
        ]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Also save YOLO format label (auto-label, needs review)
    label_path = REVIEW_DIR / f"{timestamp}.txt"
    h, w = frame.shape[:2]
    with open(label_path, 'w') as f:
        for d in high_conf:
            x1, y1, x2, y2 = d['box']
            # Convert to YOLO format (center x, center y, width, height) normalized
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # Use class 0 for bottle (you can change this)
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description='Capture high-confidence detections for review')
    parser.add_argument('url', nargs='?', help='Camera URL')
    parser.add_argument('--conf', type=float, default=0.7,
                        help='Minimum confidence to auto-save (default: 0.7)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Minimum seconds between saves (default: 2)')
    parser.add_argument('--class-filter', type=int, nargs='+', default=[81],
                        help='Class IDs to capture (default: 81 = custom bottle)')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        print("Usage: python scripts/capture_for_review.py [camera_url]")
        sys.exit(1)

    print("Loading models...")
    detector = create_surveillance_detector()

    print(f"\nAuto-capture settings:")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Min interval: {args.interval}s")
    print(f"  Class filter: {args.class_filter}")
    print(f"  Saving to: {REVIEW_DIR}/")
    print(f"\nPress 'q' to quit, 's' to force-save current frame")

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

        # Detect
        detections = detector.detect(frame, conf_threshold=0.2)

        # Filter to target classes
        filtered = [d for d in detections if d['class_id'] in args.class_filter]

        # Auto-save high confidence (with cooldown)
        now = time.time()
        if now - last_save > args.interval:
            if save_for_review(frame, filtered, args.conf):
                save_count += 1
                last_save = now
                print(f"  Auto-saved #{save_count}")

        # Draw detections
        for d in filtered:
            x1, y1, x2, y2 = d['box']
            conf = d['confidence']
            color = (0, 255, 0) if conf >= args.conf else (0, 165, 255)  # green if high conf
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d['class_name']} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Status
        cv2.putText(frame, f"Saved: {save_count} | Conf threshold: {args.conf}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capture for Review", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Force save
            if save_for_review(frame, filtered, 0.0):
                save_count += 1
                print(f"  Force-saved #{save_count}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Captured {save_count} frames for review")
    print(f"Location: {REVIEW_DIR}/")
    print(f"\nNext steps:")
    print(f"1. Review images in {REVIEW_DIR}/")
    print(f"2. Delete bad ones, fix labels if needed")
    print(f"3. Move good ones to datasets/bottles/images/ and labels/")
    print(f"4. Retrain: yolo train model=models/bottle-custom.pt data=datasets/bottles/data.yaml epochs=20")


if __name__ == '__main__':
    main()
