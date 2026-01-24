#!/usr/bin/env python3
"""Live camera detection demo."""

import os
import sys
sys.path.insert(0, 'src')

import cv2
import time
import numpy as np
from visionbox import Detector

# Set CAMERA_URL env var or pass as argument
STREAM_URL = os.environ.get('CAMERA_URL') or (sys.argv[1] if len(sys.argv) > 1 else None)

if not STREAM_URL:
    print("Usage: python scripts/camera_demo.py <camera_url>")
    print("   or: CAMERA_URL=http://host:8080/video python scripts/camera_demo.py")
    sys.exit(1)

np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (80, 3))]


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes on image."""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = COLORS[det['class_id']]
        label = f"{det['class_name']}: {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main():
    print("Loading model...")
    detector = Detector('yolov5s', device='cuda')
    print("Model loaded")

    print(f"Connecting to {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Is webcam_server.py running on Windows?")
        return

    print("Press 'q' to quit")

    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()
        detections = detector.detect(frame, conf_threshold=0.25, iou_threshold=0.45)
        frame = draw_detections(frame, detections)

        # FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("VisionBox", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
