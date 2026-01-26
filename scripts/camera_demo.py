#!/usr/bin/env python3
"""Live camera detection demo with multi-model detection and tracking."""

import argparse
import os
import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

import cv2
import time
import numpy as np
from visionbox import Tracker, create_surveillance_detector, CLASS_PRESETS_V2

np.random.seed(42)
COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]


def draw_tracks(
    image: np.ndarray,
    tracks: np.ndarray,
    track_classes: dict[int, int],
    class_names: dict[int, str]
) -> np.ndarray:
    """Draw tracked bounding boxes with IDs and class names."""
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        # Get class name for this track
        class_id = track_classes.get(track_id, 0)
        class_name = class_names.get(class_id, f'class_{class_id}')

        # Consistent color per track ID
        color = COLORS[track_id % len(COLORS)]

        # Highlight license plates differently
        if class_id == 80:  # license_plate
            color = (0, 255, 255)  # Yellow for plates
            label = f"PLATE #{track_id}"
        else:
            label = f"{class_name} #{track_id}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='VisionBox Detection Demo')
    parser.add_argument('url', nargs='?', help='Camera URL')
    parser.add_argument('--mode', choices=['outdoor', 'indoor', 'vehicles', 'all'], default='all',
                        help='Detection mode: outdoor, indoor, vehicles, all')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--legacy', action='store_true', help='Use legacy YOLOv5 detector')
    args = parser.parse_args()

    stream_url = args.url or os.environ.get('CAMERA_URL')
    if not stream_url:
        print("Usage: python scripts/camera_demo.py <camera_url> [--mode outdoor|indoor|vehicles|all]")
        print("   or: CAMERA_URL=http://host:8080/video python scripts/camera_demo.py")
        sys.exit(1)

    class_filter = CLASS_PRESETS_V2[args.mode]
    mode_name = args.mode.upper()

    print(f"Loading models... (mode: {mode_name})")

    if args.legacy:
        # Legacy YOLOv5 detector
        from visionbox import Detector
        detector = Detector('yolov5s', device='cuda')
        class_names = {i: name for i, name in enumerate(detector.model.names)}
    else:
        # New multi-model detector (YOLOv8 + license plates)
        detector = create_surveillance_detector(device='cuda')
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
        print("ERROR: Could not open camera. Is webcam_server.py running on Windows?")
        return

    print("Press 'q' to quit, 'r' to reset tracks")

    frame_times = []
    track_classes = {}  # track_id -> class_id mapping

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()

        # Detection
        if args.legacy:
            detections = detector.detect(frame, conf_threshold=args.conf, iou_threshold=0.45)
            if class_filter:
                detections = [d for d in detections if d['class_id'] in class_filter]
            det_array = np.array([
                [*d['box'], d['confidence'], d['class_id']]
                for d in detections
            ]) if detections else np.empty((0, 6))
        else:
            det_array = detector.detect_array(frame, conf_threshold=args.conf, classes=class_filter)
            detections = [
                {'box': det[:4], 'confidence': det[4], 'class_id': int(det[5])}
                for det in det_array
            ]

        # Tracking
        tracks = tracker.update(det_array)

        # Update track->class mapping
        for t in tracker.tracks:
            if t.time_since_update == 0 and len(detections) > 0:
                track_box = t.get_state()
                for d in detections:
                    det_box = np.array(d['box'])
                    if np.allclose(track_box, det_box, atol=50):
                        track_classes[t.id] = d['class_id']
                        break

        # Draw
        frame = draw_tracks(frame, tracks, track_classes, class_names)

        # FPS
        frame_times.append(time.time() - start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)

        # Stats
        cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {mode_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("VisionBox", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            track_classes.clear()
            print("Tracks reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
