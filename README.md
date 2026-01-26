# VisionBox

AI-powered CCTV surveillance system with real-time object detection and tracking.

## Features

- **Multi-model detection**: YOLOv8 (80 COCO classes) + license plate detection
- **Object tracking**: Kalman filter + Hungarian algorithm (SORT)
- **Persistent IDs**: Track objects across frames, survive brief occlusions
- **Detection modes**: outdoor, indoor, vehicles, or all classes
- **Real-time**: 30-60 FPS on RTX GPU

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/VisionBox.git
cd VisionBox
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (CUDA 12.4 - adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Download models
python scripts/setup_models.py

# Run (see Camera Setup below)
python scripts/camera_demo.py http://<camera-ip>:8080/video
```

## Camera Setup

### Option 1: USB Webcam via Network (WSL2/Remote)

For WSL2 or when camera is on a different machine, run this server on Windows:

```python
# webcam_server.py - Run on Windows with webcam
from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def generate():
    while True:
        ret, frame = cap.read()
        if ret:
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Install Flask: `pip install flask opencv-python`

Then connect: `python scripts/camera_demo.py http://<windows-ip>:8080/video`

### Option 2: IP Camera (RTSP)

```bash
python scripts/camera_demo.py rtsp://user:pass@192.168.1.100:554/stream
```

### Option 3: Local Webcam

```bash
python scripts/camera_demo.py 0  # Device index
```

## Usage

```bash
# Detection modes
python scripts/camera_demo.py <url> --mode outdoor   # person, car, dog, license plates
python scripts/camera_demo.py <url> --mode indoor    # person, laptop, cup, chair, etc.
python scripts/camera_demo.py <url> --mode vehicles  # car, truck, bike, license plates
python scripts/camera_demo.py <url> --mode all       # all 81 classes

# Adjust confidence threshold
python scripts/camera_demo.py <url> --conf 0.3

# Controls
# q - quit
# r - reset tracks
```

## Architecture

```
src/visionbox/
├── detector_v2.py    # Multi-model YOLOv8 detector
├── kalman.py         # Kalman filter for motion prediction
├── tracker.py        # SORT multi-object tracker
├── detection.py      # Legacy YOLOv5 detector
├── preprocessing.py  # Legacy preprocessing
└── nms.py            # Legacy NMS implementation

scripts/
├── camera_demo.py    # Live detection demo
└── setup_models.py   # Download required models
```

## Detection Classes

**COCO (0-79)**: person, bicycle, car, motorcycle, bus, truck, dog, cat, bird, and 71 more

**Custom (80+)**: license_plate

## Python API

```python
from visionbox import create_surveillance_detector, Tracker

# Create detector (loads YOLOv8 + license plate model)
detector = create_surveillance_detector(device='cuda')

# Create tracker
tracker = Tracker(max_age=30, min_hits=3)

# Process frame
detections = detector.detect_array(frame, classes=[0, 2, 80])  # person, car, plate
tracks = tracker.update(detections)

for track in tracks:
    x1, y1, x2, y2, track_id = track
    print(f"Track #{int(track_id)} at ({x1:.0f}, {y1:.0f})")
```

## Roadmap

- [ ] Detection zones (entry/exit counting)
- [ ] Motion-triggered recording
- [ ] Fine-tuning on custom data
- [ ] Edge deployment (Coral TPU, OpenVINO)
- [ ] Web dashboard

## License

MIT
