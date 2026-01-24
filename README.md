# VisionBox

AI-powered video surveillance system with real-time object detection.

## Features

- Real-time YOLO object detection (60 FPS on RTX 4070)
- Custom preprocessing and NMS implementation
- WSL2 camera streaming from Windows

## Setup

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python ultralytics

# Start Windows camera server (in PowerShell)
python webcam_server.py

# Run detection (use your Windows IP from ipconfig)
python scripts/camera_demo.py http://<windows-ip>:8080/video
```

## Architecture

```
src/visionbox/
├── preprocessing.py   # Image → tensor pipeline
├── nms.py             # Non-maximum suppression
└── detection.py       # Detector class
```

## Usage

```python
from src.visionbox import Detector

detector = Detector('yolov5s', device='cuda')
detections = detector.detect(frame)

for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```
