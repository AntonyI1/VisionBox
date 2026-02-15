# VisionBox

Self-hosted AI video surveillance system with motion-first detection, object tracking, dual recording, and a web dashboard.

## Features

- **Motion-first detection** — background subtraction identifies movement, YOLO runs only where motion occurs. Keeps GPU/CPU idle when nothing is happening.
- **Object tracking** — Kalman filter + Hungarian algorithm (SORT) maintains persistent IDs across frames, survives brief occlusions.
- **Dual recording** — clean FFmpeg stream (original quality) + annotated OpenCV stream (bounding boxes, labels) saved simultaneously.
- **Web dashboard** — live view, event browser, detection zone editor, crop review, and training data management.
- **Detection zones** — draw include/exclude polygons on the live view to control where detections trigger.
- **Active learning pipeline** — auto-captures detection crops for human review, approved crops move to training set.
- **YAML configuration** — all settings in `config.yml` with environment variable substitution.
- **SQLite event database** — every recording event logged with metadata, searchable from the dashboard.
- **Retention management** — automatic cleanup by age, storage budget, and per-label limits.
- **OpenVINO inference** — optimized for Intel CPUs (~37ms per frame on i5-8600).

## Quick Start

```bash
git clone https://github.com/AntonyI1/VisionBox.git
cd VisionBox

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.sample .env
# Edit .env — set CAMERA_URL and STORAGE_DIR

python scripts/surveillance.py
# Dashboard at http://localhost:8085
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CAMERA_URL` | RTSP URL for your IP camera |
| `STORAGE_DIR` | Base path for recordings, captures, and training data |

## Web Dashboard

The dashboard runs on port 8085 (configurable) with five tabs:

- **Live** — real-time MJPEG stream with bounding boxes, FPS, and recording status
- **Events** — browse recorded events with thumbnails, playback clean or annotated clips, delete events
- **Zones** — draw include/exclude polygons on a camera snapshot to control detection areas
- **Review** — review auto-captured detection crops, approve to training set or reject
- **Training** — browse approved training images by class, manage the training dataset

<!-- TODO: screenshots -->

## How It Works

```
Camera (RTSP) → Motion Detection (MOG2, ~1ms)
                      ↓ motion regions
                YOLO Detection (only where motion, capped at detect_fps)
                      ↓ detections
                Zone Filtering (exclude/include polygons)
                      ↓ filtered detections
                Kalman Tracker (predicts between detections)
                      ↓ tracks with persistent IDs
                Recording Manager (clean + annotated streams)
                      ↓
                SQLite DB + Web Dashboard
```

**Why motion-first?** Running YOLO on every frame at full resolution is expensive. Background subtraction costs ~1ms on CPU and eliminates 90%+ of frames. YOLO only runs at `detect_fps` (default 8) and only when motion is detected. The Kalman tracker predicts object positions between detections, so the display stays smooth at 15-30 FPS.

**Why dual recording?** Clean recordings preserve original quality for evidence. Annotated recordings show what the system detected, useful for debugging and review. Both start/stop together with configurable cooldown to avoid clip fragmentation.

## Architecture

```
src/visionbox/
├── api.py               # Flask REST API + web server
├── config.py             # YAML config with env var resolution
├── database.py           # SQLite event storage
├── detector_v2.py        # YOLOv8 multi-model detector (OpenVINO/CUDA/CPU)
├── kalman.py             # Kalman filter for single-box prediction
├── motion.py             # MOG2 background subtraction
├── recorder.py           # OpenCV event-triggered video writer
├── clean_recorder.py     # FFmpeg subprocess for clean RTSP recording
├── recording_manager.py  # Orchestrates dual recording + retention
├── tracker.py            # SORT multi-object tracker
├── zones.py              # Detection zone filtering (include/exclude polygons)
├── detection.py          # Legacy YOLOv5 detector
├── preprocessing.py      # Legacy preprocessing
├── nms.py                # Legacy NMS implementation
└── web/
    ├── index.html        # Dashboard SPA
    ├── app.js            # Dashboard logic (vanilla JS)
    └── style.css         # Dark theme styles

scripts/
├── surveillance.py       # Main pipeline (motion → detect → track → record)
├── camera_demo.py        # Live detection + tracking demo
├── motion_demo.py        # Motion detection demo
├── detect_and_capture.py # Standalone capture for training data
├── capture_for_review.py # Active learning capture
└── setup_models.py       # Download required models
```

## Configuration

All settings live in `config.yml`. Environment variables are resolved via `${VAR_NAME}` syntax.

```yaml
camera:
  url: ${CAMERA_URL}

storage:
  recordings: ${STORAGE_DIR}/recordings
  crops: ${STORAGE_DIR}/captures/crops
  training: ${STORAGE_DIR}/datasets/training

detection:
  mode: outdoor          # outdoor, indoor, vehicles, all
  confidence: 0.20
  detect_fps: 8
  model: yolov8n.pt
  imgsz: 640

recording:
  output_dir: ${STORAGE_DIR}/recordings
  clean:
    enabled: true
  annotated:
    enabled: true
    cooldown: 10.0
  retention:
    days: 30
    max_storage_gb: 1000
```

Detection modes filter which COCO classes are active. Per-class confidence thresholds can be set via `class_conf` to reduce false positives for specific classes.

## Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) — object detection
- **OpenVINO** — inference optimization for Intel CPUs
- **OpenCV** — video processing, background subtraction, annotated recording
- **FFmpeg** — clean RTSP stream recording
- **Flask** — REST API and web dashboard
- **SQLite** — event database
- **SciPy** — Hungarian algorithm for tracker assignment

## Roadmap

- [ ] Faster inference with INT8 precision
- [ ] Better detection at distance (tiled inference)
- [ ] One-click model retraining from approved crops
- [ ] Hardware acceleration with Coral TPU
