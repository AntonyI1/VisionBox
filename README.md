# VisionBox

Self-hosted, multi-camera AI video surveillance. Motion-first detection, persistent object tracking,
dual recording (clean RTSP copy + annotated mp4), and a single web dashboard for live tiles, events,
detection zones, and the active-learning review queue.

## Features

- **Multi-camera, single process** — one worker thread per camera, one shared YOLO detector, one
  shared SQLite event database. The dashboard shows all cameras in a tile grid; click any tile to
  fullscreen.
- **Motion-first detection** — background subtraction identifies movement, YOLO runs only where
  motion occurs. Keeps the GPU/CPU idle when nothing is happening.
- **Object tracking** — Kalman filter + Hungarian (SORT) maintains persistent IDs across frames,
  survives brief occlusions.
- **Dual recording** — clean FFmpeg stream (original quality, zero re-encode) + annotated OpenCV
  stream (bounding boxes, labels) saved simultaneously per event.
- **Per-camera detection zones** — draw include/exclude polygons on a live snapshot.
- **Active learning pipeline** — auto-captures detection crops per camera; approved crops merge
  into a global training pool.
- **YAML configuration** — all settings in `config.yml` with `${VAR}` env substitution.
- **Retention management** — automatic cleanup by age, storage budget, or per-label limits.
- **OpenVINO inference** — optimized for Intel CPUs (~37ms per frame on an i5-8600).

## Quick Start

```bash
git clone https://github.com/AntonyI1/VisionBox.git
cd VisionBox

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/setup_models.py     # downloads + exports yolov8n OpenVINO model

cp .env.sample .env
# Edit .env — set STORAGE_DIR and any camera credentials referenced in config.yml

# Add your cameras to config.yml (see Configuration below), then:
python scripts/surveillance.py
# Dashboard at http://localhost:8085
```

## Configuration

`config.yml` defines all cameras and shared defaults. Each camera gets its own RTSP URL and
optionally per-camera overrides; `${VAR}` placeholders resolve from environment / `.env`.

```yaml
cameras:
  front_door:
    url: rtsp://admin:${EMPIRETECH_PASS}@192.168.1.50:554/cam/realmonitor?channel=1&subtype=1
  backyard:
    url: rtsp://${CAM_USER}:${CAM_PASS}@192.168.1.101:554/stream1
    enabled: false   # comment-friendly switch

storage:
  recordings: ${STORAGE_DIR}/recordings   # per-camera subdir created automatically
  crops:      ${STORAGE_DIR}/captures/crops
  zones_dir:  ${STORAGE_DIR}/zones        # per-camera <name>.json files

detection:
  mode: outdoor       # outdoor | indoor | vehicles | all
  confidence: 0.20
  detect_fps: 5
  model: yolov8n.pt
  imgsz: 640

recording:
  output_dir: ${STORAGE_DIR}/recordings
  clean:     { enabled: true }
  annotated: { enabled: true, fps: 15.0, cooldown: 10.0 }
  retention: { days: 30, max_storage_gb: 0, max_per_label: 0, priority_labels: [person] }
```

Per-camera assets land at:
- `recordings/<camera>/{clean,annotated,thumbnails}/`
- `captures/crops/<camera>/<class>/`
- `captures/dataset/<camera>/{images,labels}/`
- `zones/<camera>.json`

The SQLite event DB is shared at `recordings/visionbox.db` and tags every event with its camera.

## Web Dashboard

`http://<host>:8085` — dark theme, five tabs:

- **Live** — tile grid of every enabled camera; click a tile to focus a full-screen view.
- **Events** — browse recordings, filter by camera, play clean or annotated clips, delete events.
- **Zones** — pick a camera, draw include/exclude polygons on its live snapshot.
- **Review** — pick a camera + class, approve/reject crops with keyboard shortcuts (`A` / `R`).
- **Training** — browse the approved training pool (global across cameras).

## How It Works

```
Per camera (worker thread):
  RTSP → CameraStream (always-latest frame)
    → MOG2 motion (≈1ms CPU, on downscaled frame)
    → motion-gated YOLO via shared detector (lock-serialized)
    → zone filtering (exclude masks, required-zone gate)
    → SORT tracker (Kalman predict between detections)
    → RecordingManager → clean RTSP copy + annotated mp4 + SQLite event row + thumbnail
    → API view buffer for the dashboard MJPEG endpoint
```

Shared across cameras:
- One `MultiModelDetector` (OpenVINO/CUDA/CPU auto-detect) behind a lock
- One `RecordingDatabase` at `<recordings>/visionbox.db`
- One Flask API server

## Hosting (systemd)

```bash
sudo install -m 644 visionbox.service /etc/systemd/system/visionbox.service
sudo systemctl daemon-reload
sudo systemctl enable --now visionbox.service
journalctl -u visionbox.service -f
```

## Architecture

```
src/visionbox/
├── api.py               # Flask REST + dashboard, CamerasState + CameraView
├── config.py            # YAML config with env var resolution, per-camera path helpers
├── database.py          # SQLite event storage (shared, tagged by camera)
├── detector_v2.py       # YOLOv8 multi-model detector (OpenVINO/CUDA/CPU)
├── kalman.py / tracker.py / motion.py / nms.py / preprocessing.py
├── recorder.py          # OpenCV-driven event recorder (annotated mp4)
├── clean_recorder.py    # FFmpeg subprocess for zero-CPU RTSP copy
├── recording_manager.py # Orchestrates dual recording + retention per camera
├── zones.py             # Per-camera include/exclude polygon filtering
└── web/                 # Dashboard SPA (index.html, app.js, style.css)

scripts/
├── surveillance.py      # Main: spawns one CameraPipeline thread per enabled camera
└── setup_models.py      # Downloads + exports OpenVINO model

config.yml               # The thing you edit
.env / .env.sample       # Secrets + STORAGE_DIR
visionbox.service        # systemd unit
```

## Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) — object detection
- **OpenVINO** — inference optimization for Intel CPUs
- **OpenCV** — video processing, background subtraction, annotated recording
- **FFmpeg** — clean RTSP stream copy
- **Flask** — REST API and dashboard
- **SQLite** — event database
- **SciPy** — Hungarian algorithm for tracker assignment

## Roadmap

- [ ] INT8 quantization
- [ ] Tiled inference for better distance detection
- [ ] One-click retraining from approved crops
- [ ] Coral TPU offload
- [ ] Per-camera detection presets (currently shared)
