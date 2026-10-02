# VisionBox

Self-hosted, multi-camera AI video surveillance. Motion-first detection, persistent object tracking,
dual recording (clean RTSP copy + annotated mp4), and a single web dashboard for live tiles, events,
detection zones, and the active-learning review queue. Runs as one process on a small Intel box
(OpenVINO on the iGPU), with a cron watchdog for the cameras and an optional overnight
self-training loop that retrains on the footage it collects.

## Features

- **Multi-camera, single process** — one worker thread per camera, one shared YOLO detector, one
  shared SQLite event database. The dashboard shows all cameras in a tile grid; click any tile to
  fullscreen.
- **Motion-first detection** — MOG2 background subtraction on a downscaled frame finds movement;
  YOLO runs only where motion occurs, at a capped per-camera rate (`detect_fps`).
- **Object tracking** — Kalman filter + Hungarian assignment (SORT) keeps persistent IDs across
  frames and coasts through missed detections.
- **False-positive gates** — a track only counts once the median of its recent detection scores
  clears a (per-class) threshold; an event needs a confirmed, *moving* track that overlaps live
  motion; a whole-frame illumination guard ignores IR-cut and exposure steps.
- **Dual recording** — clean FFmpeg stream copy (original quality, no re-encode) + annotated OpenCV
  mp4 (boxes, labels) per event, plus a thumbnail and a best-frame snapshot. Runaway and broken
  clips are bounded and cleaned up.
- **Per-camera detection zones** — draw include/exclude polygons on a live snapshot.
- **Active learning** — detection crops are captured per camera; approve or reject them in the
  dashboard and approved crops merge into a global training pool.
- **Self-training** — nightly CPU fine-tune on the box, gated by a base-class mAP regression check
  against a frozen COCO replay set, exported to OpenVINO and hot-swapped into the running service.
- **Camera doctor** — cron watchdog that probes every camera, reboots hung ones over ONVIF, watches
  the Frigate/go2rtc pipeline and pushes alerts via ntfy.
- **Web dashboard** — Flask API + single-page UI with a session login; installable as a PWA.
- **YAML configuration** — all settings in `config.yml` with `${VAR}` env substitution.
- **Retention management** — automatic cleanup by age, storage budget, or per-label limits.
- **OpenVINO inference** — the exported model runs on the Intel iGPU (`intel:gpu`) and falls back
  to CPU; a plain `.pt` runs on CUDA when present, and a TensorRT `.engine` next to it is used
  automatically.

## Quick Start

```bash
git clone https://github.com/AntonyI1/VisionBox.git
cd VisionBox

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/setup_models.py     # fetches yolov8n.pt (+ an optional license-plate model used by the demos)
PYTHONPATH=src python -c "from visionbox import export_openvino; export_openvino()"
mv yolov8n_openvino_model models/  # the detector looks next to the .pt first, then in models/

cp .env.sample .env && chmod 600 .env
# set STORAGE_DIR, the dashboard login, and any camera credentials referenced from config.yml

# Point config.yml at your cameras (see Configuration), then:
python scripts/surveillance.py             # dashboard at http://localhost:8085
python scripts/surveillance.py --ui-only   # dashboard only: browse past events without cameras
```

`surveillance.py` opens a browser on start; pass `--no-browser` on a headless box (the systemd unit
does). `--config` selects another config file.

## Configuration

`config.yml` defines all cameras and shared defaults; `${VAR}` placeholders resolve from the
environment / `.env`. In this deployment the cameras are pulled from Frigate's go2rtc restream
(`rtsp://127.0.0.1:8554/<name>`), so each camera is fetched once and fanned out and no camera
credentials live in the config, but any RTSP URL works.

```yaml
cameras:
  front_door:
    url:        rtsp://127.0.0.1:8554/front_door       # recorded stream
    detect_url: rtsp://127.0.0.1:8554/front_door_sub   # optional lower-res stream for motion + detection
  backyard:
    url: rtsp://${CAM_USER}:${CAM_PASS}@192.168.1.101:554/stream1
    enabled: false                                     # keep the entry, skip the camera
    # test_input: samples/backyard.mp4                 # replace the stream with a file for offline testing

storage:
  recordings: ${STORAGE_DIR}/recordings   # per-camera subdir created automatically
  crops:      ${STORAGE_DIR}/captures/crops
  dataset:    ${STORAGE_DIR}/captures/dataset
  review:     ${STORAGE_DIR}/datasets/review
  training:   ${STORAGE_DIR}/datasets/training
  zones_dir:  ${STORAGE_DIR}/zones        # per-camera <name>.json files

detection:
  mode: outdoor          # outdoor | indoor | vehicles | all (class preset)
  model: yolov8n.pt
  imgsz: 640             # must match the exported OpenVINO model
  device: gpu            # gpu (Intel iGPU via OpenVINO, CPU fallback) | cpu | auto
  detect_fps: 5
  confidence: 0.25
  class_conf: { 0: 0.45, 2: 0.40 }           # per-class floors (COCO ids)
  confirm_threshold: 0.45                    # median track score needed to confirm
  confirm_min_count: 3
  confirm_threshold_by_class: { 0: 0.50 }

recording:
  output_dir: ${STORAGE_DIR}/recordings
  clean:     { enabled: true, max_duration: 130.0, min_valid_bytes: 51200 }
  annotated: { enabled: true, fps: 15.0, cooldown: 10.0, max_duration: 120.0 }
  retention: { days: 30, max_storage_gb: 0, max_per_label: 0, priority_labels: [person] }

display:
  web_port: 8085
  # bind_host: 100.x.y.z   # bind the dashboard to the tailscale0 address only
```

The shipped `config.yml` also tunes motion (`min_area`, the illumination guard, motion-overlap
trigger) and the tracker (ages, IoU, stationary-object filter); every key maps to a field in
`src/visionbox/config.py`.

Per-camera assets land at:
- `recordings/<camera>/{clean,annotated,thumbnails,snapshots}/`
- `captures/crops/<camera>/<class>/`
- `captures/dataset/<camera>/{images,labels}/`
- `zones/<camera>.json`

The SQLite event DB is shared at `recordings/visionbox.db`. Each `events` row carries the camera,
start/end time and duration, the clean/annotated clip, thumbnail and snapshot paths, the detection
count and top label, and a `quarantined` flag that hides reviewed false positives from the UI.

## Web Dashboard

`http://<host>:8085` — dark theme, five tabs:

- **Live** — tile grid of every enabled camera; click a tile to focus a full-screen view. Streams
  stop when the tab is hidden so idle viewers cost nothing.
- **Events** — browse recordings, filter by camera, play clean or annotated clips, delete events.
- **Zones** — pick a camera, draw include/exclude polygons on its live snapshot.
- **Review** — approve/reject captured crops by class, across all cameras or one, with keyboard
  shortcuts (`A` / `R`, arrows to move).
- **Training** — browse the approved training pool (global across cameras).

Set `VISIONBOX_AUTH_USER`, `VISIONBOX_AUTH_PASS` and `VISIONBOX_SECRET_KEY` in `.env` to enable
the login page (30-day session cookie; HTTP Basic is still accepted for scripts and the API).
Leave all of them unset to disable auth; setting only some of them refuses to start.

### Install as an app (PWA)

The dashboard is an installable PWA (web manifest + service worker + offline fallback).
Installing requires HTTPS; on a tailnet the easiest way is `tailscale serve`, which
terminates TLS with a valid certificate for the machine's tailnet name:

```bash
sudo tailscale serve --bg 8085
# dashboard now at https://<machine>.<tailnet>.ts.net
```

Then on the phone (any Chromium browser):

1. Open `https://<machine>.<tailnet>.ts.net` and sign in.
2. Menu → **Add to Home screen** → **Install**.

Any reverse proxy that terminates TLS (Caddy, nginx) works too. Over plain
`http://<tailscale-ip>:8085` the dashboard still works and "Add to Home screen"
creates a regular shortcut, just without standalone install or the offline page,
since service workers require a secure context.

## How It Works

```
Per camera (worker thread):
  RTSP → CameraStream (always-latest frame, liveness probe + backoff on stalls)
    → MOG2 motion (≈1 ms CPU, on a downscaled frame)
    → motion-gated YOLO via the shared detector (lock-serialized)
    → zone filtering (exclude masks, required-zone gate)
    → SORT tracker (Kalman predict between detections)
    → confirmation gate (median score, min detections) + moving-and-overlaps-motion trigger
    → RecordingManager → clean RTSP copy + annotated mp4 + thumbnail + snapshot + SQLite row
    → crop capture for the review queue
    → view buffer for the dashboard MJPEG endpoint (encoded once, fanned out to viewers)
```

Shared across cameras:
- One `MultiModelDetector` behind a lock; `SIGHUP` (or `POST /api/model/reload`) rebuilds and
  warms a new model off-lock and swaps it in, so a promoted model goes live without a restart.
- One `RecordingDatabase` at `<recordings>/visionbox.db`.
- One Flask API server; a retention thread; an RSS watchdog that exits for a clean systemd restart
  if memory stays above `VISIONBOX_MAX_RSS_MB` (default 4096).

## Hosting

### systemd

`visionbox.service` runs `scripts/surveillance.py` from the checkout with `.env` as its
`EnvironmentFile`. It hardcodes `User=`, `Group=`, `WorkingDirectory=` and the venv path, so adjust
those (and `RequiresMountsFor=` if your storage is not under `/mnt/storage`) before installing:

```bash
./install-service.sh        # install + enable + start (refuses if the unit points elsewhere)
./restart-service.sh        # restart and tail the journal, decoder noise filtered
journalctl -u visionbox.service -f
```

### Camera doctor (cron)

`scripts/camera_doctor.py` runs every five minutes from cron and keeps its state in
`logs/camera_doctor_state.json`:

```
*/5 * * * * <repo>/venv/bin/python3 <repo>/scripts/camera_doctor.py >> <repo>/logs/camera_doctor.log 2>&1
```

- Probes each camera's own RTSP endpoint and its go2rtc restream. Three consecutive misses mark
  the camera DOWN: it is logged, pushed via ntfy, and sent an ONVIF `SystemReboot` at most hourly.
- Watches Frigate's internal liveness signals (dead threads or VAAPI errors in its log, recording
  segments piling up, a frozen `latest.jpg`, a restream that reports 0 fps) and restarts the
  container when two consecutive runs agree, with a warm-up grace period, a cooldown, a restart
  budget, and no restart while the NFS share is stale.
- Attributes every per-camera symptom to the camera link (ping loss/RTT, go2rtc producer state)
  or to the pipeline before acting, so a flaky WiFi camera never triggers a Frigate restart.

Alerts need `NTFY_TOPIC` (and optionally `NTFY_URL`) in `.env`; `FRIGATE_*` and `DOCTOR_*` tunables
are listed in the module docstring and in `.env.sample`. `--dry-run` prints the decisions without
rebooting, restarting or notifying; `--verbose` prints every probe. The camera table (IPs, ONVIF
ports, stream paths) is currently hardcoded at the top of the script.

### Network hardening

Both scripts are reviewed-then-run with sudo; they print the verification commands for this host.

- `harden-network.sh` — ufw default-deny with loopback, SSH and the whole `tailscale0` interface
  allowed, so the dashboard is reachable only over the tailnet; disables rpcbind (NFSv4 does not
  need it).
- `harden-frigate.sh` — Frigate's Docker-published ports (5000/8554/8555) bypass ufw's INPUT chain,
  so this installs DOCKER-USER rules via `/etc/ufw/after.rules` that drop LAN traffic to them while
  keeping tailnet, inter-container and host access.

## Self-training

`scripts/training/` turns reviewed footage into a better model overnight, on the CPU, without
touching live surveillance (`Nice=15`, idle IO):

| Script | Role |
| --- | --- |
| `prep_replay.py` | Freeze a small COCO replay set (`--limit`, `--force`) for the regression gate; needs internet once. |
| `assemble_dataset.py` | Gather captured, reviewed and corrected labels into a YOLO dataset with a frozen val split (`--dry-run`). |
| `train_overnight.py` | Fine-tune yolov8n with the backbone frozen, compare base-class mAP against the active model on the replay set, export to OpenVINO, then stage the candidate (or promote with `--auto-promote`). A missing replay set fails closed. |
| `promote_model.py` | Copy an export into a durable slot under `models/`, repoint the active symlink and `SIGHUP` the service; `--rollback` restores the previous model. |
| `correct_labels.py` | Assisted labelling of the review queue (OpenCV window; `--headless --min-conf` auto-accepts confident frames). |
| `run_overnight.sh` | Entrypoint used by the timer: refresh the replay set, then train. |

```bash
sudo bash ./install-training-timer.sh    # visionbox-train.timer, 03:30 daily, stage-only
sudo systemctl start visionbox-train.service && journalctl -u visionbox-train.service -f
venv/bin/python3 scripts/training/promote_model.py            # promote the staged candidate
venv/bin/python3 scripts/training/promote_model.py --rollback
```

Each run writes `runs/train/<run>/REPORT.json`. Uncomment `Environment=AUTO_PROMOTE=1` in the
service unit to hot-swap automatically whenever the gate passes. The active model is the
`models/yolov8n_openvino_model` symlink; `models/.prev_active` records the rollback target. The
training scripts locate the repository from their own path and read `STORAGE_DIR` from `.env`.

## Tools

- `scripts/dedupe_crops.py [--crops DIR] [--threshold 6] [--apply | --restore]` — cluster
  near-identical review crops by perceptual hash and move all but the best of each cluster to
  `_trash/`, with a manifest; dry-run by default, fully reversible.
- `scripts/quarantine_false_positives.py [--db PATH] [--apply | --restore [--reason BUCKET]]` —
  move the files of recognisable false-positive events (parked-car re-detections, night IR
  phantoms, orphaned rows) into `_quarantine/` and flag the rows; dry-run by default, backs up the
  DB, fully reversible.
- Single-camera demos (need a display): `scripts/camera_demo.py [url] --mode outdoor` (detection +
  tracking), `scripts/motion_demo.py [url]` (press `m` to toggle motion gating),
  `scripts/detect_and_capture.py [source] --no-display` (save crops + YOLO labels),
  `scripts/capture_for_review.py` (high-confidence frames for review). They fall back to
  `CAMERA_URL` from `.env` when no URL is given.

## Development

```bash
pip install -r requirements-dev.txt
pytest            # tests/ (pythonpath = src)
ruff check .      # lint config in pyproject.toml
```

## Architecture

```
src/visionbox/
├── api.py               # Flask REST + dashboard, login, PWA routes, CamerasState + CameraView
├── config.py            # YAML config dataclasses with env var resolution, per-camera path helpers
├── database.py          # SQLite events table (per-camera, migrations, retention queries)
├── detector_v2.py       # YOLOv8 multi-model detector (OpenVINO iGPU/CPU, CUDA, TensorRT), hot reload
├── kalman.py            # Kalman box filter
├── tracker.py           # SORT tracker with coasting + stationary-object judgement
├── motion.py            # MOG2 motion regions, illumination guard
├── zones.py             # Per-camera include/exclude polygon filtering
├── viz.py               # Drawing helpers shared by the pipeline and the demos
├── recorder.py          # OpenCV-driven event recorder (annotated mp4)
├── clean_recorder.py    # FFmpeg subprocess for the zero-CPU RTSP copy
├── recording_manager.py # Orchestrates dual recording, snapshots and retention per camera
└── web/                 # Dashboard SPA (index.html, app.js, style.css, manifest, sw.js, icons)

scripts/
├── surveillance.py      # Main: one CameraPipeline thread per enabled camera + API server
├── setup_models.py      # Downloads yolov8n.pt (+ optional license-plate model)
├── camera_doctor.py     # Cron watchdog: camera probes, ONVIF reboot, Frigate pipeline, ntfy
├── dedupe_crops.py      # Reversible crop de-duplication
├── quarantine_false_positives.py
├── camera_demo.py / motion_demo.py / detect_and_capture.py / capture_for_review.py
└── training/            # assemble_dataset, prep_replay, train_overnight, promote_model,
                         # correct_labels, run_overnight.sh

tests/                   # pytest suite
config.yml               # The thing you edit
.env / .env.sample       # Secrets, STORAGE_DIR, optional tunables
visionbox.service        # systemd unit for the pipeline
visionbox-train.service / .timer   # nightly self-training
install-service.sh / restart-service.sh / install-training-timer.sh
harden-network.sh / harden-frigate.sh
models/                  # OpenVINO exports + active-model symlink (gitignored)
```

## Tech Stack

- **Python 3.11+** (3.12 in production)
- **YOLOv8** (Ultralytics) — object detection, fine-tuning, export
- **OpenVINO** — inference on the Intel iGPU or CPU
- **OpenCV** — video decoding, background subtraction, annotated recording
- **FFmpeg** — clean RTSP stream copy
- **Flask** — REST API and dashboard
- **SQLite** — event database
- **SciPy** — Hungarian algorithm for tracker assignment
- **systemd / cron / ufw / Tailscale** — hosting, watchdog, access
- **ntfy** — push alerts

## Roadmap

- [ ] INT8 quantization
- [ ] Tiled inference for better distance detection
- [ ] One-click retraining from approved crops
- [ ] Coral TPU offload
- [ ] Per-camera detection presets (currently shared)
