"""Flask API server for VisionBox multi-camera dashboard."""

import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, abort, session, redirect

from .database import RecordingDatabase
from .recording_manager import RecordingManager
from .zones import ZoneFilter, Zone

logger = logging.getLogger(__name__)


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionBox · Sign in</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:radial-gradient(1100px 520px at 50% -8%,#241d15,#15110c); color:#ece3d6; }
  .card { width:min(92vw,360px); background:#221d17; border:1px solid #3a3229; border-radius:14px;
          padding:32px 30px; box-shadow:0 16px 50px rgba(0,0,0,.55); }
  .brand { margin-bottom:24px; }
  .brand h1 { font-size:19px; margin:0; font-weight:600; color:#ece3d6; }
  .brand small { color:#9d8f7c; font-size:12px; }
  label { display:block; font-size:12px; color:#a2937f; margin:15px 0 6px; font-weight:500; }
  input { width:100%; padding:11px 12px; background:#1a1510; border:1px solid #3a3229; border-radius:9px;
          color:#ece3d6; font-size:14px; outline:none; transition:border-color .15s,background .15s; }
  input:focus { border-color:#cda06d; background:#1f1a13; }
  button { width:100%; margin-top:24px; padding:12px; background:#cda06d; color:#221a12; border:0;
           border-radius:9px; font-size:14px; font-weight:600; cursor:pointer; transition:background .15s; }
  button:hover { background:#dbb184; }
  .err { margin-top:15px; min-height:16px; color:#e0715c; font-size:13px; text-align:center; }
  .foot { margin-top:18px; text-align:center; color:#7c6f5f; font-size:11px; }
</style>
</head>
<body>
  <form class="card" method="post" action="">
    <div class="brand"><h1>VisionBox</h1><small>Surveillance dashboard</small></div>
    <label for="u">Username</label>
    <input id="u" name="username" value="__USER__" autocomplete="username">
    <label for="p">Password</label>
    <input id="p" name="password" type="password" placeholder="Enter password" autofocus autocomplete="current-password">
    <button type="submit">Sign in</button>
    <div class="err">__ERROR__</div>
    <div class="foot">Tailscale-only · household access</div>
  </form>
</body>
</html>"""


@dataclass
class CameraView:
    """Per-camera shared state (frame buffer + stats) exposed to the web."""
    name: str
    frame: np.ndarray | None = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    fps: float = 0.0
    frame_count: int = 0
    event_count: int = 0
    recording: bool = False
    state: str = 'idle'
    connected: bool = False
    last_error: str = ''
    last_update: float = 0.0


@dataclass
class CamerasState:
    """Top-level shared state passed to the API server."""
    config: object = None
    db: RecordingDatabase | None = None
    output_dir: Path = field(default_factory=lambda: Path('.'))
    crops_dir: Path = field(default_factory=lambda: Path('captures/crops'))
    training_dir: Path = field(default_factory=lambda: Path('datasets/training'))
    zones_dir: Path = field(default_factory=lambda: Path('zones'))
    start_time: float = field(default_factory=time.time)
    offline: bool = False
    views: dict[str, CameraView] = field(default_factory=dict)
    zone_filters: dict[str, ZoneFilter] = field(default_factory=dict)
    recording_mgrs: dict[str, RecordingManager] = field(default_factory=dict)
    detector: object = None          # MultiModelDetector (None in UI-only mode)
    detector_lock: object = None     # held during inference + hot-swap reload

    def add_camera(self, view: CameraView, zone_filter: ZoneFilter, rec_mgr: RecordingManager):
        self.views[view.name] = view
        self.zone_filters[view.name] = zone_filter
        self.recording_mgrs[view.name] = rec_mgr


def create_app(state: CamerasState) -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), 'web'),
        static_url_path='/static',
    )

    # Auth: a session-cookie login (proper login page for browsers) that also accepts
    # HTTP Basic for API/scripts. Gated on VISIONBOX_AUTH_USER/PASS in .env (600,
    # gitignored); fails CLOSED if only one is set. On-box model reload is via SIGHUP,
    # not HTTP, so there is no localhost exemption to bypass.
    _auth_user = os.environ.get('VISIONBOX_AUTH_USER', '')
    _auth_pass = os.environ.get('VISIONBOX_AUTH_PASS', '')
    if bool(_auth_user) != bool(_auth_pass):
        raise SystemExit('VISIONBOX_AUTH_USER and VISIONBOX_AUTH_PASS must both be set '
                         '(or both unset to disable auth).')

    if _auth_user and _auth_pass:
        app.secret_key = os.environ.get('VISIONBOX_SECRET_KEY') or os.urandom(32)
        app.permanent_session_lifetime = timedelta(days=30)
        app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
        _user_b = _auth_user.encode('utf-8')
        _pass_b = _auth_pass.encode('utf-8')

        def _creds_ok(user, pw):
            try:
                return bool(hmac.compare_digest((user or '').encode('utf-8'), _user_b)
                            & hmac.compare_digest((pw or '').encode('utf-8'), _pass_b))
            except (TypeError, UnicodeError):
                return False

        @app.before_request
        def _require_auth():
            if request.path == '/login' or request.endpoint == 'static':
                return None
            if session.get('user'):
                return None
            auth = request.authorization  # API/scripts may still use HTTP Basic
            if auth and auth.type == 'basic' and _creds_ok(auth.username, auth.password):
                return None
            # Browsers get the login page; programmatic clients get a plain 401
            # (no WWW-Authenticate header -> no native browser popup).
            if request.method == 'GET' and 'text/html' in request.headers.get('Accept', ''):
                return redirect('/login?next=' + quote(request.full_path, safe=''))
            return Response('Authentication required', 401)

        @app.route('/login', methods=['GET', 'POST'])
        def login():
            error = ''
            if request.method == 'POST':
                if _creds_ok(request.form.get('username'), request.form.get('password')):
                    session.permanent = True
                    session['user'] = _auth_user
                    nxt = request.args.get('next', '/')
                    return redirect(nxt if nxt.startswith('/') else '/')
                error = 'Incorrect username or password.'
            page = _LOGIN_PAGE.replace('__USER__', _auth_user).replace('__ERROR__', error)
            return Response(page, mimetype='text/html', status=(401 if error else 200))

        @app.route('/logout')
        def logout():
            session.clear()
            return redirect('/login')

        logger.info('dashboard auth: ENABLED (login page + Basic)')
    else:
        logger.warning('dashboard auth: DISABLED — set VISIONBOX_AUTH_USER/PASS in .env')

    def _get_view(name: str) -> CameraView:
        if name not in state.views:
            abort(404)
        return state.views[name]

    def _get_zone_filter(name: str) -> ZoneFilter:
        if name not in state.zone_filters:
            # Allow offline mode to read zones from disk
            zpath = state.zones_dir / f'{name}.json'
            if not zpath.exists():
                abort(404)
            zf = ZoneFilter(str(zpath))
            state.zone_filters[name] = zf
            return zf
        return state.zone_filters[name]

    @app.route('/')
    def index():
        return send_file(
            os.path.join(app.static_folder, 'index.html'),
            mimetype='text/html',
        )

    # ----- Cameras -----

    @app.route('/api/cameras')
    def cameras():
        cam_cfgs = getattr(state.config, 'cameras', {}) if state.config else {}
        include_disabled = request.args.get('include_disabled') == 'true'
        out = []
        for name, cam in cam_cfgs.items():
            enabled = getattr(cam, 'enabled', True)
            if not enabled and not include_disabled:
                continue
            view = state.views.get(name)
            mgr = state.recording_mgrs.get(name)
            out.append({
                'name': name,
                'enabled': enabled,
                'fps': view.fps if view else 0,
                'recording': view.recording if view else False,
                'state': view.state if view else 'offline',
                'connected': view.connected if view else False,
                'last_error': view.last_error if view else '',
                'frame_count': view.frame_count if view else 0,
                'event_count': state.db.get_event_count(camera=name) if state.db else 0,
            })
        return jsonify(out)

    @app.route('/api/cameras/<name>/stream')
    def camera_stream(name):
        view = _get_view(name)

        def generate():
            while True:
                with view.frame_lock:
                    frame = view.frame
                if frame is not None:
                    _, jpeg = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg.tobytes() + b'\r\n'
                    )
                time.sleep(0.033)

        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )

    @app.route('/api/cameras/<name>/snapshot')
    def camera_snapshot(name):
        view = _get_view(name)
        with view.frame_lock:
            frame = view.frame
        if frame is None:
            abort(503)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

    # ----- Status -----

    @app.route('/api/status')
    def status():
        cams = list(state.views.values())
        any_recording = any(v.recording for v in cams)
        storage = _get_storage_info(state)
        return jsonify({
            'offline': state.offline,
            'cameras': len(cams),
            'connected_cameras': sum(1 for v in cams if v.connected),
            'recording_cameras': sum(1 for v in cams if v.recording),
            'any_recording': any_recording,
            'event_count': state.db.get_event_count() if state.db else 0,
            'uptime': round(time.time() - state.start_time),
            'storage': storage,
        })

    # ----- Events -----

    @app.route('/api/events')
    def events():
        limit = min(request.args.get('limit', 50, type=int), 200)
        offset = request.args.get('offset', 0, type=int)
        camera = request.args.get('camera') or None
        rows = state.db.get_events(limit=limit, offset=offset, camera=camera)
        total = state.db.get_event_count(camera=camera)
        return jsonify({
            'events': rows, 'total': total, 'limit': limit, 'offset': offset,
        })

    @app.route('/api/events/<event_id>')
    def event_detail(event_id):
        event = state.db.get_event(event_id)
        if not event:
            abort(404)
        return jsonify(event)

    @app.route('/api/events/<event_id>', methods=['DELETE'])
    def delete_event(event_id):
        event = state.db.get_event(event_id)
        if not event:
            abort(404)
        cam_out = _event_output_dir(state, event)
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            rel = event.get(key, '')
            if rel:
                p = Path(rel) if Path(rel).is_absolute() else cam_out / rel
                if p.exists():
                    p.unlink()
                if key not in ('thumbnail', 'snapshot'):
                    meta = p.with_suffix('.json')
                    if meta.exists():
                        meta.unlink()
        state.db.delete_event(event_id)
        return jsonify({'deleted': event_id})

    @app.route('/api/events/<event_id>/thumbnail')
    def event_thumbnail(event_id):
        event = state.db.get_event(event_id)
        if not event or not event.get('thumbnail'):
            abort(404)
        thumb_path = Path(event['thumbnail'])
        if not thumb_path.is_absolute():
            thumb_path = _event_output_dir(state, event) / thumb_path
        if not thumb_path.exists():
            abort(404)
        return send_file(str(thumb_path), mimetype='image/jpeg')

    @app.route('/api/events/<event_id>/snapshot')
    def event_snapshot(event_id):
        event = state.db.get_event(event_id)
        if not event or not event.get('snapshot'):
            abort(404)
        snap_path = Path(event['snapshot'])
        if not snap_path.is_absolute():
            snap_path = _event_output_dir(state, event) / snap_path
        if not snap_path.exists():
            abort(404)
        return send_file(str(snap_path), mimetype='image/jpeg')

    # ----- Model / self-training -----
    @app.route('/api/model/reload', methods=['POST'])
    def model_reload():
        """Hot-swap the on-disk model into the running detector (after promotion)."""
        if state.detector is None or state.detector_lock is None:
            return jsonify({'ok': False, 'error': 'no detector in this process'}), 409
        try:
            info = state.detector.reload(swap_lock=state.detector_lock)
            return jsonify({'ok': True, **info})
        except Exception as exc:
            logger.exception('model reload failed')
            return jsonify({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), 500

    @app.route('/api/model/status')
    def model_status():
        root = Path(__file__).resolve().parents[2]
        active = root / 'models' / 'yolov8n_openvino_model'
        target = None
        try:
            if active.is_symlink():
                target = os.readlink(str(active))
        except OSError:
            target = None
        report, latest = None, None
        runs = sorted((root / 'runs' / 'train').glob('overnight_*/REPORT.json'))
        if runs:
            latest = runs[-1].parent.name
            try:
                with open(runs[-1]) as f:
                    report = json.load(f)
            except (OSError, ValueError):
                report = None
        return jsonify({
            'loaded': state.detector is not None,
            'active_model': str(active),
            'active_target': target,
            'classes': len(state.detector.class_names) if state.detector else None,
            'device': state.detector.effective_device if state.detector else None,
            'latest_run': latest,
            'latest_report': report,
        })

    @app.route('/api/events/<event_id>/clip/<clip_type>')
    def event_clip(event_id, clip_type):
        if clip_type not in ('clean', 'annotated'):
            abort(400)
        event = state.db.get_event(event_id)
        if not event:
            abort(404)
        rel = event.get(f'{clip_type}_clip', '')
        if not rel:
            abort(404)
        clip_path = Path(rel)
        if not clip_path.is_absolute():
            clip_path = _event_output_dir(state, event) / clip_path
        h264_path = clip_path.with_name(clip_path.stem + '.h264.mp4')
        serve_path = h264_path if h264_path.exists() else clip_path
        if not serve_path.exists():
            abort(404)
        return _send_video(str(serve_path))

    # ----- Config -----

    @app.route('/api/config')
    def config():
        if not state.config:
            return jsonify({})
        from dataclasses import asdict
        return jsonify(asdict(state.config))

    # ----- Zones (per camera) -----

    @app.route('/api/cameras/<name>/zones')
    def get_zones(name):
        zf = _get_zone_filter(name)
        return jsonify(zf.get_zones())

    @app.route('/api/cameras/<name>/zones', methods=['POST'])
    def add_zone(name):
        zf = _get_zone_filter(name)
        data = request.get_json() or {}
        zname = data.get('name', '').strip()
        ztype = data.get('type', '')
        points = data.get('points', [])
        if not zname:
            return jsonify({'error': 'Name required'}), 400
        if ztype not in ('include', 'exclude'):
            return jsonify({'error': 'Type must be include or exclude'}), 400
        if not isinstance(points, list) or len(points) < 3:
            return jsonify({'error': 'At least 3 points required'}), 400
        for p in points:
            if not isinstance(p, list) or len(p) != 2:
                return jsonify({'error': 'Points must be [x, y] pairs'}), 400
            if not all(0 <= v <= 1 for v in p):
                return jsonify({'error': 'Coordinates must be 0-1'}), 400
        zf.add_zone(Zone(name=zname, type=ztype, points=points))
        return jsonify({'ok': True})

    @app.route('/api/cameras/<name>/zones/<zname>', methods=['DELETE'])
    def delete_zone(name, zname):
        zf = _get_zone_filter(name)
        if zf.remove_zone(zname):
            return jsonify({'deleted': zname})
        abort(404)

    # ----- Review (per camera, per class) -----

    @app.route('/api/cameras/<name>/review/classes')
    def review_classes(name):
        cam_crops = state.crops_dir / name
        if not cam_crops.is_dir():
            return jsonify([])
        out = []
        for d in sorted(cam_crops.iterdir()):
            if not d.is_dir():
                continue
            count = sum(
                1 for f in d.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            )
            if count > 0:
                out.append({'name': d.name, 'count': count})
        return jsonify(out)

    @app.route('/api/cameras/<name>/review/<class_name>')
    def review_crop(name, class_name):
        offset = request.args.get('offset', 0, type=int)
        files = _list_images(state.crops_dir / name, class_name)
        if not files:
            return jsonify({'total': 0, 'offset': offset, 'crop': None})
        offset = max(0, min(offset, len(files) - 1))
        filename = files[offset]
        meta = _parse_crop_filename(filename) or {}
        meta['filename'] = filename
        meta['class'] = class_name
        meta['camera'] = name
        return jsonify({'total': len(files), 'offset': offset, 'crop': meta})

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/image')
    def review_image(name, class_name, filename):
        path = _safe_path(state.crops_dir / name, class_name, filename)
        if not path.is_file():
            abort(404)
        return send_file(str(path), mimetype='image/jpeg')

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/approve', methods=['POST'])
    def review_approve(name, class_name, filename):
        src = _safe_path(state.crops_dir / name, class_name, filename)
        if not src.is_file():
            abort(404)
        dest_dir = _safe_path(state.training_dir, class_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f'{name}_{filename}'  # prefix with camera to avoid name collisions
        shutil.move(str(src), str(dest))
        logger.info('Approved %s/%s/%s → training', name, class_name, filename)
        return jsonify({'action': 'approved', 'file': filename})

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/reject', methods=['POST'])
    def review_reject(name, class_name, filename):
        src = _safe_path(state.crops_dir / name, class_name, filename)
        if not src.is_file():
            abort(404)
        src.unlink()
        logger.info('Rejected %s/%s/%s', name, class_name, filename)
        return jsonify({'action': 'rejected', 'file': filename})

    # ----- Training (global pool) -----

    @app.route('/api/training/classes')
    def training_classes():
        tdir = state.training_dir
        if not tdir.is_dir():
            return jsonify([])
        out = []
        for d in sorted(tdir.iterdir()):
            if not d.is_dir():
                continue
            count = sum(
                1 for f in d.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            )
            if count > 0:
                out.append({'name': d.name, 'count': count})
        return jsonify(out)

    @app.route('/api/training/<class_name>')
    def training_image(class_name):
        offset = request.args.get('offset', 0, type=int)
        files = _list_images(state.training_dir, class_name)
        if not files:
            return jsonify({'total': 0, 'offset': offset, 'image': None})
        offset = max(0, min(offset, len(files) - 1))
        filename = files[offset]
        meta = _parse_crop_filename(filename) or {}
        meta['filename'] = filename
        meta['class'] = class_name
        return jsonify({'total': len(files), 'offset': offset, 'image': meta})

    @app.route('/api/training/<class_name>/<filename>/image')
    def training_serve_image(class_name, filename):
        path = _safe_path(state.training_dir, class_name, filename)
        if not path.is_file():
            abort(404)
        return send_file(str(path), mimetype='image/jpeg')

    @app.route('/api/training/<class_name>/<filename>', methods=['DELETE'])
    def training_delete(class_name, filename):
        path = _safe_path(state.training_dir, class_name, filename)
        if not path.is_file():
            abort(404)
        path.unlink()
        return jsonify({'deleted': filename})

    return app


# ---------- Helpers ----------

def _event_output_dir(state: CamerasState, event: dict) -> Path:
    """Return the on-disk directory that owns an event's clips."""
    camera = event.get('camera') or ''
    if camera:
        return (state.output_dir / camera).resolve()
    return state.output_dir.resolve()


def _get_storage_info(state: CamerasState) -> dict:
    out = state.output_dir
    recordings_bytes = sum(
        f.stat().st_size for f in out.rglob('*') if f.is_file()
    ) if out.exists() else 0
    cfg = state.config
    rec_cfg = getattr(cfg, 'recording', None) if cfg else None
    max_gb = rec_cfg.retention.max_storage_gb if rec_cfg else 0
    if max_gb > 0:
        budget_bytes = int(max_gb * 1024 * 1024 * 1024)
        budget_free = max(0, budget_bytes - recordings_bytes)
    else:
        try:
            stat = os.statvfs(str(out))
            budget_bytes = stat.f_blocks * stat.f_frsize
            budget_free = stat.f_bavail * stat.f_frsize
        except OSError:
            budget_bytes = budget_free = 0
    return {
        'recordings_bytes': recordings_bytes,
        'recordings_human': _human_size(recordings_bytes),
        'disk_total_bytes': budget_bytes,
        'disk_total_human': _human_size(budget_bytes),
        'disk_free_bytes': budget_free,
        'disk_free_human': _human_size(budget_free),
    }


def _safe_path(base: Path, *parts: str) -> Path:
    resolved = (base / Path(*parts)).resolve()
    if not resolved.is_relative_to(base.resolve()):
        abort(400)
    return resolved


def _parse_crop_filename(filename):
    """Parse track{id}_{YYYYMMDD}_{HHMMSS}[..._]{conf}.jpg with optional camera prefix."""
    m = re.search(
        r'track(\d+)_(\d{8})_(\d{6})_\d*_?(\d+\.\d+)\.jpg$', filename
    )
    if not m:
        return None
    try:
        ts = datetime.strptime(f'{m.group(2)}_{m.group(3)}', '%Y%m%d_%H%M%S')
    except ValueError:
        ts = None
    return {
        'track_id': int(m.group(1)),
        'timestamp': ts.isoformat() if ts else None,
        'confidence': round(float(m.group(4)), 2),
    }


def _list_images(base_dir: Path, class_name: str):
    class_dir = _safe_path(base_dir, class_name)
    if not class_dir.is_dir():
        return []
    return sorted(
        f.name for f in class_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )


def _human_size(nbytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024:
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024
    return f'{nbytes:.1f} PB'


def _send_video(path: str) -> Response:
    """Serve MP4 with Range header support for HTML5 video seeking."""
    file_size = os.path.getsize(path)
    range_header = request.headers.get('Range')
    if range_header:
        match = range_header.replace('bytes=', '').split('-')
        byte_start = int(match[0])
        byte_end = int(match[1]) if match[1] else file_size - 1
        content_length = byte_end - byte_start + 1

        def generate():
            with open(path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return Response(
            generate(), status=206, mimetype='video/mp4',
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': content_length,
            },
        )
    return send_file(path, mimetype='video/mp4')


def start_api_server(state: CamerasState, port: int, host: str = '0.0.0.0') -> threading.Thread:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app = create_app(state)

    def _serve():
        try:
            app.run(host=host, port=port, threaded=True, use_reloader=False)
        except OSError as exc:
            # A bad bind_host / taken port must fail LOUDLY (systemd restarts + logs),
            # not die silently in this daemon thread while main() waits forever.
            logger.error('API server could not bind %s:%s — %s', host, port, exc)
            os._exit(1)

    thread = threading.Thread(target=_serve, daemon=True, name='api-server')
    thread.start()
    return thread


# ---------- Backwards-compat shims (kept light to avoid stale references) ----------

# Old `PipelineState` import paths may still exist in saved scripts.
PipelineState = CamerasState
