"""Flask API server for VisionBox multi-camera dashboard."""

import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, request, send_file, session

from .database import RecordingDatabase
from .recording_manager import RecordingManager
from .zones import Zone, ZoneFilter

logger = logging.getLogger(__name__)

# A /stream client re-sends its last frame after this long without a new one. Werkzeug
# only detects a vanished client on the next write, so without this an unbounded wait on
# a stalled/offline camera would leak the worker thread + its stream_clients slot.
_STREAM_KEEPALIVE_S = 2.0

_IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#221d17">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/favicon.ico" sizes="48x48">
<link rel="icon" type="image/png" sizes="32x32" href="/static/icons/favicon-32.png">
<link rel="apple-touch-icon" href="/static/icons/apple-touch-icon.png">
<title>VisionBox · Sign in</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; min-height:100dvh; display:flex; align-items:center; justify-content:center;
         font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:radial-gradient(1100px 520px at 50% -8%,#241d15,#15110c); color:#ece3d6;
         padding: max(16px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right))
                  max(16px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left)); }
  .card { width:min(92vw,360px); background:#221d17; border:1px solid #3a3229; border-radius:14px;
          padding:32px 30px; box-shadow:0 16px 50px rgba(0,0,0,.55); }
  .brand { margin-bottom:24px; }
  .brand h1 { font-size:19px; margin:0; font-weight:600; color:#ece3d6; }
  .brand small { color:#9d8f7c; font-size:12px; }
  label { display:block; font-size:12px; color:#a2937f; margin:15px 0 6px; font-weight:500; }
  input { width:100%; min-height:44px; padding:11px 12px; background:#1a1510; border:1px solid #3a3229;
          border-radius:9px; color:#ece3d6; font-size:14px; outline:none;
          transition:border-color .15s,background .15s; }
  input:focus { border-color:#cda06d; background:#1f1a13; }
  button { width:100%; min-height:44px; margin-top:24px; padding:12px; background:#cda06d; color:#221a12; border:0;
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
    <input id="p" name="password" type="password" placeholder="Enter password" autofocus
           autocomplete="current-password">
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
    frame_version: int = 0
    frame_jpeg: bytes | None = None
    jpeg_version: int = 0
    stream_clients: int = 0
    frame_cond: threading.Condition = field(init=False)

    def __post_init__(self):
        # One mutex for the whole feed path: the Condition wraps frame_lock,
        # so existing `with view.frame_lock:` sites still serialise correctly.
        self.frame_cond = threading.Condition(self.frame_lock)


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
            # PWA plumbing stays public: the manifest is fetched without cookies,
            # and the service worker / favicon carry nothing sensitive.
            if request.path in ('/login', '/manifest.webmanifest', '/sw.js', '/favicon.ico') \
                    or request.endpoint == 'static':
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
                    # Same-site paths only: '//host' and '/\host' are protocol-relative.
                    if not nxt.startswith('/') or nxt[1:2] in ('/', '\\'):
                        nxt = '/'
                    return redirect(nxt)
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

    def _web_file(*parts: str, mimetype: str) -> Response:
        return send_file(os.path.join(app.static_folder, *parts), mimetype=mimetype)

    @app.route('/')
    def index():
        return _web_file('index.html', mimetype='text/html')

    # ----- PWA (manifest + service worker at root scope) -----

    @app.route('/manifest.webmanifest')
    def manifest():
        return _web_file('manifest.webmanifest', mimetype='application/manifest+json')

    @app.route('/sw.js')
    def service_worker():
        # no-cache so browsers revalidate on every load and updates roll out promptly
        resp = _web_file('sw.js', mimetype='text/javascript')
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    @app.route('/favicon.ico')
    def favicon():
        return _web_file('icons', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

    # ----- Cameras -----

    @app.route('/api/cameras')
    def cameras():
        cam_cfgs = getattr(state.config, 'cameras', {}) if state.config else {}
        include_disabled = request.args.get('include_disabled') == 'true'
        counts = state.db.get_camera_counts() if state.db else {}
        out = []
        for name, cam in cam_cfgs.items():
            enabled = getattr(cam, 'enabled', True)
            if not enabled and not include_disabled:
                continue
            view = state.views.get(name)
            out.append({
                'name': name,
                'enabled': enabled,
                'fps': view.fps if view else 0,
                'recording': view.recording if view else False,
                'state': view.state if view else 'offline',
                'connected': view.connected if view else False,
                'last_error': view.last_error if view else '',
                'frame_count': view.frame_count if view else 0,
                'event_count': counts.get(name, 0),
            })
        return jsonify(out)

    @app.route('/api/cameras/<name>/stream')
    def camera_stream(name):
        view = _get_view(name)

        def generate():
            cond = view.frame_cond
            with cond:
                view.stream_clients += 1
                cond.notify_all()
            last_sent = 0
            try:
                while True:
                    jpeg, last_sent = _next_jpeg(view, last_sent)
                    if jpeg is None:
                        continue
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n'
                    )
            finally:
                with cond:
                    view.stream_clients -= 1
                    cond.notify_all()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        for key in ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot'):
            path = _event_path(state, event, key)
            if path is None:
                continue
            if path.exists():
                path.unlink()
            if key.endswith('_clip'):
                meta = path.with_suffix('.json')
                if meta.exists():
                    meta.unlink()
        state.db.delete_event(event_id)
        return jsonify({'deleted': event_id})

    def _event_image(event_id: str, key: str) -> Response:
        event = state.db.get_event(event_id)
        path = _event_path(state, event, key) if event else None
        if path is None or not path.exists():
            abort(404)
        return send_file(str(path), mimetype='image/jpeg')

    @app.route('/api/events/<event_id>/thumbnail')
    def event_thumbnail(event_id):
        return _event_image(event_id, 'thumbnail')

    @app.route('/api/events/<event_id>/snapshot')
    def event_snapshot(event_id):
        return _event_image(event_id, 'snapshot')

    @app.route('/api/events/<event_id>/clip/<clip_type>')
    def event_clip(event_id, clip_type):
        if clip_type not in ('clean', 'annotated'):
            abort(400)
        event = state.db.get_event(event_id)
        clip_path = _event_path(state, event, f'{clip_type}_clip') if event else None
        if clip_path is None:
            abort(404)
        h264_path = clip_path.with_name(clip_path.stem + '.h264.mp4')
        serve_path = h264_path if h264_path.exists() else clip_path
        if not serve_path.exists():
            abort(404)
        return _send_video(str(serve_path))

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

    # ----- Config -----

    @app.route('/api/config')
    def config():
        return jsonify(asdict(state.config) if state.config else {})

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
        if not zf.remove_zone(zname):
            abort(404)
        return jsonify({'deleted': zname})

    # ----- Review (per camera, per class) -----

    def _camera_crops(name: str) -> Path:
        return _safe_path(state.crops_dir, name)

    @app.route('/api/cameras/<name>/review/classes')
    def review_classes(name):
        counts = _class_counts(_camera_crops(name))
        return jsonify([{'name': k, 'count': n} for k, n in counts.items()])

    @app.route('/api/cameras/<name>/review/<class_name>')
    def review_crop(name, class_name):
        files = _list_images(_camera_crops(name), class_name)
        return _paged(files, 'crop', lambda fn: _crop_meta(fn, class_name, camera=name))

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/image')
    def review_image(name, class_name, filename):
        return _send_image(_safe_path(_camera_crops(name), class_name, filename))

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/approve', methods=['POST'])
    def review_approve(name, class_name, filename):
        src = _safe_path(_camera_crops(name), class_name, filename)
        if not src.is_file():
            abort(404)
        dest_dir = _safe_path(state.training_dir, class_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f'{name}_{filename}'  # camera prefix avoids cross-camera name collisions
        shutil.move(str(src), str(dest))
        logger.info('Approved %s/%s/%s → training', name, class_name, filename)
        return jsonify({'action': 'approved', 'file': filename})

    @app.route('/api/cameras/<name>/review/<class_name>/<filename>/reject', methods=['POST'])
    def review_reject(name, class_name, filename):
        src = _safe_path(_camera_crops(name), class_name, filename)
        if not src.is_file():
            abort(404)
        src.unlink()
        logger.info('Rejected %s/%s/%s', name, class_name, filename)
        return jsonify({'action': 'rejected', 'file': filename})

    # ----- Review (all cameras aggregated) -----

    def _camera_dirs() -> list[Path]:
        if not state.crops_dir.is_dir():
            return []
        return [d for d in sorted(state.crops_dir.iterdir()) if d.is_dir()]

    @app.route('/api/review/classes')
    def review_classes_all():
        counts: Counter[str] = Counter()
        for cam_dir in _camera_dirs():
            counts.update(_class_counts(cam_dir))
        return jsonify([{'name': k, 'count': counts[k]} for k in sorted(counts)])

    @app.route('/api/review/<class_name>')
    def review_crop_all(class_name):
        files = [(d.name, fn) for d in _camera_dirs() for fn in _list_images(d, class_name)]
        # image/approve/reject go through the per-camera routes with the crop's camera
        return _paged(files, 'crop', lambda item: _crop_meta(item[1], class_name, camera=item[0]))

    # ----- Training (global pool) -----

    @app.route('/api/training/classes')
    def training_classes():
        counts = _class_counts(state.training_dir)
        return jsonify([{'name': k, 'count': n} for k, n in counts.items()])

    @app.route('/api/training/<class_name>')
    def training_image(class_name):
        files = _list_images(state.training_dir, class_name)
        return _paged(files, 'image', lambda fn: _crop_meta(fn, class_name))

    @app.route('/api/training/<class_name>/<filename>/image')
    def training_serve_image(class_name, filename):
        return _send_image(_safe_path(state.training_dir, class_name, filename))

    @app.route('/api/training/<class_name>/<filename>', methods=['DELETE'])
    def training_delete(class_name, filename):
        path = _safe_path(state.training_dir, class_name, filename)
        if not path.is_file():
            abort(404)
        path.unlink()
        return jsonify({'deleted': filename})

    @app.after_request
    def _static_cache(resp):
        # Cache only /static/* (app.js/style.css); ETag/Last-Modified stay intact
        # so the browser still 304-revalidates after expiry. index.html ('/') and
        # event media keep their default no-cache so the app shell is never stale.
        if request.endpoint == 'static':
            resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    return app


# ---------- Helpers ----------

def _event_output_dir(state: CamerasState, event: dict) -> Path:
    camera = event.get('camera') or ''
    if camera:
        return (state.output_dir / camera).resolve()
    return state.output_dir.resolve()


def _event_path(state: CamerasState, event: dict, key: str) -> Path | None:
    rel = event.get(key)
    if not rel:
        return None
    path = Path(rel)
    return path if path.is_absolute() else _event_output_dir(state, event) / path


# Storage info is cached per-process (one Flask app per process) so /api/status
# never blocks on the NFS recordings walk; at most one background scan runs per TTL.
_STORAGE_TTL = 30.0
_storage_lock = threading.Lock()
_storage_cache: dict | None = None
_storage_cache_ts = 0.0
_storage_refreshing = False


def _scan_recordings_bytes(state: CamerasState) -> int:
    out = state.output_dir
    if not out.exists():
        return 0
    return sum(f.stat().st_size for f in out.rglob('*') if f.is_file())


def _build_storage_info(state: CamerasState, recordings_bytes: int) -> dict:
    cfg = state.config
    rec_cfg = getattr(cfg, 'recording', None) if cfg else None
    max_gb = rec_cfg.retention.max_storage_gb if rec_cfg else 0
    if max_gb > 0:
        budget_bytes = int(max_gb * 1024 * 1024 * 1024)
        budget_free = max(0, budget_bytes - recordings_bytes)
    else:
        try:
            stat = os.statvfs(str(state.output_dir))
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


def _refresh_storage_cache(state: CamerasState):
    global _storage_cache, _storage_cache_ts, _storage_refreshing
    try:
        info = _build_storage_info(state, _scan_recordings_bytes(state))
        with _storage_lock:
            _storage_cache = info
            _storage_cache_ts = time.monotonic()
    except OSError:
        logger.debug('storage scan failed; keeping previous value', exc_info=True)
    finally:
        with _storage_lock:
            _storage_refreshing = False


def _get_storage_info(state: CamerasState) -> dict:
    global _storage_refreshing
    now = time.monotonic()
    with _storage_lock:
        cached = _storage_cache
        fresh = cached is not None and (now - _storage_cache_ts) < _STORAGE_TTL
        if fresh:
            return cached
        spawn = not _storage_refreshing
        if spawn:
            _storage_refreshing = True
    if spawn:
        threading.Thread(target=_refresh_storage_cache, args=(state,),
                         daemon=True, name='storage-refresh').start()
    return cached if cached is not None else _build_storage_info(state, 0)


def _safe_path(base: Path, *parts: str) -> Path:
    resolved = (base / Path(*parts)).resolve()
    if not resolved.is_relative_to(base.resolve()):
        abort(400)
    return resolved


def _parse_crop_filename(filename: str) -> dict | None:
    """Parse track{id}_{YYYYMMDD}_{HHMMSS}[..._]{conf}.jpg with optional camera prefix."""
    m = re.search(r'track(\d+)_(\d{8})_(\d{6})_\d*_?(\d+\.\d+)\.jpg$', filename)
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


def _crop_meta(filename: str, class_name: str, camera: str | None = None) -> dict:
    meta = _parse_crop_filename(filename) or {}
    meta['filename'] = filename
    meta['class'] = class_name
    if camera is not None:
        meta['camera'] = camera
    return meta


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _list_images(base_dir: Path, class_name: str) -> list[str]:
    class_dir = _safe_path(base_dir, class_name)
    if not class_dir.is_dir():
        return []
    return sorted(f.name for f in class_dir.iterdir() if _is_image(f))


def _class_counts(base_dir: Path) -> dict[str, int]:
    if not base_dir.is_dir():
        return {}
    counts = {}
    for d in sorted(base_dir.iterdir()):
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if _is_image(f))
            if n:
                counts[d.name] = n
    return counts


def _paged(items: list, key: str, describe) -> Response:
    offset = request.args.get('offset', 0, type=int)
    if not items:
        return jsonify({'total': 0, 'offset': offset, key: None})
    offset = max(0, min(offset, len(items) - 1))
    return jsonify({'total': len(items), 'offset': offset, key: describe(items[offset])})


def _send_image(path: Path) -> Response:
    if not path.is_file():
        abort(404)
    return send_file(str(path), mimetype='image/jpeg')


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
                    chunk = f.read(min(262144, remaining))
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


def _next_jpeg(view: CameraView, last_sent: int) -> tuple[bytes | None, int]:
    with view.frame_cond:
        view.frame_cond.wait_for(
            lambda: view.jpeg_version != last_sent and view.frame_jpeg is not None,
            _STREAM_KEEPALIVE_S,
        )
        return view.frame_jpeg, view.jpeg_version


def _encode_loop(view: CameraView):
    # One shared encoder per camera: encodes each produced frame at most once and
    # fans the bytes out to every /stream client. Blocks (≈0 CPU) while unwatched;
    # imencode always runs OUTSIDE the lock on a reference copied under it (the
    # producer rebinds view.frame to a fresh array each tick, never mutates in place).
    cond = view.frame_cond
    last = -1
    while True:
        with cond:
            while not (view.stream_clients and view.frame is not None
                       and view.frame_version != last):
                cond.wait()
            frame = view.frame
            version = view.frame_version
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg = buf.tobytes() if ok else None
        with cond:
            last = version
            if jpeg is not None:
                view.frame_jpeg = jpeg
                view.jpeg_version += 1
                cond.notify_all()


def start_api_server(state: CamerasState, port: int, host: str = '0.0.0.0') -> threading.Thread:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app = create_app(state)

    for v in state.views.values():
        threading.Thread(target=_encode_loop, args=(v,), daemon=True,
                         name=f'jpeg-{v.name}').start()

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


# Pre-multi-camera name, still imported by older scripts.
PipelineState = CamerasState
