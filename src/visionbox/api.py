"""Flask API server for VisionBox web UI and REST endpoints."""

import logging
import os
import re
import shutil
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, abort

from .database import RecordingDatabase
from .recording_manager import RecordingManager
from .zones import ZoneFilter, Zone

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    frame: np.ndarray | None = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    recording_mgr: RecordingManager | None = None
    zone_filter: ZoneFilter | None = None
    config: object = None
    fps: float = 0.0
    frame_count: int = 0
    event_count: int = 0
    start_time: float = field(default_factory=time.time)
    crops_dir: Path = field(default_factory=lambda: Path('/mnt/storage/visionbox/captures/crops'))
    training_dir: Path = field(default_factory=lambda: Path('/mnt/storage/visionbox/datasets/training'))


def create_app(state: PipelineState) -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), 'web'),
        static_url_path='/static',
    )

    def _db() -> RecordingDatabase:
        return state.recording_mgr.db

    def _output_dir() -> Path:
        return state.recording_mgr.output_dir.resolve()

    def _get_storage_info(mgr) -> dict:
        out = mgr.output_dir.resolve() if mgr else Path('.')
        recordings_bytes = sum(
            f.stat().st_size for f in out.rglob('*') if f.is_file()
        ) if out.exists() else 0
        max_gb = mgr.config.retention.max_storage_gb if mgr else 0
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

    @app.route('/')
    def index():
        return send_file(
            os.path.join(app.static_folder, 'index.html'),
            mimetype='text/html',
        )

    @app.route('/api/stream')
    def stream():
        def generate():
            while True:
                with state.frame_lock:
                    frame = state.frame
                if frame is not None:
                    _, jpeg = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg.tobytes()
                        + b'\r\n'
                    )
                time.sleep(0.033)

        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )

    @app.route('/api/status')
    def status():
        mgr = state.recording_mgr
        storage = _get_storage_info(mgr)
        return jsonify({
            'recording': mgr.is_recording if mgr else False,
            'state': mgr.state.value if mgr else 'idle',
            'fps': round(state.fps, 1),
            'frame_count': state.frame_count,
            'event_count': state.event_count,
            'uptime': round(time.time() - state.start_time),
            'storage': storage,
        })

    @app.route('/api/events')
    def events():
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        limit = min(limit, 200)

        rows = _db().get_events(limit=limit, offset=offset)
        total = _db().get_event_count()
        return jsonify({
            'events': rows,
            'total': total,
            'limit': limit,
            'offset': offset,
        })

    @app.route('/api/events/<event_id>')
    def event_detail(event_id):
        event = _db().get_event(event_id)
        if not event:
            abort(404)
        return jsonify(event)

    @app.route('/api/events/<event_id>', methods=['DELETE'])
    def delete_event(event_id):
        event = _db().get_event(event_id)
        if not event:
            abort(404)

        out = _output_dir()
        for key in ('clean_clip', 'annotated_clip', 'thumbnail'):
            rel = event.get(key, '')
            if rel:
                p = Path(rel)
                if not p.is_absolute():
                    p = out / p
                if p.exists():
                    p.unlink()
                if key != 'thumbnail':
                    meta = p.with_suffix('.json')
                    if meta.exists():
                        meta.unlink()

        _db().delete_event(event_id)
        return jsonify({'deleted': event_id})

    @app.route('/api/events/<event_id>/thumbnail')
    def event_thumbnail(event_id):
        event = _db().get_event(event_id)
        if not event or not event.get('thumbnail'):
            abort(404)

        thumb_path = Path(event['thumbnail'])
        if not thumb_path.is_absolute():
            thumb_path = _output_dir() / thumb_path
        if not thumb_path.exists():
            abort(404)

        return send_file(str(thumb_path), mimetype='image/jpeg')

    @app.route('/api/events/<event_id>/clip/<clip_type>')
    def event_clip(event_id, clip_type):
        if clip_type not in ('clean', 'annotated'):
            abort(400)

        event = _db().get_event(event_id)
        if not event:
            abort(404)

        key = f'{clip_type}_clip'
        rel = event.get(key, '')
        if not rel:
            abort(404)

        clip_path = Path(rel)
        if not clip_path.is_absolute():
            clip_path = _output_dir() / clip_path

        h264_path = clip_path.with_name(clip_path.stem + '.h264.mp4')
        serve_path = h264_path if h264_path.exists() else clip_path

        if not serve_path.exists():
            abort(404)

        return _send_video(str(serve_path))

    @app.route('/api/config')
    def config():
        if not state.config:
            return jsonify({})
        from dataclasses import asdict
        return jsonify(asdict(state.config))

    @app.route('/api/zones')
    def get_zones():
        if not state.zone_filter:
            return jsonify([])
        return jsonify(state.zone_filter.get_zones())

    @app.route('/api/zones', methods=['POST'])
    def add_zone():
        if not state.zone_filter:
            abort(500)
        data = request.get_json()
        if not data:
            abort(400)

        name = data.get('name', '').strip()
        ztype = data.get('type', '')
        points = data.get('points', [])

        if not name:
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

        state.zone_filter.add_zone(Zone(name=name, type=ztype, points=points))
        return jsonify({'ok': True})

    @app.route('/api/zones/<name>', methods=['DELETE'])
    def delete_zone(name):
        if not state.zone_filter:
            abort(500)
        if state.zone_filter.remove_zone(name):
            return jsonify({'deleted': name})
        abort(404)

    @app.route('/api/snapshot')
    def snapshot():
        with state.frame_lock:
            frame = state.frame
        if frame is None:
            abort(503)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

    # --- Helpers ---

    def _safe_path(base: Path, *parts: str) -> Path:
        resolved = (base / Path(*parts)).resolve()
        if not resolved.is_relative_to(base.resolve()):
            abort(400)
        return resolved

    def _parse_crop_filename(filename):
        """Parse track{id}_{YYYYMMDD}_{HHMMSS}_{confidence}.jpg"""
        m = re.match(
            r'track(\d+)_(\d{8})_(\d{6})_\d*_?(\d+\.\d+)\.jpg$', filename
        )
        if not m:
            return None
        track_id = int(m.group(1))
        date_str, time_str = m.group(2), m.group(3)
        confidence = float(m.group(4))
        try:
            timestamp = datetime.strptime(
                f'{date_str}_{time_str}', '%Y%m%d_%H%M%S'
            )
        except ValueError:
            timestamp = None
        return {
            'track_id': track_id,
            'timestamp': timestamp.isoformat() if timestamp else None,
            'confidence': round(confidence, 2),
        }

    def _list_images(base_dir: Path, class_name: str):
        class_dir = _safe_path(base_dir, class_name)
        if not class_dir.is_dir():
            return []
        return sorted(
            f.name for f in class_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        )

    @app.route('/api/review/classes')
    def review_classes():
        crops = state.crops_dir
        if not crops.is_dir():
            return jsonify([])
        classes = []
        for d in sorted(crops.iterdir()):
            if not d.is_dir():
                continue
            count = sum(
                1 for f in d.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            )
            if count > 0:
                classes.append({'name': d.name, 'count': count})
        return jsonify(classes)

    @app.route('/api/review/<class_name>')
    def review_crop(class_name):
        offset = request.args.get('offset', 0, type=int)
        files = _list_images(state.crops_dir, class_name)
        if not files:
            return jsonify({'total': 0, 'offset': offset, 'crop': None})

        offset = max(0, min(offset, len(files) - 1))
        filename = files[offset]
        meta = _parse_crop_filename(filename) or {}
        meta['filename'] = filename
        meta['class'] = class_name
        return jsonify({
            'total': len(files),
            'offset': offset,
            'crop': meta,
        })

    @app.route('/api/review/<class_name>/<filename>/image')
    def review_image(class_name, filename):
        path = _safe_path(state.crops_dir, class_name, filename)
        if not path.is_file():
            abort(404)
        return send_file(str(path), mimetype='image/jpeg')

    @app.route('/api/review/<class_name>/<filename>/approve', methods=['POST'])
    def review_approve(class_name, filename):
        src = _safe_path(state.crops_dir, class_name, filename)
        if not src.is_file():
            abort(404)

        dest_dir = _safe_path(state.training_dir, class_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        shutil.move(str(src), str(dest))
        logger.info('Approved %s/%s → training', class_name, filename)
        return jsonify({'action': 'approved', 'file': filename})

    @app.route('/api/review/<class_name>/<filename>/reject', methods=['POST'])
    def review_reject(class_name, filename):
        src = _safe_path(state.crops_dir, class_name, filename)
        if not src.is_file():
            abort(404)

        src.unlink()
        logger.info('Rejected %s/%s (deleted)', class_name, filename)
        return jsonify({'action': 'rejected', 'file': filename})

    # --- Training endpoints ---

    @app.route('/api/training/classes')
    def training_classes():
        tdir = state.training_dir
        if not tdir.is_dir():
            return jsonify([])
        classes = []
        for d in sorted(tdir.iterdir()):
            if not d.is_dir():
                continue
            count = sum(
                1 for f in d.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            )
            if count > 0:
                classes.append({'name': d.name, 'count': count})
        return jsonify(classes)

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
        return jsonify({
            'total': len(files),
            'offset': offset,
            'image': meta,
        })

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
        logger.info('Deleted training image %s/%s', class_name, filename)
        return jsonify({'deleted': filename})

    return app


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
        byte_start = 0
        byte_end = file_size - 1

        match = range_header.replace('bytes=', '').split('-')
        byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])

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
            generate(),
            status=206,
            mimetype='video/mp4',
            headers={
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': content_length,
            },
        )

    return send_file(path, mimetype='video/mp4')


def start_api_server(state: PipelineState, port: int) -> threading.Thread:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    app = create_app(state)
    thread = threading.Thread(
        target=lambda: app.run(
            host='0.0.0.0', port=port, threaded=True, use_reloader=False,
        ),
        daemon=True,
    )
    thread.start()
    return thread
