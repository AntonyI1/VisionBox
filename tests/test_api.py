"""Flask API tests: auth gate, PWA exemptions, JSON shapes, review flow and path safety.
Run with: PYTHONPATH=src venv/bin/python3 -m pytest -q tests/test_api.py"""
import base64
from types import SimpleNamespace

import pytest
from werkzeug.exceptions import BadRequest

from visionbox import api

USER, PASSWORD = 'night', 'correct horse'
HTML = {'Accept': 'text/html,application/xhtml+xml'}
FRAME = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n%s\r\n'


def basic(user, password):
    token = base64.b64encode(f'{user}:{password}'.encode()).decode()
    return {'Authorization': f'Basic {token}'}


def crop(root, camera, class_name, name):
    path = root / camera / class_name / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\xff\xd8' + name.encode())
    return path


class FakeDB:
    def __init__(self, events):
        self.events = events

    def get_event(self, event_id):
        return self.events.get(event_id)

    def get_event_count(self, camera=None):
        return len(self.events)

    def get_camera_counts(self):
        return {}

    def delete_event(self, event_id):
        self.events.pop(event_id, None)


@pytest.fixture
def state(tmp_path):
    return api.CamerasState(
        output_dir=tmp_path / 'recordings',
        crops_dir=tmp_path / 'crops',
        training_dir=tmp_path / 'training',
        zones_dir=tmp_path / 'zones',
    )


@pytest.fixture
def no_auth_env(monkeypatch):
    for key in ('VISIONBOX_AUTH_USER', 'VISIONBOX_AUTH_PASS', 'VISIONBOX_SECRET_KEY'):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def auth_env(monkeypatch, no_auth_env):
    monkeypatch.setenv('VISIONBOX_AUTH_USER', USER)
    monkeypatch.setenv('VISIONBOX_AUTH_PASS', PASSWORD)
    monkeypatch.setenv('VISIONBOX_SECRET_KEY', 'test-secret')


@pytest.fixture
def client(state, no_auth_env):
    return api.create_app(state).test_client()


@pytest.fixture
def auth_client(state, auth_env):
    return api.create_app(state).test_client()


# ----- Auth -----

def test_partial_auth_env_fails_closed(state, no_auth_env, monkeypatch):
    monkeypatch.setenv('VISIONBOX_AUTH_USER', USER)
    with pytest.raises(SystemExit):
        api.create_app(state)


def test_auth_disabled_without_env(client):
    assert client.get('/api/status').status_code == 200
    assert client.get('/login').status_code == 404


def test_browser_request_redirects_to_login(auth_client):
    resp = auth_client.get('/api/status?x=1', headers=HTML)
    assert resp.status_code == 302
    assert resp.headers['Location'] == '/login?next=%2Fapi%2Fstatus%3Fx%3D1'


def test_api_request_gets_plain_401_without_challenge(auth_client):
    responses = (
        auth_client.get('/api/status'),
        auth_client.post('/api/model/reload'),
        auth_client.delete('/api/events/e1'),
    )
    for resp in responses:
        assert resp.status_code == 401
        assert 'WWW-Authenticate' not in resp.headers


@pytest.mark.parametrize('path', ['/manifest.webmanifest', '/sw.js', '/favicon.ico', '/static/app.js', '/login'])
def test_pwa_plumbing_is_public(auth_client, path):
    assert auth_client.get(path).status_code == 200


def test_session_login_and_logout(auth_client):
    page = auth_client.get('/login')
    assert page.status_code == 200
    assert f'value="{USER}"' in page.get_data(as_text=True)

    bad = auth_client.post('/login', data={'username': USER, 'password': 'wrong'})
    assert bad.status_code == 401
    assert 'Incorrect username or password' in bad.get_data(as_text=True)
    assert 'Set-Cookie' not in bad.headers
    assert auth_client.get('/api/status').status_code == 401

    good = auth_client.post('/login', data={'username': USER, 'password': PASSWORD})
    assert good.status_code == 302
    assert good.headers['Location'] == '/'
    cookie = good.headers['Set-Cookie']
    assert 'HttpOnly' in cookie and 'SameSite=Lax' in cookie and 'Expires=' in cookie
    assert auth_client.get('/api/status').status_code == 200

    out = auth_client.get('/logout')
    assert out.status_code == 302
    assert out.headers['Location'] == '/login'
    assert auth_client.get('/api/status').status_code == 401


@pytest.mark.parametrize('nxt, location', [
    ('/events', '/events'),
    ('/api/status?x=1', '/api/status?x=1'),
    ('//evil.example', '/'),
    ('/%5Cevil.example', '/'),
    ('http://evil.example/', '/'),
])
def test_login_next_is_same_site_only(auth_client, nxt, location):
    resp = auth_client.post('/login?next=' + nxt, data={'username': USER, 'password': PASSWORD})
    assert resp.status_code == 302
    assert resp.headers['Location'] == location


def test_basic_auth_accepted_for_scripts(auth_client):
    assert auth_client.get('/api/status', headers=basic(USER, PASSWORD)).status_code == 200
    assert auth_client.get('/api/status', headers=basic(USER, 'wrong')).status_code == 401
    assert auth_client.get('/api/status', headers=basic('', PASSWORD)).status_code == 401


# ----- Status and cameras -----

def test_status_shape(client):
    data = client.get('/api/status').get_json()
    assert data['cameras'] == 0
    assert data['connected_cameras'] == 0
    assert data['recording_cameras'] == 0
    assert data['any_recording'] is False
    assert data['offline'] is False
    assert data['event_count'] == 0
    assert isinstance(data['uptime'], int)
    assert set(data['storage']) == {
        'recordings_bytes', 'recordings_human',
        'disk_total_bytes', 'disk_total_human',
        'disk_free_bytes', 'disk_free_human',
    }


def test_cameras_empty_without_config(client):
    assert client.get('/api/cameras').get_json() == []


def test_cameras_lists_enabled_config_entries(state, no_auth_env):
    state.config = SimpleNamespace(cameras={
        'front': SimpleNamespace(enabled=True),
        'spare': SimpleNamespace(enabled=False),
    })
    state.views['front'] = api.CameraView(name='front', fps=4.5, connected=True, frame_count=12)
    client = api.create_app(state).test_client()

    cams = client.get('/api/cameras').get_json()
    assert cams == [{
        'name': 'front', 'enabled': True, 'fps': 4.5, 'recording': False, 'state': 'idle',
        'connected': True, 'last_error': '', 'frame_count': 12, 'event_count': 0,
    }]

    cams = client.get('/api/cameras?include_disabled=true').get_json()
    assert [(c['name'], c['enabled'], c['state']) for c in cams] == [
        ('front', True, 'idle'), ('spare', False, 'offline'),
    ]


@pytest.mark.parametrize('path', ['/api/cameras/nope/stream', '/api/cameras/nope/snapshot', '/api/cameras/nope/zones'])
def test_unknown_camera_is_404(client, path):
    assert client.get(path).status_code == 404


def test_stream_fans_out_encoded_jpeg_and_releases_client_slot(client, state, monkeypatch):
    monkeypatch.setattr(api, '_STREAM_KEEPALIVE_S', 0.01)
    view = api.CameraView(name='cam', frame_jpeg=b'JPEG1', jpeg_version=1)
    state.views['cam'] = view

    resp = client.get('/api/cameras/cam/stream')
    assert resp.status_code == 200
    assert resp.mimetype == 'multipart/x-mixed-replace'
    chunks = iter(resp.response)
    assert next(chunks) == FRAME % b'JPEG1'
    assert view.stream_clients == 1
    assert next(chunks) == FRAME % b'JPEG1'  # keepalive re-send while the camera is stalled

    with view.frame_cond:
        view.frame_jpeg, view.jpeg_version = b'JPEG2', 2
        view.frame_cond.notify_all()
    assert next(chunks) == FRAME % b'JPEG2'

    resp.close()
    assert view.stream_clients == 0


# ----- Events -----

def test_event_media_resolves_relative_to_camera_dir(client, state):
    cam_dir = state.output_dir / 'front'
    cam_dir.mkdir(parents=True)
    (cam_dir / 'e1_thumb.jpg').write_bytes(b'\xff\xd8thumb')
    state.db = FakeDB({'e1': {
        'event_id': 'e1', 'camera': 'front', 'thumbnail': 'e1_thumb.jpg',
        'snapshot': None, 'clean_clip': 'e1.mp4', 'annotated_clip': None,
    }})

    assert client.get('/api/events/e1/thumbnail').data == b'\xff\xd8thumb'
    assert client.get('/api/events/e1/snapshot').status_code == 404
    assert client.get('/api/events/e1/clip/clean').status_code == 404  # listed but missing on disk
    assert client.get('/api/events/e1/clip/bogus').status_code == 400
    assert client.get('/api/events/nope/thumbnail').status_code == 404
    assert client.get('/api/events/nope').status_code == 404


def test_delete_event_removes_files_and_sidecars(client, state):
    cam_dir = state.output_dir / 'front'
    cam_dir.mkdir(parents=True)
    for name in ('e1.mp4', 'e1.json', 'e1_thumb.jpg'):
        (cam_dir / name).write_bytes(b'x')
    db = FakeDB({'e1': {'event_id': 'e1', 'camera': 'front', 'thumbnail': 'e1_thumb.jpg', 'clean_clip': 'e1.mp4'}})
    state.db = db

    assert client.delete('/api/events/e1').get_json() == {'deleted': 'e1'}
    assert not any(cam_dir.iterdir())
    assert 'e1' not in db.events
    assert client.delete('/api/events/e1').status_code == 404


# ----- Review and training -----

def test_review_aggregates_cameras_and_routes_actions_per_camera(client, state):
    a = crop(state.crops_dir, 'cam_a', 'person', 'track1_20260913_185503_0.91.jpg')
    b = crop(state.crops_dir, 'cam_b', 'person', 'track2_20260913_185504_0.72.jpg')
    crop(state.crops_dir, 'cam_b', 'car', 'track3_20260913_185505_0.80.jpg')

    assert client.get('/api/review/classes').get_json() == [{'name': 'car', 'count': 1}, {'name': 'person', 'count': 2}]
    assert client.get('/api/cameras/cam_b/review/classes').get_json() == [
        {'name': 'car', 'count': 1}, {'name': 'person', 'count': 1},
    ]

    second = client.get('/api/review/person?offset=1').get_json()
    assert second['total'] == 2 and second['offset'] == 1
    assert second['crop'] == {
        'camera': 'cam_b', 'class': 'person', 'filename': b.name,
        'track_id': 2, 'confidence': 0.72, 'timestamp': '2026-09-13T18:55:04',
    }
    assert client.get('/api/review/person?offset=99').get_json()['offset'] == 1
    assert client.get('/api/review/nothing').get_json() == {'total': 0, 'offset': 0, 'crop': None}

    img = client.get(f'/api/cameras/cam_b/review/person/{b.name}/image')
    assert img.status_code == 200
    assert img.mimetype == 'image/jpeg'
    assert img.data == b.read_bytes()

    assert client.post(f'/api/cameras/cam_a/review/person/{a.name}/approve').get_json() == {
        'action': 'approved', 'file': a.name,
    }
    assert not a.exists()
    assert (state.training_dir / 'person' / f'cam_a_{a.name}').exists()
    assert client.post(f'/api/cameras/cam_b/review/person/{b.name}/reject').get_json() == {
        'action': 'rejected', 'file': b.name,
    }
    assert not b.exists()
    assert client.post(f'/api/cameras/cam_b/review/person/{b.name}/reject').status_code == 404

    assert client.get('/api/training/classes').get_json() == [{'name': 'person', 'count': 1}]
    item = client.get('/api/training/person').get_json()
    assert item['total'] == 1
    assert item['image']['filename'] == f'cam_a_{a.name}'
    assert 'camera' not in item['image']
    assert client.delete(f'/api/training/person/cam_a_{a.name}').get_json() == {'deleted': f'cam_a_{a.name}'}
    assert client.get('/api/training/person').get_json() == {'total': 0, 'offset': 0, 'image': None}


# ----- Path safety -----

def test_safe_path_rejects_escaping_the_base(tmp_path):
    base = tmp_path / 'crops'
    assert api._safe_path(base, 'person', 'x.jpg') == (base / 'person' / 'x.jpg').resolve()
    for parts in (('..', 'x.jpg'), ('person', '../../etc/passwd'), ('/etc/passwd',), ('..',)):
        with pytest.raises(BadRequest):
            api._safe_path(base, *parts)


def test_review_routes_reject_dot_dot_camera(client, state):
    crop(state.crops_dir, 'cam', 'person', 'track1_20260913_185503_0.91.jpg')
    for path in ('/api/cameras/../review/classes', '/api/cameras/../review/crops',
                 '/api/cameras/../review/crops/x.jpg/image'):
        assert client.get(path).status_code == 400
    assert client.post('/api/cameras/../review/crops/x.jpg/reject').status_code == 400
