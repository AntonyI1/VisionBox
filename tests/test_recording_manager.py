import io
import json
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pytest

from visionbox.config import RecordingConfig
from visionbox.database import RecordingDatabase
from visionbox.recorder import RecorderState
from visionbox.recording_manager import RecordingManager

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeFFmpeg:
    """Stands in for the annotated recorder's ffmpeg pipe."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.stdin = io.BytesIO()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _config(**retention):
    cfg = RecordingConfig()
    cfg.clean.enabled = False
    cfg.annotated.cooldown = 0.0
    cfg.retention.days = 0
    for key, value in retention.items():
        setattr(cfg.retention, key, value)
    return cfg


def _person(conf=0.9, box=(10, 10, 40, 40)):
    return {'box': list(box), 'confidence': conf, 'class_id': 0, 'class_name': 'person'}


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', FakeFFmpeg)


@pytest.fixture
def db(tmp_path):
    database = RecordingDatabase(tmp_path / 'visionbox.db')
    yield database
    database.close()


@pytest.fixture
def manager(tmp_path, db, fake_ffmpeg):
    mgr = RecordingManager(_config(), camera='front_door', output_dir=tmp_path / 'front_door', db=db)
    mgr.start()
    yield mgr
    mgr.stop()


def test_event_lifecycle_writes_row_sidecar_thumbnail_and_snapshot(manager, db, tmp_path):
    out = tmp_path / 'front_door'

    manager.update(FRAME, None, triggered=True, detections=[_person()])
    assert manager.is_recording
    assert manager.state == RecorderState.RECORDING
    event_id = manager.event_id
    assert event_id.endswith('_front_door')
    row = db.get_event(event_id)
    assert row['camera'] == 'front_door'
    assert row['annotated_clip'].startswith('annotated/event_')
    assert row['clean_clip'] == ''
    assert row['end_time'] is None

    car = {'box': [0, 0, 5, 5], 'confidence': 0.8, 'class_id': 2, 'class_name': 'car'}
    manager.update(FRAME, None, triggered=True, detections=[_person(), car])
    manager.update(FRAME, None, triggered=False)
    assert manager.is_recording  # cooldown
    manager.update(FRAME, None, triggered=False)
    assert not manager.is_recording
    assert manager.event_id is None

    row = db.get_event(event_id)
    assert row['end_time'] is not None
    assert row['top_label'] == 'person'
    assert row['detection_count'] == 3
    assert (out / row['thumbnail']).is_file()
    assert (out / row['snapshot']).is_file()
    sidecar = json.loads((out / row['annotated_clip']).with_suffix('.json').read_text())
    assert sidecar['detection_count'] == 3
    assert sidecar['max_objects_in_frame'] == 2
    assert sidecar['clip'] == row['annotated_clip'].removeprefix('annotated/')


def test_trigger_subset_drives_counts_and_count_now_gates_tallying(manager, db):
    parked_car = {'box': [0, 0, 40, 40], 'confidence': 0.95, 'class_id': 2, 'class_name': 'car'}
    manager.update(FRAME, None, triggered=True, detections=[parked_car, _person()],
                   trigger_detections=[_person()])
    manager.update(FRAME, None, triggered=True, detections=[parked_car, _person()],
                   trigger_detections=[_person()], count_now=False)
    event_id = manager.event_id
    manager.update(FRAME, None, triggered=False)
    manager.update(FRAME, None, triggered=False)

    row = db.get_event(event_id)
    assert row['top_label'] == 'person'
    assert row['detection_count'] == 1


def test_frame_score_prefers_large_confident_centered_boxes():
    score = RecordingManager._frame_score
    assert score([], 100, 100) == 0.0
    assert score([{'box': [0, 0, 1], 'confidence': 1.0}], 100, 100) == 0.0

    centered = {'box': [25, 25, 75, 75], 'confidence': 0.8}
    edge = {'box': [0, 25, 50, 75], 'confidence': 0.8}
    assert score([centered], 100, 100) == pytest.approx(0.8 * 0.5)
    assert score([edge], 100, 100) == pytest.approx(0.8 * 0.5 * 0.6)
    assert score([edge, centered], 100, 100) == pytest.approx(0.4)


def test_delete_event_files_removes_media_and_sidecars(manager, tmp_path):
    out = tmp_path / 'front_door'
    files = {
        'clean_clip': 'clean/event_x.mp4',
        'annotated_clip': 'annotated/event_x.mp4',
        'thumbnail': 'thumbnails/event_x.jpg',
        'snapshot': 'snapshots/event_x.jpg',
    }
    for rel in files.values():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b'x' * 10)
    (out / 'clean/event_x.json').write_bytes(b'y' * 5)

    assert manager._delete_event_files({'event_id': 'x', **files}) == 45
    assert not any((out / rel).exists() for rel in files.values())
    assert not (out / 'clean/event_x.json').exists()
    assert manager._delete_event_files({'event_id': 'x', 'clean_clip': 'clean/missing.mp4'}) == 0


def test_per_label_cap_drops_oldest_events(tmp_path, db, fake_ffmpeg):
    out = tmp_path / 'front_door'
    mgr = RecordingManager(_config(max_per_label=1), camera='front_door', output_dir=out, db=db)
    for i in range(2):
        start = datetime(2026, 1, 1, 12, i)
        db.insert_event(f'e{i}', start, camera='front_door', clean_clip=f'clean/event_e{i}.mp4')
        db.update_event_end(f'e{i}', start + timedelta(seconds=5), 5.0, 1, 'person')
        clip = out / f'clean/event_e{i}.mp4'
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b'x')

    mgr._enforce_limits()

    assert [e['event_id'] for e in db.get_events()] == ['e1']
    assert not (out / 'clean/event_e0.mp4').exists()
    assert (out / 'clean/event_e1.mp4').exists()
    mgr.stop()
