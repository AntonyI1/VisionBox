import io
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

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


class FakeDeadFFmpeg:
    """A clean-copy ffmpeg that fails at once, leaving a header-sized stub behind."""

    returncode = 1

    def __init__(self, cmd, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b'\x00' * 10)
        kwargs['stderr'].write(b'Connection refused\n')

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def send_signal(self, sig):
        pass

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


def test_clean_clip_that_ffmpeg_abandoned_is_dropped_from_the_row(tmp_path, db, monkeypatch, caplog):
    monkeypatch.setattr(subprocess, 'Popen', FakeDeadFFmpeg)
    monkeypatch.setattr(shutil, 'which', lambda name: '/usr/bin/ffmpeg')
    cfg = _config()
    cfg.clean.enabled = True
    cfg.annotated.enabled = False
    out = tmp_path / 'front_door'
    mgr = RecordingManager(cfg, rtsp_url='rtsp://cam/stream', camera='front_door', output_dir=out, db=db)

    mgr.update(FRAME, None, triggered=True, detections=[_person()])
    event_id = mgr.event_id
    assert db.get_event(event_id)['clean_clip'] == f'clean/event_{event_id}.mp4'
    mgr.update(FRAME, None, triggered=False)

    assert db.get_event(event_id)['clean_clip'] == ''
    assert list((out / 'clean').iterdir()) == []  # stub clip and ffmpeg log both removed
    assert 'Connection refused' in caplog.text
    mgr.stop()


def test_abort_closes_the_event_and_the_annotated_recorder(manager, db):
    manager.update(FRAME, None, triggered=True, detections=[_person()])
    first = manager.event_id

    manager.abort()

    assert not manager.is_recording and manager.state == RecorderState.IDLE
    assert db.get_event(first)['end_time'] is not None
    manager.update(FRAME, None, triggered=False)  # the reconnect's first, quiet frame
    assert not manager.is_recording
    assert db.get_event_count() == 1


def test_without_annotated_recorder_events_follow_cooldown_and_max_duration(tmp_path, db):
    cfg = _config()
    cfg.annotated.enabled = False
    cfg.annotated.cooldown = 5.0
    cfg.annotated.max_duration = 60.0
    mgr = RecordingManager(cfg, camera='front_door', output_dir=tmp_path / 'front_door', db=db)
    clock = {'t': 1000.0}
    mgr._clock = lambda: clock['t']

    mgr.update(FRAME, None, triggered=True, detections=[_person()])
    first = mgr.event_id
    assert mgr.is_recording and mgr.state == RecorderState.RECORDING
    clock['t'] += 1
    mgr.update(FRAME, None, triggered=False)
    assert mgr.is_recording and mgr.event_id == first and mgr.state == RecorderState.COOLDOWN
    clock['t'] += 3
    mgr.update(FRAME, None, triggered=True)
    assert mgr.state == RecorderState.RECORDING
    clock['t'] += 5
    mgr.update(FRAME, None, triggered=False)
    assert not mgr.is_recording and mgr.state == RecorderState.IDLE
    row = db.get_event(first)
    assert row['end_time'] is not None and row['annotated_clip'] == ''

    mgr.update(FRAME, None, triggered=True)
    second = mgr.event_id
    assert second != first  # started within the same second as the first: ids stay unique
    clock['t'] += 61
    mgr.update(FRAME, None, triggered=True)
    assert not mgr.is_recording  # max_duration backstop
    assert db.get_event_count() == 2
    mgr.stop()


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

    assert manager._delete_event_files({'event_id': 'x', 'camera': 'front_door', **files}) == 45
    assert not any((out / rel).exists() for rel in files.values())
    assert not (out / 'clean/event_x.json').exists()
    missing = {'event_id': 'x', 'camera': 'front_door', 'clean_clip': 'clean/missing.mp4'}
    assert manager._delete_event_files(missing) == 0


def test_retention_thread_survives_a_failing_pass_and_vanished_files(tmp_path, db, fake_ffmpeg, monkeypatch, caplog):
    out = tmp_path / 'front_door'
    mgr = RecordingManager(_config(days=1, check_interval=0.01), camera='front_door', output_dir=out, db=db)
    old = datetime.now() - timedelta(days=2)
    db.insert_event('old', old, camera='front_door', clean_clip='clean/event_old.mp4')  # clip already gone from disk
    db.update_event_end('old', old + timedelta(seconds=5), 5.0, 1, 'person')

    calls = {'n': 0}
    real_query = db.get_events_before

    def flaky(before):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError('stale NFS file handle')
        return real_query(before)

    monkeypatch.setattr(db, 'get_events_before', flaky)
    mgr.start()
    deadline = time.time() + 5
    while db.get_event('old') is not None and time.time() < deadline:
        time.sleep(0.02)
    mgr.stop()

    assert db.get_event('old') is None
    assert calls['n'] >= 2
    assert 'retention pass failed' in caplog.text


def test_delete_event_files_reaches_quarantined_media(manager, tmp_path):
    parked = tmp_path / 'front_door' / '_quarantine' / 'clean' / 'event_q.mp4'
    parked.parent.mkdir(parents=True)
    parked.write_bytes(b'x' * 7)
    parked.with_suffix('.json').write_bytes(b'y' * 3)

    row = {'event_id': 'q', 'camera': 'front_door', 'clean_clip': 'clean/event_q.mp4', 'quarantined': 1}
    assert manager._delete_event_files(row) == 10
    assert not any(parked.parent.iterdir())


def test_start_settles_rows_a_crash_left_open(tmp_path, db, fake_ffmpeg):
    out = tmp_path / 'front_door'
    started = datetime(2026, 1, 1, 12, 0, 0)
    db.insert_event('kept', started, camera='front_door', clean_clip='clean/event_kept.mp4',
                    annotated_clip='annotated/event_kept.mp4')
    db.insert_event('gone', started, camera='front_door', clean_clip='clean/event_gone.mp4',
                    annotated_clip='annotated/event_gone.mp4')
    db.insert_event('other', started, camera='backyard', clean_clip='clean/event_other.mp4')
    kept_clip = out / 'clean' / 'event_kept.mp4'
    kept_clip.parent.mkdir(parents=True)
    kept_clip.write_bytes(b'x' * 60000)
    ended = (started + timedelta(seconds=42)).timestamp()
    os.utime(kept_clip, (ended, ended))
    stub = out / 'annotated' / 'event_gone.mp4'
    stub.parent.mkdir(parents=True)
    stub.write_bytes(b'x')

    mgr = RecordingManager(_config(), camera='front_door', output_dir=out, db=db)
    mgr.start()
    mgr.stop()

    kept = db.get_event('kept')
    assert kept['end_time'] == (started + timedelta(seconds=42)).isoformat()
    assert kept['duration'] == 42.0
    assert kept['annotated_clip'] == ''  # the annotated file never made it to disk
    assert kept_clip.exists()
    assert db.get_event('gone') is None and not stub.exists()
    assert db.get_event('other')['end_time'] is None  # another camera's row is not ours to settle


def test_storage_budget_spans_all_cameras(tmp_path, db, fake_ffmpeg):
    root = tmp_path / 'recordings'
    budget_gb = 1536 / 1024 ** 3
    managers = {}
    for i, cam in enumerate(('front_door', 'backyard')):
        managers[cam] = RecordingManager(_config(max_storage_gb=budget_gb), camera=cam, output_dir=root / cam, db=db)
        start = datetime(2026, 1, 1, 12, i)
        db.insert_event(f'e_{cam}', start, camera=cam, clean_clip=f'clean/event_e_{cam}.mp4')
        db.update_event_end(f'e_{cam}', start + timedelta(seconds=5), 5.0, 1, 'car')
        clip = root / cam / 'clean' / f'event_e_{cam}.mp4'
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b'x' * 1000)

    managers['backyard']._enforce_storage_limit()

    # 2000 bytes against a 1536-byte budget: the oldest event goes, even though it
    # belongs to the other camera, and the tree is then left alone.
    assert not (root / 'front_door' / 'clean' / 'event_e_front_door.mp4').exists()
    assert (root / 'backyard' / 'clean' / 'event_e_backyard.mp4').exists()
    assert [e['event_id'] for e in db.get_events()] == ['e_backyard']
    for mgr in managers.values():
        mgr.stop()


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
