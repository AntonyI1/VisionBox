import sqlite3
from datetime import datetime, timedelta

import pytest

from visionbox.database import RecordingDatabase

T0 = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture
def db(tmp_path):
    database = RecordingDatabase(tmp_path / 'visionbox.db')
    yield database
    database.close()


def _add(db, event_id, camera='front_door', start=T0, end_after=5.0, label='person'):
    db.insert_event(event_id, start, camera=camera, clean_clip=f'clean/event_{event_id}.mp4')
    if end_after is not None:
        db.update_event_end(event_id, start + timedelta(seconds=end_after), end_after,
                            detection_count=3, top_label=label)


def test_insert_and_get_event(db):
    _add(db, 'e1')
    event = db.get_event('e1')
    assert event['event_id'] == 'e1'
    assert event['camera'] == 'front_door'
    assert event['start_time'] == T0.isoformat()
    assert event['end_time'] == (T0 + timedelta(seconds=5)).isoformat()
    assert event['duration'] == 5.0
    assert event['detection_count'] == 3
    assert event['top_label'] == 'person'
    assert event['clean_clip'] == 'clean/event_e1.mp4'
    assert event['annotated_clip'] == ''
    assert event['quarantined'] == 0
    assert db.get_event('missing') is None


def test_listings_hide_open_and_quarantined_events(db):
    _add(db, 'done')
    _add(db, 'open', end_after=None)
    _add(db, 'bad')
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE events SET quarantined=1 WHERE event_id='bad'")

    assert [e['event_id'] for e in db.get_events()] == ['done']
    assert db.get_event_count() == 1
    assert db.get_camera_counts() == {'front_door': 1}
    assert db.get_event('open')['end_time'] is None


def test_pagination_and_camera_filter(db):
    for i in range(5):
        _add(db, f'fd{i}', start=T0 + timedelta(minutes=i))
    _add(db, 'by0', camera='backyard', start=T0 + timedelta(hours=1))

    assert [e['event_id'] for e in db.get_events(limit=2, offset=1)] == ['fd4', 'fd3']
    assert [e['event_id'] for e in db.get_events(camera='backyard')] == ['by0']
    assert db.get_event_count(camera='front_door') == 5
    assert db.get_camera_counts() == {'front_door': 5, 'backyard': 1}


def test_delete_event(db):
    _add(db, 'e1')
    db.delete_event('e1')
    assert db.get_event('e1') is None
    assert db.get_event_count() == 0


def test_events_before_cutoff_only_returns_finished_events(db):
    _add(db, 'old', start=T0 - timedelta(days=30))
    _add(db, 'new', start=T0)
    _add(db, 'open', start=T0 - timedelta(days=30), end_after=None)
    before = db.get_events_before(T0 - timedelta(days=14))
    assert [e['event_id'] for e in before] == ['old']


def test_label_counts_and_overflow(db):
    for i in range(3):
        _add(db, f'p{i}', start=T0 + timedelta(minutes=i), label='person')
    _add(db, 'c0', label='car')
    _add(db, 'blank', label='')

    assert db.get_label_counts() == {'person': 3, 'car': 1}
    assert db.get_label_counts(camera='backyard') == {}
    assert [e['event_id'] for e in db.get_overflow_events('person', max_keep=2)] == ['p0']
    assert db.get_overflow_events('person', max_keep=3) == []


def test_delete_priority_puts_non_priority_oldest_first(db):
    _add(db, 'person_old', start=T0, label='person')
    _add(db, 'car_new', start=T0 + timedelta(hours=2), label='car')
    _add(db, 'car_old', start=T0 + timedelta(hours=1), label='car')

    ordered = [e['event_id'] for e in db.get_events_by_delete_priority(['person'])]
    assert ordered == ['car_old', 'car_new', 'person_old']
    unprioritized = [e['event_id'] for e in db.get_events_by_delete_priority([])]
    assert unprioritized == ['person_old', 'car_old', 'car_new']


def test_update_media_paths(db):
    _add(db, 'e1')
    db.update_event_thumbnail('e1', 'thumbnails/event_e1.jpg')
    db.update_event_snapshot('e1', 'snapshots/event_e1.jpg')
    db.update_event_clean_clip('e1', '')
    event = db.get_event('e1')
    assert event['thumbnail'] == 'thumbnails/event_e1.jpg'
    assert event['snapshot'] == 'snapshots/event_e1.jpg'
    assert event['clean_clip'] == ''


def test_migration_adds_missing_columns_to_a_legacy_table(tmp_path):
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute(
            'CREATE TABLE events (event_id TEXT PRIMARY KEY, start_time TEXT NOT NULL, end_time TEXT, '
            "duration REAL, clean_clip TEXT, annotated_clip TEXT, detection_count INTEGER DEFAULT 0, "
            "top_label TEXT DEFAULT '')"
        )
        conn.execute("INSERT INTO events (event_id, start_time, end_time) VALUES ('legacy', '2026', '2026')")

    db = RecordingDatabase(path)
    try:
        event = db.get_event('legacy')
        assert event['camera'] == ''
        assert event['thumbnail'] is None
        assert event['snapshot'] is None
        assert event['quarantined'] == 0
        assert event['quarantine_reason'] is None
        assert [e['event_id'] for e in db.get_events()] == ['legacy']
    finally:
        db.close()
