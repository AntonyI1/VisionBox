"""SQLite database for recording events."""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime


class RecordingDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._init_db()

    def _init_db(self):
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL,
                camera TEXT DEFAULT '',
                clean_clip TEXT,
                annotated_clip TEXT,
                thumbnail TEXT,
                detection_count INTEGER DEFAULT 0,
                top_label TEXT DEFAULT ''
            )
        ''')
        self._conn.commit()
        self._migrate()

    def _migrate(self):
        cursor = self._conn.execute('PRAGMA table_info(events)')
        columns = {row[1] for row in cursor.fetchall()}
        if 'thumbnail' not in columns:
            self._conn.execute('ALTER TABLE events ADD COLUMN thumbnail TEXT')
            self._conn.commit()

    def insert_event(self, event_id: str, start_time: datetime,
                     camera: str = '', clean_clip: str = '', annotated_clip: str = ''):
        with self._lock:
            self._conn.execute(
                'INSERT INTO events (event_id, start_time, camera, clean_clip, annotated_clip) '
                'VALUES (?, ?, ?, ?, ?)',
                (event_id, start_time.isoformat(), camera, clean_clip, annotated_clip)
            )
            self._conn.commit()

    def update_event_end(self, event_id: str, end_time: datetime, duration: float,
                         detection_count: int = 0, top_label: str = ''):
        with self._lock:
            self._conn.execute(
                'UPDATE events SET end_time=?, duration=?, detection_count=?, top_label=? '
                'WHERE event_id=?',
                (end_time.isoformat(), duration, detection_count, top_label, event_id)
            )
            self._conn.commit()

    def get_events_before(self, before: datetime) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                'SELECT * FROM events WHERE end_time IS NOT NULL AND end_time < ?',
                (before.isoformat(),)
            ).fetchall()
            self._conn.row_factory = None
        return [dict(row) for row in rows]

    def delete_event(self, event_id: str):
        with self._lock:
            self._conn.execute('DELETE FROM events WHERE event_id=?', (event_id,))
            self._conn.commit()

    def get_events(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                'SELECT * FROM events WHERE end_time IS NOT NULL '
                'ORDER BY start_time DESC LIMIT ? OFFSET ?',
                (limit, offset)
            ).fetchall()
            self._conn.row_factory = None
        return [dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict | None:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            row = self._conn.execute(
                'SELECT * FROM events WHERE event_id=?', (event_id,)
            ).fetchone()
            self._conn.row_factory = None
        return dict(row) if row else None

    def get_event_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                'SELECT COUNT(*) FROM events WHERE end_time IS NOT NULL'
            ).fetchone()
        return row[0] if row else 0

    def update_event_thumbnail(self, event_id: str, thumbnail: str):
        with self._lock:
            self._conn.execute(
                'UPDATE events SET thumbnail=? WHERE event_id=?',
                (thumbnail, event_id)
            )
            self._conn.commit()

    def close(self):
        self._conn.close()
