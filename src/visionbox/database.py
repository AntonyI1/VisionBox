"""SQLite database for recording events."""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class RecordingDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
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
                snapshot TEXT,
                detection_count INTEGER DEFAULT 0,
                top_label TEXT DEFAULT ''
            )
        ''')
        self._conn.commit()
        self._migrate()
        self._conn.execute('CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera)')
        self._conn.execute('CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time DESC)')
        self._conn.commit()

    def _migrate(self):
        cursor = self._conn.execute('PRAGMA table_info(events)')
        columns = {row[1] for row in cursor.fetchall()}
        if 'thumbnail' not in columns:
            self._conn.execute('ALTER TABLE events ADD COLUMN thumbnail TEXT')
        if 'camera' not in columns:
            self._conn.execute("ALTER TABLE events ADD COLUMN camera TEXT DEFAULT ''")
        if 'snapshot' not in columns:
            self._conn.execute('ALTER TABLE events ADD COLUMN snapshot TEXT')
        if 'quarantined' not in columns:
            self._conn.execute('ALTER TABLE events ADD COLUMN quarantined INTEGER DEFAULT 0')
        if 'quarantine_reason' not in columns:
            self._conn.execute('ALTER TABLE events ADD COLUMN quarantine_reason TEXT')
        self._conn.commit()

    def _execute(self, query: str, params: tuple = ()):
        with self._lock:
            self._conn.execute(query, params)
            self._conn.commit()

    def _select(self, query: str, params: tuple | list = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def insert_event(self, event_id: str, start_time: datetime,
                     camera: str = '', clean_clip: str = '', annotated_clip: str = ''):
        self._execute(
            'INSERT INTO events (event_id, start_time, camera, clean_clip, annotated_clip) '
            'VALUES (?, ?, ?, ?, ?)',
            (event_id, start_time.isoformat(), camera, clean_clip, annotated_clip),
        )

    def update_event_end(self, event_id: str, end_time: datetime, duration: float,
                         detection_count: int = 0, top_label: str = ''):
        self._execute(
            'UPDATE events SET end_time=?, duration=?, detection_count=?, top_label=? '
            'WHERE event_id=?',
            (end_time.isoformat(), duration, detection_count, top_label, event_id),
        )

    def get_events_before(self, before: datetime) -> list[dict]:
        return self._select(
            'SELECT * FROM events WHERE end_time IS NOT NULL AND end_time < ?',
            (before.isoformat(),),
        )

    def delete_event(self, event_id: str):
        self._execute('DELETE FROM events WHERE event_id=?', (event_id,))

    def get_events(
        self, limit: int = 50, offset: int = 0, camera: str | None = None,
    ) -> list[dict]:
        query = 'SELECT * FROM events WHERE end_time IS NOT NULL AND COALESCE(quarantined,0)=0'
        params: list = []
        if camera:
            query += ' AND camera=?'
            params.append(camera)
        query += ' ORDER BY start_time DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        return self._select(query, params)

    def get_event(self, event_id: str) -> dict | None:
        rows = self._select('SELECT * FROM events WHERE event_id=?', (event_id,))
        return rows[0] if rows else None

    def get_event_count(self, camera: str | None = None) -> int:
        query = 'SELECT COUNT(*) FROM events WHERE end_time IS NOT NULL AND COALESCE(quarantined,0)=0'
        params: list = []
        if camera:
            query += ' AND camera=?'
            params.append(camera)
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def get_camera_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                'SELECT camera, COUNT(*) FROM events WHERE end_time IS NOT NULL '
                'AND COALESCE(quarantined,0)=0 GROUP BY camera'
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_overflow_events(
        self, label: str, max_keep: int, camera: str | None = None,
    ) -> list[dict]:
        query = 'SELECT * FROM events WHERE end_time IS NOT NULL AND top_label=?'
        params: list = [label]
        if camera:
            query += ' AND camera=?'
            params.append(camera)
        query += ' ORDER BY start_time DESC LIMIT -1 OFFSET ?'
        params.append(max_keep)
        return self._select(query, params)

    def get_label_counts(self, camera: str | None = None) -> dict[str, int]:
        query = 'SELECT top_label, COUNT(*) FROM events WHERE end_time IS NOT NULL'
        params: list = []
        if camera:
            query += ' AND camera=?'
            params.append(camera)
        query += ' GROUP BY top_label'
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row[0]: row[1] for row in rows if row[0]}

    def get_events_by_delete_priority(self, priority_labels: list[str]) -> list[dict]:
        """Return events ordered for deletion: non-priority oldest first, then priority oldest."""
        placeholders = ','.join('?' * len(priority_labels)) if priority_labels else "''"
        query = (
            'SELECT * FROM events WHERE end_time IS NOT NULL '
            'ORDER BY '
            f'CASE WHEN top_label IN ({placeholders}) THEN 1 ELSE 0 END ASC, '
            'start_time ASC'
        )
        return self._select(query, priority_labels)

    def update_event_thumbnail(self, event_id: str, thumbnail: str):
        self._execute('UPDATE events SET thumbnail=? WHERE event_id=?', (thumbnail, event_id))

    def update_event_snapshot(self, event_id: str, snapshot: str):
        self._execute('UPDATE events SET snapshot=? WHERE event_id=?', (snapshot, event_id))

    def update_event_clean_clip(self, event_id: str, clean_clip: str):
        self._execute('UPDATE events SET clean_clip=? WHERE event_id=?', (clean_clip, event_id))

    def close(self):
        with self._lock:
            self._conn.close()
