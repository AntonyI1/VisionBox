#!/usr/bin/env python3
"""Reversible quarantine of false-positive events.

Moves (never deletes) the matching events' files into a per-camera `_quarantine/`
tree, records everything in a manifest, and flags the rows `quarantined=1` so the
dashboard hides them. Fully restorable from the manifest.

Buckets:
  parked_car_spam     top_label=car and detection_count/duration >= RATIO  (front_garage parked-car re-detection)
  night_person_review top_label=person between 00:00-05:59                  (night/IR person artifacts — kept for REVIEW)
  orphan              end_time/duration NULL and older than ORPHAN_MIN min  (stalled/runaway events; skips in-progress)

Usage:
  quarantine_false_positives.py                 # dry-run (counts only)
  quarantine_false_positives.py --apply         # do it (backs up the DB first)
  quarantine_false_positives.py --restore [--reason night_person_review]
"""
import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_DEFAULT = '/mnt/storage/visionbox/recordings/visionbox.db'
RATIO = {'front_garage': 15.0}   # per-camera car det/sec spam cutoff (front_garage is a pure parked-car scene)
RATIO_DEFAULT = 40.0             # other cams: real traffic peaks ~30, so stay safe
NIGHT_HOURS = range(0, 6)
ORPHAN_MIN = 10       # only quarantine orphans older than this (avoids in-progress recordings)
FILE_KEYS = ('clean_clip', 'annotated_clip', 'thumbnail', 'snapshot')


def manifest_path(db_path: str) -> Path:
    return Path(db_path).parent / '_quarantine_manifest.jsonl'


def cam_out(db_path: str, camera: str) -> Path:
    # mirrors VisionBoxConfig.camera_recordings_dir: <recordings>/<camera>
    return Path(db_path).parent / (camera or '')


def classify(row: dict) -> str | None:
    end, dur = row.get('end_time'), row.get('duration')
    if not end or dur is None:
        try:
            age_min = (datetime.now() - datetime.fromisoformat(row['start_time'])).total_seconds() / 60
        except Exception:
            age_min = ORPHAN_MIN + 1
        return 'orphan' if age_min >= ORPHAN_MIN else None
    thr = RATIO.get(row.get('camera', ''), RATIO_DEFAULT)
    if row.get('top_label') == 'car' and dur and (row.get('detection_count', 0) / dur) >= thr:
        return 'parked_car_spam'
    if row.get('top_label') == 'person':
        try:
            if datetime.fromisoformat(row['start_time']).hour in NIGHT_HOURS:
                return 'night_person_review'
        except Exception:
            pass
    return None


def ensure_columns(conn):
    cols = {r[1] for r in conn.execute('PRAGMA table_info(events)')}
    if 'quarantined' not in cols:
        conn.execute('ALTER TABLE events ADD COLUMN quarantined INTEGER DEFAULT 0')
    if 'quarantine_reason' not in cols:
        conn.execute('ALTER TABLE events ADD COLUMN quarantine_reason TEXT')
    conn.commit()


def quarantine(db_path: str, apply: bool):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)
    rows = [dict(r) for r in conn.execute(
        'SELECT * FROM events WHERE COALESCE(quarantined,0)=0')]
    buckets: dict[str, list] = {}
    for row in rows:
        tag = classify(row)
        if tag:
            buckets.setdefault(tag, []).append(row)

    total = sum(len(v) for v in buckets.values())
    print(f"DB: {db_path}")
    print(f"Scanned {len(rows)} active events. Matches to quarantine: {total}")
    for tag, evs in sorted(buckets.items()):
        cams = {}
        for e in evs:
            cams[e.get('camera', '')] = cams.get(e.get('camera', ''), 0) + 1
        print(f"  {tag:20s} {len(evs):4d}   by camera: {cams}")
    print(f"  KEPT visible: {len(rows) - total}")

    if not apply:
        print("\n[dry-run] nothing moved. Re-run with --apply to quarantine.")
        conn.close()
        return

    # back up the DB
    bak = f"{db_path}.bak-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, bak)
    print(f"\nDB backed up -> {bak}")

    mpath = manifest_path(db_path)
    moved_files = moved_events = 0
    with open(mpath, 'a') as mf:
        for tag, evs in buckets.items():
            for row in evs:
                out = cam_out(db_path, row.get('camera', ''))
                qdir = out / '_quarantine'
                moved = []
                for key in FILE_KEYS:
                    rel = row.get(key) or ''
                    if not rel:
                        continue
                    candidates = [Path(rel)] if Path(rel).is_absolute() else [out / rel]
                    if key in ('clean_clip', 'annotated_clip'):
                        candidates.append(candidates[0].with_suffix('.json'))
                    for src in candidates:
                        if src.exists():
                            dst = qdir / src.relative_to(out)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(src), str(dst))
                            moved.append({'src': str(src), 'dst': str(dst)})
                            moved_files += 1
                mf.write(json.dumps({
                    'ts': datetime.now().isoformat(), 'event_id': row['event_id'],
                    'camera': row.get('camera', ''), 'reason': tag,
                    'row': row, 'moved': moved,
                }) + '\n')
                conn.execute('UPDATE events SET quarantined=1, quarantine_reason=? WHERE event_id=?',
                             (tag, row['event_id']))
                moved_events += 1
            conn.commit()
    conn.close()
    print(f"Quarantined {moved_events} events, relocated {moved_files} files.")
    print(f"Manifest: {mpath}")
    print("Restore everything with:  python scripts/quarantine_false_positives.py --restore")


def restore(db_path: str, reason: str | None):
    mpath = manifest_path(db_path)
    if not mpath.exists():
        print("No manifest; nothing to restore.")
        return
    conn = sqlite3.connect(db_path, timeout=30)
    ensure_columns(conn)
    entries = [json.loads(l) for l in mpath.read_text().splitlines() if l.strip()]
    done = 0
    keep = []
    for e in entries:
        if reason and e['reason'] != reason:
            keep.append(e)
            continue
        for mv in e.get('moved', []):
            dst, src = Path(mv['dst']), Path(mv['src'])
            if dst.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
        # row may have been reaped by retention — re-insert if missing
        if not conn.execute('SELECT 1 FROM events WHERE event_id=?', (e['event_id'],)).fetchone():
            row = e['row']
            cols = [c for c in row if c not in ('quarantined', 'quarantine_reason')]
            conn.execute(f"INSERT INTO events ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [row[c] for c in cols])
        conn.execute('UPDATE events SET quarantined=0, quarantine_reason=NULL WHERE event_id=?',
                     (e['event_id'],))
        done += 1
    conn.commit()
    conn.close()
    mpath.write_text(''.join(json.dumps(k) + '\n' for k in keep))
    print(f"Restored {done} events. Remaining quarantined in manifest: {len(keep)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DB_DEFAULT)
    ap.add_argument('--apply', action='store_true', help='perform the quarantine (default: dry-run)')
    ap.add_argument('--restore', action='store_true', help='move files back and un-flag rows')
    ap.add_argument('--reason', help='restrict --restore to one bucket (e.g. night_person_review)')
    args = ap.parse_args()
    if not Path(args.db).exists():
        sys.exit(f"DB not found: {args.db}")
    if args.restore:
        restore(args.db, args.reason)
    else:
        quarantine(args.db, args.apply)


if __name__ == '__main__':
    main()
