#!/usr/bin/env python3
"""Reversible de-duplication of Review crops.

Within each camera/class folder, near-identical crops (same object / same scene,
snapped repeatedly) are clustered by perceptual hash (dHash); the best one per
cluster is kept (highest detection confidence, then largest file) and the rest are
MOVED (never deleted) to captures/crops/_trash/ with a manifest for restore.

Usage:
  dedupe_crops.py                 # dry-run (counts only)
  dedupe_crops.py --apply         # do it
  dedupe_crops.py --restore       # move everything back
  dedupe_crops.py --threshold 6   # hamming distance for "near-identical" (default 6)
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

CROPS_DEFAULT = '/mnt/storage/visionbox/captures/crops'
IMG_EXT = ('.jpg', '.jpeg', '.png')
CONF_RE = re.compile(r'_(\d+\.\d+)\.(?:jpg|jpeg|png)$', re.I)


def manifest_path(root: Path) -> Path:
    return root / '_dedup_manifest.jsonl'


def dhash(path: Path, size: int = 8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    r = cv2.resize(img, (size + 1, size))
    h = 0
    for row in (r[:, 1:] > r[:, :-1]):
        for bit in row:
            h = (h << 1) | int(bit)
    return h


def conf_of(path: Path) -> float:
    m = CONF_RE.search(path.name)
    return float(m.group(1)) if m else -1.0


def cluster(folder: Path, threshold: int):
    """Return (keep, drop) lists of paths for one class folder."""
    files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT]
    clusters = []  # each: {'rep': hash, 'members': [path]}
    unhashable = []
    for f in sorted(files):
        h = dhash(f)
        if h is None:
            unhashable.append(f)
            continue
        for c in clusters:
            if bin(h ^ c['rep']).count('1') <= threshold:
                c['members'].append(f)
                break
        else:
            clusters.append({'rep': h, 'members': [f]})
    keep, drop = list(unhashable), []   # never drop something we couldn't hash
    for c in clusters:
        best = max(c['members'], key=lambda p: (conf_of(p), p.stat().st_size))
        keep.append(best)
        drop.extend(p for p in c['members'] if p != best)
    return keep, drop


def run(root: Path, threshold: int, apply: bool):
    trash = root / '_trash'
    total_keep = total_drop = 0
    rows = []
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir() or cam_dir.name in ('_trash',):
            continue
        for cls_dir in sorted(cam_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            keep, drop = cluster(cls_dir, threshold)
            if not (keep or drop):
                continue
            total_keep += len(keep)
            total_drop += len(drop)
            rows.append((cam_dir.name, cls_dir.name, len(keep) + len(drop), len(keep), len(drop)))
            if apply and drop:
                for p in drop:
                    dst = trash / cam_dir.name / cls_dir.name / p.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dst))
                with open(manifest_path(root), 'a') as mf:
                    for p in drop:
                        dst = trash / cam_dir.name / cls_dir.name / p.name
                        mf.write(json.dumps({'src': str(p), 'dst': str(dst)}) + '\n')

    print(f"crops root: {root}   threshold(hamming)<= {threshold}")
    print(f"{'camera':16s} {'class':10s} {'total':>6s} {'keep':>6s} {'drop':>6s}")
    for cam, cls, tot, k, d in rows:
        print(f"{cam:16s} {cls:10s} {tot:6d} {k:6d} {d:6d}")
    print(f"{'TOTAL':27s} {total_keep + total_drop:6d} {total_keep:6d} {total_drop:6d}")
    if not apply:
        print("\n[dry-run] nothing moved. Re-run with --apply to dedupe.")
    else:
        print(f"\nMoved {total_drop} duplicate crops to {trash}")
        print("Restore with:  python scripts/dedupe_crops.py --restore")


def restore(root: Path):
    mp = manifest_path(root)
    if not mp.exists():
        print("No manifest; nothing to restore.")
        return
    entries = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
    n = 0
    for e in entries:
        dst, src = Path(e['dst']), Path(e['src'])
        if dst.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            n += 1
    mp.unlink()
    print(f"Restored {n} crops.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crops', default=CROPS_DEFAULT)
    ap.add_argument('--threshold', type=int, default=6, help='max hamming distance for near-identical')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    root = Path(args.crops)
    if not root.is_dir():
        sys.exit(f"crops dir not found: {root}")
    if args.restore:
        restore(root)
    else:
        run(root, args.threshold, args.apply)


if __name__ == '__main__':
    main()
