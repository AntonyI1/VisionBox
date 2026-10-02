#!/usr/bin/env python3
"""Assemble the trainable YOLO dataset from VisionBox's auto-labelled frames.

Pairs every label with its image across the capture, review and corrected trees, drops
malformed labels, keeps a frozen validation split and emits
$STORAGE_DIR/datasets/yolo/{images,labels,data.yaml} as symlinks into the source trees.

    assemble_dataset.py [--dry-run]
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

BASE_WEIGHTS = ROOT / "yolov8n.pt"
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", ""))  # validated by require_storage_dir()

YOLO_ROOT = STORAGE_DIR / "datasets" / "yolo"
CUSTOM_CLASSES_YAML = YOLO_ROOT / "classes_custom.yaml"
SPLIT_FILE = YOLO_ROOT / "splits" / "val_frozen.txt"
COCO_REPLAY_ROOT = STORAGE_DIR / "datasets" / "coco_replay"
COCO_REPLAY_IMAGES = COCO_REPLAY_ROOT / "images"
COCO_REPLAY_LABELS = COCO_REPLAY_ROOT / "labels"
COCO_REPLAY_YAML = COCO_REPLAY_ROOT / "base.yaml"

CAPTURES_ROOT = STORAGE_DIR / "captures" / "dataset"
REVIEW_ROOT = STORAGE_DIR / "datasets" / "review"
CORRECTED_ROOT = STORAGE_DIR / "datasets" / "corrected"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VAL_FRACTION = 0.20
SPLIT_SEED = 0

SOURCES = (
    ("captures", CAPTURES_ROOT),
    ("review", REVIEW_ROOT),
    ("corrected", CORRECTED_ROOT),
)


def require_storage_dir():
    if not os.environ.get("STORAGE_DIR"):
        sys.exit("STORAGE_DIR is not set: copy .env.sample to .env and point it at the storage root")


def coco_names():
    from ultralytics import YOLO

    return {int(k): v for k, v in YOLO(str(BASE_WEIGHTS)).names.items()}


def _find_image(directory, stem):
    for ext in IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _camera_dirs(camera_root):
    if not camera_root.is_dir():
        return
    for camera_dir in sorted(p for p in camera_root.iterdir() if p.is_dir()):
        images_dir = camera_dir / "images"
        labels_dir = camera_dir / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            yield camera_dir.name, images_dir, labels_dir
        else:
            yield camera_dir.name, camera_dir, camera_dir


def load_pairs():
    pairs = []
    for source_name, source_root in SOURCES:
        for camera, images_dir, labels_dir in _camera_dirs(source_root):
            for label_path in sorted(labels_dir.glob("*.txt")):
                stem = label_path.stem
                if stem == "classes":
                    continue
                image_path = _find_image(images_dir, stem)
                if image_path is None:
                    continue
                try:
                    if image_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                pairs.append(
                    {
                        "key": f"{source_name}-{camera}__{stem}",
                        "source": source_name,
                        "camera": camera,
                        "image": image_path,
                        "label": label_path,
                    }
                )
    return pairs


def validate_label(line, nc):
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if class_id < 0 or class_id >= nc:
        return None
    if any(not (0.0 <= v <= 1.0) for v in coords):
        return None
    return class_id, f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"


def read_valid_label(label_path, nc):
    """Normalised valid lines of a label file, or None if it has none."""
    try:
        raw = label_path.read_text()
    except OSError:
        return None
    valid = []
    for raw_line in raw.splitlines():
        result = validate_label(raw_line.strip(), nc)
        if result is not None:
            valid.append(result[1])
    return valid or None


def build_class_names():
    names = coco_names()
    if CUSTOM_CLASSES_YAML.is_file():
        with open(CUSTOM_CLASSES_YAML) as f:
            doc = yaml.safe_load(f) or {}
        custom = doc.get("names", doc)
        if isinstance(custom, dict):
            for k, v in custom.items():
                names[int(k)] = v
        elif isinstance(custom, list):
            for offset, name in enumerate(custom):
                names[80 + offset] = name
    nc = max(80, max(names) + 1) if names else 80
    return names, nc


def frozen_split(keys, write=True):
    """Validation keys: previously frozen ones first, topped up deterministically to VAL_FRACTION."""
    keys = list(keys)
    key_set = set(keys)
    val_keys = set()
    if SPLIT_FILE.is_file():
        for line in SPLIT_FILE.read_text().splitlines():
            key = line.strip()
            if key and key in key_set:
                val_keys.add(key)

    target = round(len(keys) * VAL_FRACTION)
    if len(val_keys) < target:
        candidates = sorted(k for k in keys if k not in val_keys)
        random.Random(SPLIT_SEED).shuffle(candidates)
        for k in candidates:
            if len(val_keys) >= target:
                break
            val_keys.add(k)

    if write:
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text("\n".join(sorted(val_keys)) + "\n")
    return val_keys


def _relink(target, link_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def emit(dry_run=False):
    require_storage_dir()
    names, nc = build_class_names()
    pairs = load_pairs()

    kept = []
    class_counts = Counter()
    for pair in pairs:
        valid = read_valid_label(pair["label"], nc)
        if valid is None:
            continue
        pair["lines"] = valid
        kept.append(pair)
        class_counts.update(int(line.split()[0]) for line in valid)

    val_keys = frozen_split([p["key"] for p in kept], write=not dry_run)

    split_counts = {"train": 0, "val": 0}
    img_train = YOLO_ROOT / "images" / "train"
    img_val = YOLO_ROOT / "images" / "val"
    lbl_train = YOLO_ROOT / "labels" / "train"
    lbl_val = YOLO_ROOT / "labels" / "val"

    if not dry_run:
        for d in (img_train, img_val, lbl_train, lbl_val):
            if d.is_dir():
                for old in d.glob("*"):
                    if old.is_symlink():
                        old.unlink()
            d.mkdir(parents=True, exist_ok=True)

    for pair in kept:
        is_val = pair["key"] in val_keys
        split_counts["val" if is_val else "train"] += 1
        if dry_run:
            continue
        img_dir = img_val if is_val else img_train
        lbl_dir = lbl_val if is_val else lbl_train
        _relink(pair["image"], img_dir / f"{pair['key']}{pair['image'].suffix}")
        (lbl_dir / f"{pair['key']}.txt").write_text("\n".join(pair["lines"]) + "\n")

    data_yaml = YOLO_ROOT / "data.yaml"
    if not dry_run:
        doc = {
            "path": str(YOLO_ROOT),
            "train": ["images/train", str(COCO_REPLAY_IMAGES)],
            "val": "images/val",
            "nc": nc,
            "names": {int(k): names[k] for k in sorted(names)},
        }
        with open(data_yaml, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)

    return {
        "data_yaml": str(data_yaml),
        "names": names,
        "n_train": split_counts["train"],
        "n_val": split_counts["val"],
        "nc": nc,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assemble the trainable YOLO dataset from VisionBox's labelled frames."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the counts without writing the split, symlinks, labels or data.yaml.",
    )
    args = parser.parse_args()

    result = emit(dry_run=args.dry_run)
    names = result["names"]

    print(f"mode:        {'dry-run' if args.dry_run else 'write'}")
    print(f"data.yaml:   {result['data_yaml']}")
    print(f"nc:          {result['nc']}")
    print(f"train:       {result['n_train']}")
    print(f"val:         {result['n_val']}")
    print("per-class counts:")
    for class_id in sorted(result["class_counts"]):
        name = names.get(class_id, f"id{class_id}")
        print(f"  {class_id:>3} {name:<20} {result['class_counts'][class_id]}")


if __name__ == "__main__":
    main()
