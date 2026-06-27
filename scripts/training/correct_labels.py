#!/usr/bin/env python3
"""Assisted-labelling CLI for the VisionBox review queue.

Walks /mnt/storage/visionbox/datasets/review/<camera>, draws each prelabel over
its frame and lets a human accept / drop / reclass boxes or discard the frame.
Accepted frames (with any in-session edits applied) are emitted to
/mnt/storage/visionbox/datasets/corrected/<camera>/{images,labels} where
assemble_dataset.py ingests them. Labels stay in RAW COCO id space.

Interactive keys:
    a  accept frame (write img + edited label to corrected/)
    d  delete frame and its .jpg/.txt/.json sidecars from the review queue
    e  cycle the selected box's class id
    x  drop the selected box
    n / p  select next / previous box
    s  skip frame
    q  quit (cursor is persisted)

Headless: --headless --min-conf X auto-accepts every frame whose prelabel
confidences (from the sidecar .json) all exceed X, with no GUI.
"""

import argparse
import json
import os
import shutil
import sys

REVIEW_ROOT = "/mnt/storage/visionbox/datasets/review"
CORRECTED_ROOT = "/mnt/storage/visionbox/datasets/corrected"
CURSOR_PATH = os.path.join(REVIEW_ROOT, ".correct_cursor.json")

WINDOW = "VisionBox correct_labels"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _coco_names():
    from ultralytics import YOLO

    return dict(YOLO("/home/night/VisionBox/yolov8n.pt").names)


def _list_cameras():
    if not os.path.isdir(REVIEW_ROOT):
        return []
    return sorted(
        d for d in os.listdir(REVIEW_ROOT)
        if os.path.isdir(os.path.join(REVIEW_ROOT, d)) and not d.startswith(".")
    )


def _list_frames(camera):
    cam_dir = os.path.join(REVIEW_ROOT, camera)
    if not os.path.isdir(cam_dir):
        return []
    stems = []
    for f in sorted(os.listdir(cam_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS and os.path.isfile(os.path.join(cam_dir, stem + ".txt")):
            stems.append(stem)
    return stems


def _img_path(camera, stem):
    cam_dir = os.path.join(REVIEW_ROOT, camera)
    for ext in IMG_EXTS:
        p = os.path.join(cam_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return os.path.join(cam_dir, stem + ".jpg")


def _load_boxes(camera, stem):
    """Return list of [cls, cx, cy, w, h] from the YOLO prelabel."""
    txt = os.path.join(REVIEW_ROOT, camera, stem + ".txt")
    boxes = []
    try:
        with open(txt) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                boxes.append([int(float(parts[0]))] + [float(v) for v in parts[1:5]])
    except OSError:
        pass
    return boxes


def _load_confidences(camera, stem):
    js = os.path.join(REVIEW_ROOT, camera, stem + ".json")
    try:
        with open(js) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return [d.get("confidence") for d in data.get("detections", [])]


def _write_boxes(path, boxes):
    lines = [
        "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(c), cx, cy, w, h)
        for c, cx, cy, w, h in boxes
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


def _accept(camera, stem, boxes):
    out_img_dir = os.path.join(CORRECTED_ROOT, camera, "images")
    out_lbl_dir = os.path.join(CORRECTED_ROOT, camera, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    src_img = _img_path(camera, stem)
    ext = os.path.splitext(src_img)[1] or ".jpg"
    shutil.copy2(src_img, os.path.join(out_img_dir, stem + ext))
    _write_boxes(os.path.join(out_lbl_dir, stem + ".txt"), boxes)


def _delete(camera, stem):
    cam_dir = os.path.join(REVIEW_ROOT, camera)
    for ext in IMG_EXTS + (".txt", ".json"):
        p = os.path.join(cam_dir, stem + ext)
        if os.path.isfile(p):
            os.remove(p)


def _load_cursor():
    try:
        with open(CURSOR_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_cursor(camera, index):
    cursor = _load_cursor()
    cursor[camera] = index
    tmp = CURSOR_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cursor, fh)
    os.replace(tmp, CURSOR_PATH)


def _draw(cv2, image, names, boxes, confs, selected, info):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        chosen = i == selected
        color = (0, 0, 255) if chosen else (0, 200, 0)
        thickness = 3 if chosen else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = names.get(int(cls), str(int(cls)))
        if i < len(confs) and confs[i] is not None:
            label += " {:.2f}".format(confs[i])
        cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(canvas, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(canvas, "a accept  d delete  e class  x drop  n/p box  s skip  q quit",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return canvas


def run_interactive(camera, names):
    try:
        import cv2
    except ImportError:
        print("[correct_labels] opencv is not installed.", file=sys.stderr)
        return 1

    frames = _list_frames(camera)
    if not frames:
        print(f"[correct_labels] no reviewable frames for camera '{camera}'.")
        return 0

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error:
        print(
            "[correct_labels] cv2 has no display backend available on this host.\n"
            "  Run headless instead, e.g.:\n"
            f"    correct_labels.py --headless --min-conf 0.5 --camera {camera}",
            file=sys.stderr,
        )
        return 1

    nc = (max(names) + 1) if names else 1
    i = max(0, min(_load_cursor().get(camera, 0), len(frames)))
    loaded = -1
    boxes = []
    confs = []
    selected = 0

    while i < len(frames):
        stem = frames[i]
        if loaded != i:
            boxes = _load_boxes(camera, stem)
            confs = _load_confidences(camera, stem)
            selected = 0
            loaded = i

        img = cv2.imread(_img_path(camera, stem))
        if img is None:
            i += 1
            loaded = -1
            continue

        info = f"{camera}  {i + 1}/{len(frames)}  boxes={len(boxes)}"
        cv2.imshow(WINDOW, _draw(cv2, img, names, boxes, confs, selected, info))
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            _save_cursor(camera, i)
            break
        if key == ord("a"):
            _accept(camera, stem, boxes)
            i += 1
            loaded = -1
            _save_cursor(camera, i)
        elif key == ord("d"):
            _delete(camera, stem)
            frames.pop(i)
            loaded = -1
            _save_cursor(camera, i)
        elif key == ord("s"):
            i += 1
            loaded = -1
            _save_cursor(camera, i)
        elif key == ord("e") and boxes:
            boxes[selected][0] = (int(boxes[selected][0]) + 1) % nc
        elif key == ord("x") and boxes:
            boxes.pop(selected)
            if selected >= len(boxes):
                selected = max(0, len(boxes) - 1)
        elif key == ord("n") and boxes:
            selected = (selected + 1) % len(boxes)
        elif key == ord("p") and boxes:
            selected = (selected - 1) % len(boxes)

    cv2.destroyAllWindows()
    if i >= len(frames):
        print(f"[correct_labels] camera '{camera}' fully reviewed.")
    return 0


def run_headless(camera, min_conf):
    cameras = [camera] if camera else _list_cameras()
    accepted = 0
    scanned = 0
    for cam in cameras:
        for stem in _list_frames(cam):
            scanned += 1
            confs = _load_confidences(cam, stem)
            if not confs or any(c is None for c in confs):
                continue
            if all(c > min_conf for c in confs):
                _accept(cam, stem, _load_boxes(cam, stem))
                accepted += 1
    print(f"[correct_labels] headless: accepted {accepted}/{scanned} frames "
          f"(min-conf {min_conf}).")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Assisted labelling for the VisionBox review queue.")
    parser.add_argument("--camera", help="Camera name under the review root.")
    parser.add_argument("--headless", action="store_true",
                        help="Auto-accept high-confidence frames without a GUI.")
    parser.add_argument("--min-conf", type=float, default=0.6,
                        help="Headless confidence threshold (all boxes must exceed it).")
    args = parser.parse_args()

    os.makedirs(CORRECTED_ROOT, exist_ok=True)

    if args.headless:
        return run_headless(args.camera, args.min_conf)

    if not args.camera:
        parser.error("--camera is required for interactive mode (or pass --headless).")

    return run_interactive(args.camera, _coco_names())


if __name__ == "__main__":
    sys.exit(main())
