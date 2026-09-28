"""Drawing helpers shared by the surveillance pipeline and the demo scripts."""

import cv2
import numpy as np

PLATE_CLASS_ID = 80
PLATE_COLOR = (0, 255, 255)
MOTION_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Fixed palette so a track keeps its colour across frames and runs.
COLORS = [tuple(int(v) for v in rgb) for rgb in np.random.RandomState(42).randint(0, 255, (100, 3))]


def track_color(track_id: int, class_id: int = -1) -> tuple[int, int, int]:
    if class_id == PLATE_CLASS_ID:
        return PLATE_COLOR
    return COLORS[track_id % len(COLORS)]


def draw_labeled_box(image: np.ndarray, box, label: str, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), FONT, 0.5, (0, 0, 0), 1)


def draw_tracks(image: np.ndarray, tracks, track_classes: dict[int, int], class_names: dict[int, str]):
    """Box each (x1, y1, x2, y2, id) track row; track_classes maps track id -> class id."""
    for x1, y1, x2, y2, tid in tracks:
        track_id = int(tid)
        class_id = track_classes.get(track_id, 0)
        name = 'PLATE' if class_id == PLATE_CLASS_ID else class_names.get(class_id, f'class_{class_id}')
        draw_labeled_box(image, (x1, y1, x2, y2), f'{name} #{track_id}', track_color(track_id, class_id))
    return image


def draw_motion_regions(image: np.ndarray, regions, color=MOTION_COLOR):
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    return image


def box_iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0
