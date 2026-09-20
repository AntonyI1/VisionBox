"""Multi-object tracker (SORT algorithm) using Kalman filters and Hungarian matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxTracker


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) and (M, 4) arrays of [x1, y1, x2, y2] boxes as an (N, M) array."""
    a = np.asarray(boxes_a, dtype=float)[:, None, :]
    b = np.asarray(boxes_b, dtype=float)[None, :, :]

    inter_w = np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    inter_h = np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def iou_cost_matrix(tracks: list[KalmanBoxTracker], detections: np.ndarray) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    track_boxes = np.array([t.get_state() for t in tracks])
    return 1 - iou_matrix(track_boxes, detections)


def associate_detections_to_tracks(
    tracks: list[KalmanBoxTracker],
    detections: np.ndarray,
    iou_threshold: float = 0.3
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to tracks via Hungarian algorithm on IoU cost."""
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    cost_matrix = iou_cost_matrix(tracks, detections[:, :4])
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    matches = [
        (t_idx, d_idx)
        for t_idx, d_idx in zip(track_indices, det_indices, strict=True)
        if cost_matrix[t_idx, d_idx] <= (1 - iou_threshold)
    ]

    matched_tracks = {t for t, _ in matches}
    matched_dets = {d for _, d in matches}
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

    return matches, unmatched_dets, unmatched_tracks


class Tracker:
    """Multi-object tracker with track birth/update/death lifecycle."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_coast: int = 30,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_coast = max_coast
        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame. Returns (M, 5) as [x1, y1, x2, y2, track_id]."""
        for track in self.tracks:
            track.predict()

        matches, unmatched_dets, _ = associate_detections_to_tracks(
            self.tracks, detections, self.iou_threshold
        )

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx, :4])

        for det_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(detections[det_idx, :4]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = [
            [*track.get_state(), track.id]
            for track in self.tracks
            if track.hits >= self.min_hits and track.time_since_update <= self.max_coast
        ]

        return np.array(results) if results else np.empty((0, 5))

    def reset(self):
        self.tracks = []
        KalmanBoxTracker.count = 0
