"""Multi-object tracker (SORT algorithm) using Kalman filters and Hungarian matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxTracker


def iou_numpy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def iou_cost_matrix(tracks: list[KalmanBoxTracker], detections: np.ndarray) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    track_boxes = np.array([t.get_state() for t in tracks])
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for t_idx, track_box in enumerate(track_boxes):
        for d_idx, det_box in enumerate(detections):
            cost_matrix[t_idx, d_idx] = 1 - iou_numpy(track_box, det_box)

    return cost_matrix


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

    matches = []
    for t_idx, d_idx in zip(track_indices, det_indices):
        if cost_matrix[t_idx, d_idx] > (1 - iou_threshold):
            continue
        matches.append((t_idx, d_idx))

    matched_tracks = {m[0] for m in matches}
    matched_dets = {m[1] for m in matches}
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

        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update <= self.max_coast:
                bbox = track.get_state()
                results.append([*bbox, track.id])

        return np.array(results) if results else np.empty((0, 5))

    def reset(self):
        self.tracks = []
        KalmanBoxTracker.count = 0
