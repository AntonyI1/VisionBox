"""
Multi-object tracker using Kalman filters and Hungarian algorithm.

This is the SORT algorithm (Simple Online Realtime Tracking):
1. Predict all existing tracks forward
2. Match detections to tracks using IoU + Hungarian algorithm
3. Update matched tracks with their detections
4. Create new tracks for unmatched detections
5. Delete tracks that haven't been seen in a while
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxTracker


def iou_numpy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
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
    """
    Compute cost matrix for Hungarian algorithm using IoU.

    Cost = 1 - IoU (so higher IoU = lower cost = better match)

    Args:
        tracks: List of existing trackers
        detections: Array of detections, shape (N, 4) as [x1, y1, x2, y2]

    Returns:
        Cost matrix of shape (num_tracks, num_detections)
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    # Get predicted positions for all tracks
    track_boxes = np.array([t.get_state() for t in tracks])

    # Compute IoU between every track and every detection
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for t_idx, track_box in enumerate(track_boxes):
        for d_idx, det_box in enumerate(detections):
            iou = iou_numpy(track_box, det_box)
            cost_matrix[t_idx, d_idx] = 1 - iou  # Convert to cost

    return cost_matrix


def associate_detections_to_tracks(
    tracks: list[KalmanBoxTracker],
    detections: np.ndarray,
    iou_threshold: float = 0.3
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match detections to existing tracks using Hungarian algorithm.

    The Hungarian algorithm solves the "assignment problem":
    Given N workers and M jobs with costs, assign workers to jobs
    to minimize total cost. Here: tracks are workers, detections are jobs.

    Args:
        tracks: Existing trackers
        detections: New detections [x1, y1, x2, y2, conf, class]
        iou_threshold: Minimum IoU to consider a valid match

    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_detections: Detection indices with no track match (new objects)
        unmatched_tracks: Track indices with no detection (occluded/left frame)
    """
    if len(tracks) == 0:
        return [], list(range(len(detections))), []

    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    # Build cost matrix
    cost_matrix = iou_cost_matrix(tracks, detections[:, :4])

    # Hungarian algorithm finds minimum cost assignment
    # linear_sum_assignment returns (row_indices, col_indices)
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matches = []
    for t_idx, d_idx in zip(track_indices, det_indices):
        if cost_matrix[t_idx, d_idx] > (1 - iou_threshold):
            # IoU too low, not a valid match
            continue
        matches.append((t_idx, d_idx))

    # Find unmatched
    matched_tracks = {m[0] for m in matches}
    matched_dets = {m[1] for m in matches}

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

    return matches, unmatched_dets, unmatched_tracks


class Tracker:
    """
    Multi-object tracker managing track lifecycle.

    Lifecycle:
    - Birth: new detection with no match → create new track
    - Update: matched detection → update track's Kalman filter
    - Death: track not matched for max_age frames → delete

    The min_hits parameter prevents flickering:
    tracks must be confirmed (seen multiple times) before being reported.
    """

    def __init__(
        self,
        max_age: int = 30,      # frames to keep unmatched track alive
        min_hits: int = 3,      # hits required before track is confirmed
        iou_threshold: float = 0.3,
        max_coast: int = 30     # frames to show track without fresh detection
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_coast = max_coast
        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Process one frame of detections.

        Args:
            detections: Shape (N, 6) as [x1, y1, x2, y2, confidence, class_id]

        Returns:
            Tracked objects (M, 5) as [x1, y1, x2, y2, track_id]
            Only returns confirmed tracks (hits >= min_hits)
        """
        # Step 1: Predict all tracks forward
        for track in self.tracks:
            track.predict()

        # Step 2: Match detections to tracks
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            self.tracks, detections, self.iou_threshold
        )

        # Step 3: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx, :4])

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = KalmanBoxTracker(detections[det_idx, :4])
            self.tracks.append(new_track)

        # Step 5: Remove dead tracks (not seen for too long)
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Step 6: Return confirmed tracks
        results = []
        for track in self.tracks:
            # Show tracks that have been confirmed (enough detections)
            # and haven't been lost for too long.
            # Between detection frames, Kalman predicts position — this
            # keeps boxes visible and smooth instead of flickering.
            if track.hits >= self.min_hits and track.time_since_update <= self.max_coast:
                bbox = track.get_state()
                results.append([*bbox, track.id])

        return np.array(results) if results else np.empty((0, 5))

    def reset(self):
        """Clear all tracks."""
        self.tracks = []
        KalmanBoxTracker.count = 0
