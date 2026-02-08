"""Kalman filter for bounding box tracking.

State: [x, y, w, h, vx, vy, vw, vh] — position + velocity in center/size format.
"""

import numpy as np


class KalmanBoxTracker:
    """Tracks a single bounding box with a constant-velocity Kalman filter."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        self.state = self._bbox_to_state(bbox)
        self.state = np.concatenate([self.state, np.zeros(4)])

        self.P = np.diag([
            10, 10, 10, 10,        # position uncertainty
            1000, 1000, 10, 10     # velocity uncertainty (high initially)
        ]).astype(float)

        # Constant velocity transition: pos += vel each step
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # We observe position only, not velocity
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Tuned for 5 FPS detection with 15 FPS loop —
        # high velocity noise so filter doesn't overcommit between detections
        self.Q = np.diag([10, 10, 10, 10, 50, 50, 1, 1])
        self.R = np.diag([5, 5, 15, 15])

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox()

    def update(self, bbox: np.ndarray):
        z = self._bbox_to_state(bbox)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        return self._state_to_bbox()

    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.state[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
