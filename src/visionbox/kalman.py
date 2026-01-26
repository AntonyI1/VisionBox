"""
Kalman Filter for bounding box tracking.

State vector: [x, y, w, h, vx, vy, vw, vh]
- (x, y): center position
- (w, h): width and height
- (vx, vy, vw, vh): velocities (change per frame)

This is NOT machine learning. It's classical state estimation:
- Predict where object will be (assume constant velocity)
- Update estimate when we get a new detection
- Track uncertainty to know how much to trust prediction vs detection
"""

import numpy as np


class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman filter.

    The filter maintains 8 state variables and predicts the next position
    based on assumed constant velocity motion model.
    """

    count = 0  # Global ID counter

    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with first detection.

        Args:
            bbox: [x1, y1, x2, y2] format bounding box
        """
        # Convert [x1, y1, x2, y2] to [cx, cy, w, h]
        self.state = self._bbox_to_state(bbox)

        # Velocity starts at zero (we don't know direction yet)
        # Full state: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.concatenate([self.state, np.zeros(4)])

        # Uncertainty matrix (how confident we are in each state variable)
        # High values = uncertain, low values = confident
        # Start uncertain about velocity since we've only seen one frame
        self.P = np.diag([
            10, 10, 10, 10,     # position uncertainty (moderate)
            1000, 1000, 10, 10  # velocity uncertainty (high - we don't know yet)
        ]).astype(float)

        # State transition matrix: assumes constant velocity
        # x_new = x + vx,  y_new = y + vy, etc.
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx (constant)
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=float)

        # Measurement matrix: we only observe position, not velocity
        # Detection gives us [x, y, w, h], not velocities
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Process noise: how much we expect state to randomly change
        # Higher = model is less trusted, adapts faster to observations
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001, 0.0001])

        # Measurement noise: how much we trust detections
        # Lower = trust detections more
        self.R = np.diag([1, 1, 10, 10])  # trust position more than size

        # Track metadata
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1           # how many times detected
        self.age = 0            # frames since creation
        self.time_since_update = 0  # frames since last detection match

    def predict(self) -> np.ndarray:
        """
        Advance state forward one frame. Call once per frame.

        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # State prediction: x = F @ x (matrix multiplication)
        # This applies: x_new = x + vx, y_new = y + vy, etc.
        self.state = self.F @ self.state

        # Uncertainty grows when we predict (we're less sure about future)
        # P = F @ P @ F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        return self._state_to_bbox()

    def update(self, bbox: np.ndarray):
        """
        Update state with a matched detection.

        This is where prediction meets reality:
        - Compute difference between predicted and observed
        - Blend them based on respective uncertainties

        Args:
            bbox: Matched detection [x1, y1, x2, y2]
        """
        # Convert detection to measurement format
        z = self._bbox_to_state(bbox)

        # Innovation: difference between observation and prediction
        # "How wrong was our prediction?"
        y = z - self.H @ self.state

        # Innovation covariance: combined uncertainty
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: how much to trust observation vs prediction
        # High K = trust observation, Low K = trust prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state: blend prediction with observation
        self.state = self.state + K @ y

        # Update uncertainty: we're more confident after seeing a detection
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

        # Update metadata
        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate [x1, y1, x2, y2]."""
        return self._state_to_bbox()

    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])

    def _state_to_bbox(self) -> np.ndarray:
        """Convert internal state to [x1, y1, x2, y2]."""
        cx, cy, w, h = self.state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
