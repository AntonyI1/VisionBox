import numpy as np
import pytest

from visionbox.kalman import KalmanBoxTracker
from visionbox.tracker import Tracker, associate_detections_to_tracks, iou_cost_matrix, iou_matrix


def _reference_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def _det(x, y, size=20, conf=0.9, cls=0):
    return np.array([[x, y, x + size, y + size, conf, cls]], dtype=float)


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [0, 0, 0, 0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 25 / 175, 0.0, 0.0]])
    assert iou_matrix(np.zeros((1, 4)), np.zeros((1, 4))) == [[0.0]]


def test_iou_matrix_matches_pairwise_reference():
    rng = np.random.default_rng(0)
    origins = rng.uniform(0, 100, size=(6, 2))
    a = np.hstack([origins, origins + rng.uniform(1, 50, size=(6, 2))])
    origins = rng.uniform(0, 100, size=(4, 2))
    b = np.hstack([origins, origins + rng.uniform(1, 50, size=(4, 2))])

    expected = [[_reference_iou(x, y) for y in b] for x in a]
    np.testing.assert_allclose(iou_matrix(a, b), expected)


def test_iou_cost_matrix_handles_empty_inputs():
    track = KalmanBoxTracker(np.array([0, 0, 10, 10]))
    assert iou_cost_matrix([], np.empty((0, 4))).shape == (0, 0)
    assert iou_cost_matrix([track], np.empty((0, 4))).shape == (1, 0)
    np.testing.assert_allclose(iou_cost_matrix([track], np.array([[0, 0, 10, 10]])), [[0.0]])


def test_association_matches_by_iou_and_reports_unmatched():
    tracks = [KalmanBoxTracker(np.array([0, 0, 10, 10])), KalmanBoxTracker(np.array([50, 50, 60, 60]))]
    detections = np.array([
        [51, 51, 61, 61, 0.9, 0],
        [1, 0, 11, 10, 0.9, 0],
        [200, 200, 210, 210, 0.9, 0],
    ])
    matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(tracks, detections, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched_dets == [2]
    assert unmatched_tracks == []


def test_association_rejects_overlap_below_threshold():
    tracks = [KalmanBoxTracker(np.array([0, 0, 10, 10]))]
    detections = np.array([[8, 8, 18, 18, 0.9, 0]])  # IoU = 4 / 196
    assert associate_detections_to_tracks(tracks, detections, 0.3) == ([], [0], [0])


def test_association_with_no_tracks_or_no_detections():
    tracks = [KalmanBoxTracker(np.array([0, 0, 10, 10]))]
    assert associate_detections_to_tracks([], np.zeros((2, 6))) == ([], [0, 1], [])
    assert associate_detections_to_tracks(tracks, np.empty((0, 6))) == ([], [], [0])


@pytest.fixture
def tracker():
    instance = Tracker(max_age=3, min_hits=2, iou_threshold=0.3, max_coast=1)
    instance.reset()
    return instance


def test_track_is_reported_only_after_min_hits(tracker):
    assert tracker.update(_det(0, 0)).shape == (0, 5)
    out = tracker.update(_det(1, 0))
    assert out.shape == (1, 5)
    assert out[0, 4] == 0


def test_track_keeps_its_id_while_moving(tracker):
    ids = set()
    for step in range(10):
        out = tracker.update(_det(step * 3, 0))
        ids.update(out[:, 4].astype(int).tolist())
    assert ids == {0}
    assert len(tracker.tracks) == 1


def test_coasting_track_is_hidden_then_dropped(tracker):
    tracker.update(_det(0, 0))
    tracker.update(_det(0, 0))
    empty = np.empty((0, 6))

    assert len(tracker.update(empty)) == 1  # one missed frame: still displayed (max_coast)
    assert len(tracker.update(empty)) == 0  # beyond max_coast: hidden but alive
    assert len(tracker.tracks) == 1
    tracker.update(empty)
    assert len(tracker.tracks) == 1  # time_since_update == max_age: kept
    tracker.update(empty)
    assert tracker.tracks == []  # beyond max_age: dropped


def test_new_object_gets_a_new_id(tracker):
    tracker.update(_det(0, 0))
    tracker.update(_det(0, 0))
    both = np.vstack([_det(0, 0), _det(100, 100)])
    tracker.update(both)
    out = tracker.update(both)
    assert sorted(out[:, 4].astype(int)) == [0, 1]


def test_reset_clears_tracks_and_ids(tracker):
    tracker.update(_det(0, 0))
    tracker.reset()
    assert tracker.tracks == []
    assert KalmanBoxTracker.count == 0
