import numpy as np

from visionbox.kalman import KalmanBoxTracker


def test_initial_state_is_box_center_size_with_zero_velocity():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    tracker = KalmanBoxTracker(bbox)
    assert tracker.state.shape == (8,)
    np.testing.assert_allclose(tracker.state[:4], [30, 50, 40, 60])
    np.testing.assert_allclose(tracker.state[4:], 0)
    np.testing.assert_allclose(tracker.get_state(), bbox)


def test_predict_and_update_bookkeeping():
    tracker = KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0]))
    assert (tracker.hits, tracker.age, tracker.time_since_update) == (1, 0, 0)

    predicted = tracker.predict()
    assert predicted.shape == (4,)
    assert (tracker.age, tracker.time_since_update) == (1, 1)

    tracker.update(np.array([1.0, 1.0, 11.0, 11.0]))
    assert (tracker.hits, tracker.time_since_update) == (2, 0)


def test_stationary_box_stays_put():
    bbox = np.array([100.0, 100.0, 140.0, 160.0])
    tracker = KalmanBoxTracker(bbox)
    for _ in range(20):
        tracker.predict()
        tracker.update(bbox)
    np.testing.assert_allclose(tracker.predict(), bbox, atol=1e-6)


def test_constant_velocity_box_is_anticipated():
    velocity = np.array([4.0, 2.0, 4.0, 2.0])
    bbox = np.array([100.0, 100.0, 140.0, 160.0])
    tracker = KalmanBoxTracker(bbox)
    for step in range(1, 31):
        tracker.predict()
        tracker.update(bbox + velocity * step)

    np.testing.assert_allclose(tracker.predict(), bbox + velocity * 31, atol=1.0)
    assert tracker.state[4] > 0 and tracker.state[5] > 0


def test_ids_increment_per_instance():
    first = KalmanBoxTracker(np.zeros(4))
    second = KalmanBoxTracker(np.zeros(4))
    assert second.id == first.id + 1
