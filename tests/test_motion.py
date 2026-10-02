import numpy as np

from visionbox.motion import MotionDetector, MotionRegion, merge_overlapping_regions


def _region(x, y, w, h):
    return MotionRegion(x=x, y=y, w=w, h=h, area=w * h)


def test_region_box_is_xyxy():
    assert _region(1, 2, 3, 4).box == (1, 2, 4, 6)


def test_merge_returns_empty_for_no_regions():
    assert merge_overlapping_regions([]) == []


def test_merge_pads_and_clamps_at_frame_origin():
    assert merge_overlapping_regions([_region(5, 5, 10, 10)], padding=20) == [(0, 0, 35, 35)]


def test_merge_keeps_distant_regions_separate():
    merged = merge_overlapping_regions([_region(0, 0, 10, 10), _region(100, 100, 10, 10)], padding=5)
    assert merged == [(0, 0, 15, 15), (95, 95, 115, 115)]


def test_merge_fuses_overlapping_regions():
    merged = merge_overlapping_regions([_region(0, 0, 20, 20), _region(15, 15, 20, 20)], padding=0)
    assert merged == [(0, 0, 35, 35)]


def test_merge_is_transitive():
    # A touches C and C touches B, so all three fuse even though A and B are apart.
    regions = [_region(0, 0, 10, 10), _region(40, 0, 10, 10), _region(20, 0, 10, 10)]
    assert merge_overlapping_regions(regions, padding=6) == [(0, 0, 56, 16)]


def test_get_mask_shape_and_dtype():
    mask = MotionDetector().get_mask(np.zeros((50, 60, 3), dtype=np.uint8))
    assert mask.shape == (50, 60)
    assert mask.dtype == np.uint8


def test_detector_finds_a_new_object_against_a_learned_background():
    detector = MotionDetector(history=20, var_threshold=16.0, min_area=100)
    background = np.full((120, 160, 3), 40, dtype=np.uint8)
    for _ in range(30):
        detector.detect(background)
    assert detector.detect(background) == []
    assert detector.last_coverage == 0.0

    frame = background.copy()
    frame[10:50, 10:50] = 220  # entirely inside the top-left quadrant
    regions = detector.detect(frame)

    assert len(regions) == 1
    region = regions[0]
    assert region.x <= 10 and region.y <= 10
    assert region.x + region.w >= 50 and region.y + region.h >= 50
    assert 0.0 < detector.last_coverage < 0.5
    assert detector.quadrant_min == 0.0


def test_min_area_frac_raises_the_area_floor():
    detector = MotionDetector(history=20, min_area=100, min_area_frac=0.2)
    background = np.full((120, 160, 3), 40, dtype=np.uint8)
    for _ in range(30):
        detector.detect(background)
    frame = background.copy()
    frame[30:70, 50:90] = 220  # 1600 px < 20% of the frame
    assert detector.detect(frame) == []


def test_reset_rebuilds_subtractor_with_configured_params():
    detector = MotionDetector(history=42, var_threshold=9.0, detect_shadows=True)
    detector.reset()
    subtractor = detector.bg_subtractor
    assert subtractor.getHistory() == 42
    assert subtractor.getVarThreshold() == 9.0
    assert subtractor.getDetectShadows() is True
