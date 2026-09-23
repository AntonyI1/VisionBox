import json

import pytest

from visionbox.zones import Zone, ZoneFilter

SHAPE = (100, 200)  # (h, w) as frame.shape
LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
LEFT_BOX = [10, 10, 30, 30]
RIGHT_BOX = [150, 10, 170, 30]


@pytest.fixture
def zones_path(tmp_path):
    return tmp_path / 'zones.json'


def _write(path, zones):
    path.write_text(json.dumps(zones))


def test_missing_file_means_no_restrictions(zones_path):
    zf = ZoneFilter(str(zones_path))
    assert zf.get_zones() == []
    assert not zf.has_exclude and not zf.has_include
    assert zf.filter_motion_regions([tuple(LEFT_BOX)], SHAPE) == [tuple(LEFT_BOX)]
    assert zf.filter_detections([{'box': LEFT_BOX}], SHAPE) == [{'box': LEFT_BOX}]
    assert zf.check_required_zones([], SHAPE) is True


def test_corrupt_file_is_ignored(zones_path):
    zones_path.write_text('not json')
    assert ZoneFilter(str(zones_path)).get_zones() == []


def test_exclude_zone_masks_motion_and_detections(zones_path):
    _write(zones_path, [{'name': 'street', 'type': 'exclude', 'points': LEFT_HALF}])
    zf = ZoneFilter(str(zones_path))
    assert zf.has_exclude and not zf.has_include

    assert zf.filter_motion_regions([tuple(LEFT_BOX), tuple(RIGHT_BOX)], SHAPE) == [tuple(RIGHT_BOX)]
    dets = [{'box': LEFT_BOX, 'class_id': 0}, {'box': RIGHT_BOX, 'class_id': 2}]
    assert zf.filter_detections(dets, SHAPE) == [dets[1]]
    assert zf.check_required_zones(dets, SHAPE) is True


def test_include_zone_requires_a_detection_inside(zones_path):
    _write(zones_path, [{'name': 'porch', 'type': 'include', 'points': RIGHT_HALF}])
    zf = ZoneFilter(str(zones_path))

    assert zf.check_required_zones([{'box': LEFT_BOX}], SHAPE) is False
    assert zf.check_required_zones([{'box': LEFT_BOX}, {'box': RIGHT_BOX}], SHAPE) is True
    assert zf.check_required_zones([], SHAPE) is False
    assert zf.filter_detections([{'box': LEFT_BOX}], SHAPE) == [{'box': LEFT_BOX}]


def test_normalized_points_scale_with_frame_resolution(zones_path):
    _write(zones_path, [{'name': 'street', 'type': 'exclude', 'points': LEFT_HALF}])
    zf = ZoneFilter(str(zones_path))
    box = [60, 10, 80, 30]  # center x=70: inside the left half of a 200px frame, outside for 100px
    assert zf.filter_detections([{'box': box}], (100, 200)) == []
    assert zf.filter_detections([{'box': box}], (100, 100)) == [{'box': box}]
    assert zf.filter_detections([{'box': box}], (100, 200)) == []


def test_add_and_remove_zone_persist_to_disk(zones_path):
    zf = ZoneFilter(str(zones_path))

    zf.add_zone(Zone(name='street', type='exclude', points=LEFT_HALF))
    assert json.loads(zones_path.read_text()) == [{'name': 'street', 'type': 'exclude', 'points': LEFT_HALF}]
    assert zf.filter_detections([{'box': LEFT_BOX}], SHAPE) == []

    zf.add_zone(Zone(name='street', type='exclude', points=RIGHT_HALF))
    assert [z['points'] for z in zf.get_zones()] == [RIGHT_HALF]
    assert zf.filter_detections([{'box': LEFT_BOX}], SHAPE) == [{'box': LEFT_BOX}]

    assert zf.remove_zone('street') is True
    assert zf.remove_zone('street') is False
    assert ZoneFilter(str(zones_path)).get_zones() == []
