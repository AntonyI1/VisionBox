from typing import ClassVar

import numpy as np
import pytest

from visionbox import detector_v2
from visionbox.detector_v2 import ModelConfig, MultiModelDetector

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


class FakeYOLO:
    """Records every inference call; never returns a detection."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, path, task=None):
        self.path = path
        self.names = {0: 'person', 2: 'car'}

    def __call__(self, frame, **kwargs):
        FakeYOLO.calls.append(kwargs)
        return []

    def to(self, device):
        return self


@pytest.fixture
def fake_yolo(monkeypatch):
    FakeYOLO.calls = []
    monkeypatch.setattr(detector_v2, 'YOLO', FakeYOLO)
    monkeypatch.setattr(detector_v2, '_has_cuda', lambda: False)
    monkeypatch.setattr(MultiModelDetector, '_find_best_model', staticmethod(lambda path: path))
    monkeypatch.setattr(MultiModelDetector, '_resolve_ov_device', staticmethod(lambda device: 'intel:gpu'))
    return FakeYOLO


def test_detect_uses_the_callers_confidence_unless_the_model_pins_one(fake_yolo):
    det = MultiModelDetector(
        [ModelConfig('coco.pt'), ModelConfig('plates.pt', class_offset=80, conf_threshold=0.6)], device='cpu',
    )
    assert det.class_names[82] == 'car'
    fake_yolo.calls.clear()

    det.detect(FRAME, conf_threshold=0.5)

    assert [c['conf'] for c in fake_yolo.calls] == [0.5, 0.6]


def test_reload_publishes_the_device_with_the_new_models(fake_yolo, tmp_path, monkeypatch):
    ov_dir = tmp_path / 'yolov8n_openvino_model'
    ov_dir.mkdir()
    det = MultiModelDetector([ModelConfig(str(ov_dir))], device='gpu')
    assert det.ov_device == 'intel:gpu'
    det.ov_device = 'intel:cpu'  # a live inference fell back to the CPU earlier

    seen_live_device = []

    def warm(self, frame, **kwargs):
        seen_live_device.append(det.ov_device)
        if kwargs.get('device') == 'intel:gpu':
            raise RuntimeError('GPU plugin unavailable')
        return []

    monkeypatch.setattr(FakeYOLO, '__call__', warm)
    info = det.reload()

    # The live model kept its device while the replacement was built and warmed...
    assert seen_live_device and set(seen_live_device) == {'intel:cpu'}
    # ...and the device that actually warmed up is what gets published.
    assert info['device'] == 'intel:cpu'
    assert det.effective_device == 'intel:cpu'
