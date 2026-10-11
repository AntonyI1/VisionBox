import textwrap

import pytest

from visionbox.config import CameraConfig, ConfigError, MotionConfig, VisionBoxConfig, _resolve_env_vars, load_config


def test_resolve_env_vars_substitutes_known_and_rejects_unknown(monkeypatch):
    monkeypatch.setenv('STORAGE_DIR', '/mnt/storage')
    monkeypatch.delenv('MISSING_VAR', raising=False)
    assert _resolve_env_vars('${STORAGE_DIR}/recordings') == '/mnt/storage/recordings'
    assert _resolve_env_vars('plain') == 'plain'
    with pytest.raises(ConfigError, match='MISSING_VAR'):
        _resolve_env_vars('${MISSING_VAR}/x')


def test_emptied_sections_and_misspelled_keys_keep_defaults(tmp_path, caplog):
    cfg_file = tmp_path / 'config.yml'
    cfg_file.write_text(textwrap.dedent('''
        motion:
        detection:
          model:
          confidnce: 0.5
        tracker: 7
    '''))

    cfg = load_config(cfg_file)

    assert cfg.motion == MotionConfig()
    assert cfg.detection.model == 'yolov8n.pt'
    assert cfg.detection.confidence == 0.3
    assert 'detection.confidnce' in caplog.text
    assert 'tracker must be a mapping' in caplog.text


def test_missing_or_empty_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / 'nope.yml') == VisionBoxConfig()
    empty = tmp_path / 'empty.yml'
    empty.write_text('')
    assert load_config(empty) == VisionBoxConfig()


def test_load_config_applies_overrides_and_keeps_defaults(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('STORAGE_DIR', '/mnt/storage')
    cfg_file = tmp_path / 'config.yml'
    cfg_file.write_text(textwrap.dedent('''
        cameras:
          front_door:
            url: rtsp://127.0.0.1:8554/front_door
            detect_url: rtsp://127.0.0.1:8554/front_door_sub
            name: ignored
          backyard:
            url: rtsp://127.0.0.1:8554/backyard
            enabled: false
          bare:
        storage:
          crops: ${STORAGE_DIR}/captures/crops
        detection:
          device: gpu
          class_conf: {0: 0.5}
        recording:
          output_dir: ${STORAGE_DIR}/recordings
          clean:
            max_duration: 30.0
          retention:
            priority_labels: [person, dog]
        unknown_section:
          key: value
    '''))

    cfg = load_config(cfg_file)

    assert set(cfg.cameras) == {'front_door', 'backyard', 'bare'}
    assert cfg.cameras['front_door'] == CameraConfig(
        name='front_door',
        url='rtsp://127.0.0.1:8554/front_door',
        detect_url='rtsp://127.0.0.1:8554/front_door_sub',
    )
    assert cfg.cameras['bare'] == CameraConfig(name='bare')
    assert set(cfg.enabled_cameras()) == {'front_door', 'bare'}

    assert cfg.storage.crops == '/mnt/storage/captures/crops'
    assert cfg.storage.dataset == 'captures/dataset'
    assert cfg.detection.device == 'gpu'
    assert cfg.detection.class_conf == {0: 0.5}
    assert cfg.detection.imgsz == 640
    assert cfg.recording.output_dir == '/mnt/storage/recordings'
    assert cfg.recording.clean.max_duration == 30.0
    assert cfg.recording.clean.enabled is True
    assert cfg.recording.annotated.cooldown == 10.0
    assert cfg.recording.retention.priority_labels == ['person', 'dog']
    assert cfg.recording.retention.days == 14
    assert not hasattr(cfg, 'unknown_section')
    assert 'unknown key unknown_section' in caplog.text
    assert 'cameras.front_door.name' not in caplog.text  # a known field, just not overridable


def test_per_camera_path_helpers():
    cfg = VisionBoxConfig()
    cfg.recording.output_dir = '/data/rec'
    cfg.storage.crops = '/data/crops'
    cfg.storage.zones_dir = '/data/zones'
    assert str(cfg.camera_recordings_dir('cam')) == '/data/rec/cam'
    assert str(cfg.camera_crops_dir('cam')) == '/data/crops/cam'
    assert str(cfg.camera_zones_path('cam')) == '/data/zones/cam.json'
