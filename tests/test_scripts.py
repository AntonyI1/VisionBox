#!/usr/bin/env python3
"""Tests for the pure helpers in scripts/ and src/visionbox/viz.py. Run with:
    venv/bin/python3 -m pytest -q tests/test_scripts.py
Script modules are loaded by path (scripts/ is not a package). The training scripts derive
their storage layout from STORAGE_DIR, which is pointed at a temp dir before they are loaded."""
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

from visionbox import viz
from visionbox.database import RecordingDatabase

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix='visionbox_scripts_test_')).resolve()
STORAGE = TMP / 'storage'
os.environ['STORAGE_DIR'] = str(STORAGE)


def load_script(rel_path):
    path = HERE.parent / 'scripts' / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module   # sibling imports (train_overnight -> assemble_dataset) share it
    spec.loader.exec_module(module)
    return module


assemble = load_script('training/assemble_dataset.py')
prep_replay = load_script('training/prep_replay.py')
promote = load_script('training/promote_model.py')
train = load_script('training/train_overnight.py')
quarantine = load_script('quarantine_false_positives.py')
dedupe = load_script('dedupe_crops.py')
surveillance = load_script('surveillance.py')


class VizHelpers(unittest.TestCase):
    def test_palette_matches_legacy_seeded_global(self):
        np.random.seed(42)
        legacy = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(0, 255, (100, 3))]
        self.assertEqual(viz.COLORS, legacy)

    def test_track_color(self):
        self.assertEqual(viz.track_color(7, 0), viz.COLORS[7])
        self.assertEqual(viz.track_color(107, 2), viz.COLORS[7])
        self.assertEqual(viz.track_color(7, viz.PLATE_CLASS_ID), viz.PLATE_COLOR)

    def test_draw_labeled_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        viz.draw_labeled_box(image, (10.4, 20.9, 50, 60), 'car #1', (1, 2, 3))
        self.assertEqual(tuple(image[40, 10]), (1, 2, 3))   # box edge
        self.assertEqual(tuple(image[40, 30]), (0, 0, 0))   # interior untouched

    def test_draw_tracks_and_motion_regions(self):
        tracks = np.array([[10, 20, 50, 60, 3.0]])
        image = viz.draw_tracks(np.zeros((100, 100, 3), dtype=np.uint8), tracks, {3: 2}, {2: 'car'})
        self.assertEqual(tuple(image[40, 10]), viz.COLORS[3])
        image = viz.draw_motion_regions(np.zeros((50, 50, 3), dtype=np.uint8), [(5, 5, 20, 20)])
        self.assertEqual(tuple(image[10, 5]), viz.MOTION_COLOR)


class SurveillanceHelpers(unittest.TestCase):
    def test_redact_url_strips_credentials(self):
        self.assertEqual(surveillance._redact_url('rtsp://admin:secret@10.0.0.5:554/cam'),
                         'rtsp://***@10.0.0.5:554/cam')
        self.assertEqual(surveillance._redact_url('rtsp://127.0.0.1:8554/front_door'),
                         'rtsp://127.0.0.1:8554/front_door')

    def test_overlaps_motion(self):
        self.assertTrue(surveillance._overlaps_motion((10, 10, 50, 50), [(40, 40, 80, 80)]))
        self.assertFalse(surveillance._overlaps_motion((10, 10, 50, 50), [(50, 10, 80, 80)]))  # edge contact
        self.assertFalse(surveillance._overlaps_motion((10, 10, 50, 50), []))

    def test_rtsp_ready_skips_non_rtsp_sources(self):
        self.assertTrue(surveillance._rtsp_ready('/tmp/clip.mp4'))


class AssembleDataset(unittest.TestCase):
    def test_validate_label(self):
        self.assertEqual(assemble.validate_label('2 0.5 0.5 0.25 0.125', 80),
                         (2, '2 0.500000 0.500000 0.250000 0.125000'))
        self.assertIsNone(assemble.validate_label('2 0.5 0.5 0.25', 80))
        self.assertIsNone(assemble.validate_label('80 0.5 0.5 0.2 0.2', 80))
        self.assertIsNone(assemble.validate_label('x 0.5 0.5 0.2 0.2', 80))
        self.assertIsNone(assemble.validate_label('0 1.5 0.5 0.2 0.2', 80))

    def test_read_valid_label_filters_bad_lines(self):
        path = TMP / 'label.txt'
        path.write_text('0 0.5 0.5 0.2 0.2\nbad line\n\n99 0.1 0.1 0.1 0.1\n')
        self.assertEqual(assemble.read_valid_label(path, 80), ['0 0.500000 0.500000 0.200000 0.200000'])
        path.write_text('bad\n')
        self.assertIsNone(assemble.read_valid_label(path, 80))
        self.assertIsNone(assemble.read_valid_label(TMP / 'missing.txt', 80))

    @staticmethod
    def _write_pair(images_dir, labels_dir, stem, label='0 0.5 0.5 0.2 0.2\n'):
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / f'{stem}.jpg').write_bytes(b'\xff\xd8 not really a jpeg')
        (labels_dir / f'{stem}.txt').write_text(label)

    def test_emit_dry_run_is_read_only_and_write_materialises(self):
        shutil.rmtree(STORAGE, ignore_errors=True)
        cap = assemble.CAPTURES_ROOT / 'front_door'
        for i in range(4):
            self._write_pair(cap / 'images', cap / 'labels', f'f{i}')
        review = assemble.REVIEW_ROOT / 'backyard'              # flat layout: image + label side by side
        self._write_pair(review, review, 'r0', '2 0.5 0.5 0.3 0.3\n')
        self._write_pair(review, review, 'bad', 'nope\n')       # no valid line -> dropped
        original = assemble.build_class_names
        assemble.build_class_names = lambda: ({0: 'person', 2: 'car'}, 80)
        self.addCleanup(setattr, assemble, 'build_class_names', original)

        result = assemble.emit(dry_run=True)
        self.assertEqual((result['n_train'], result['n_val']), (4, 1))
        self.assertEqual(result['class_counts'], {0: 4, 2: 1})
        self.assertFalse(assemble.SPLIT_FILE.exists())
        self.assertFalse(assemble.YOLO_ROOT.exists())

        assemble.emit(dry_run=False)
        val_keys = assemble.SPLIT_FILE.read_text().split()
        self.assertEqual(len(val_keys), 1)
        data = yaml.safe_load((assemble.YOLO_ROOT / 'data.yaml').read_text())
        self.assertEqual(data['nc'], 80)
        self.assertEqual(data['train'], ['images/train', str(assemble.COCO_REPLAY_IMAGES)])
        train_links = list((assemble.YOLO_ROOT / 'images' / 'train').iterdir())
        self.assertEqual(len(train_links), 4)
        self.assertTrue(all(p.is_symlink() and p.resolve().is_file() for p in train_links))
        self.assertEqual(len(list((assemble.YOLO_ROOT / 'labels' / 'val').iterdir())), 1)

        assemble.emit(dry_run=False)   # the split is frozen across runs
        self.assertEqual(assemble.SPLIT_FILE.read_text().split(), val_keys)


class PrepReplay(unittest.TestCase):
    def test_img2label(self):
        self.assertEqual(prep_replay._img2label('/d/coco128/images/train/a.jpg'), '/d/coco128/labels/train/a.txt')
        self.assertEqual(prep_replay._img2label('/d/a.png'), '/d/a.txt')

    def test_not_populated_without_replay_set(self):
        self.assertFalse(prep_replay._is_populated())


class TrainOvernight(unittest.TestCase):
    def test_gate_fails_closed_without_replay_set(self):
        self.assertEqual(train.regression_gate('models/x', 'best.pt', TMP / 'missing.yaml'), (False, None, None))

    def test_stage_candidate_and_report(self):
        self.addCleanup(setattr, promote, 'CANDIDATE_LINK', promote.CANDIDATE_LINK)
        promote.CANDIDATE_LINK = str(TMP / 'candidate')
        export = TMP / 'stage_run' / 'weights' / 'best_openvino_model'
        export.mkdir(parents=True)
        train._stage_candidate(str(export))
        train._stage_candidate(str(export))   # re-staging replaces the link
        self.assertEqual(os.readlink(promote.CANDIDATE_LINK), str(export))
        train._write_report(TMP / 'stage_run', {'gate': 'PASS'})
        self.assertEqual(json.loads((TMP / 'stage_run' / 'REPORT.json').read_text()), {'gate': 'PASS'})


class PromoteModel(unittest.TestCase):
    def setUp(self):
        self.models = TMP / f'models_{self._testMethodName}'
        self.models.mkdir()
        for name, value in {
            'MODELS_DIR': str(self.models),
            'ACTIVE_LINK': str(self.models / 'yolov8n_openvino_model'),
            'CANDIDATE_LINK': str(self.models / 'candidate'),
            'PREV_ACTIVE': str(self.models / '.prev_active'),
        }.items():
            self.addCleanup(setattr, promote, name, getattr(promote, name))
            setattr(promote, name, value)
        self.reloads = []
        self.addCleanup(setattr, promote, '_reload', promote._reload)
        promote._reload = lambda: self.reloads.append('sighup') or 'sighup'

    def _export(self, run_name):
        export = TMP / 'runs' / run_name / 'weights' / 'best_openvino_model'
        export.mkdir(parents=True, exist_ok=True)
        (export / 'metadata.yaml').write_text('names: {}\n')
        return export

    def _activate_base(self):
        base = self.models / 'yolov8n_base_openvino_model'
        base.mkdir()
        os.symlink(base, promote.ACTIVE_LINK)
        return base

    def test_find_export_walks_run_dir_then_resolves_candidate_link(self):
        export = self._export('overnight_1')
        self.assertEqual(promote.find_export(str(export.parents[1])), str(export))
        with self.assertRaises(FileNotFoundError):
            promote.find_export(str(TMP / 'nowhere'))
        os.symlink(export, promote.CANDIDATE_LINK)
        self.assertEqual(promote.find_export(None), str(export))

    def test_promote_copies_into_durable_slot_and_records_previous(self):
        base = self._activate_base()
        export = self._export('overnight_2')
        durable = promote.promote(str(export.parents[1]))
        self.assertEqual(durable, str(self.models / 'overnight_2_openvino_model'))
        self.assertTrue((Path(durable) / 'metadata.yaml').is_file())
        self.assertEqual(os.path.realpath(promote.ACTIVE_LINK), durable)
        self.assertEqual(Path(promote.PREV_ACTIVE).read_text().strip(), str(base))
        self.assertEqual(self.reloads, ['sighup'])
        # a model that is already active is left alone
        self.assertEqual(promote.promote(durable), durable)
        self.assertEqual(self.reloads, ['sighup'])

    def test_rollback_restores_previous_target(self):
        base = self._activate_base()
        promote.promote(str(self._export('overnight_3').parents[1]))
        self.assertEqual(promote.rollback(), str(base))
        self.assertEqual(os.path.realpath(promote.ACTIVE_LINK), str(base))

    def test_rollback_without_history_raises(self):
        with self.assertRaises(FileNotFoundError):
            promote.rollback()


class QuarantineClassify(unittest.TestCase):
    @staticmethod
    def event(**overrides):
        row = {'event_id': 'e', 'camera': 'backyard', 'start_time': '2026-09-01T12:00:00',
               'end_time': '2026-09-01T12:00:30', 'duration': 30.0, 'detection_count': 30, 'top_label': 'car'}
        row.update(overrides)
        return row

    def test_parked_car_spam_uses_per_camera_ratio(self):
        self.assertIsNone(quarantine.classify(self.event()))                                       # 1 det/s
        self.assertEqual(quarantine.classify(self.event(detection_count=1500)), 'parked_car_spam')  # 50/s
        self.assertIsNone(quarantine.classify(self.event(detection_count=600)))                    # 20/s < 40
        self.assertEqual(quarantine.classify(self.event(camera='front_garage', detection_count=600)),
                         'parked_car_spam')                                                        # 20/s >= 15

    def test_night_person_review(self):
        self.assertEqual(quarantine.classify(self.event(top_label='person', start_time='2026-09-01T03:10:00')),
                         'night_person_review')
        self.assertIsNone(quarantine.classify(self.event(top_label='person')))

    def test_orphans_only_after_grace_period(self):
        old = (datetime.now() - timedelta(minutes=30)).isoformat()
        fresh = (datetime.now() - timedelta(minutes=2)).isoformat()
        self.assertEqual(quarantine.classify(self.event(end_time=None, duration=None, start_time=old)), 'orphan')
        self.assertIsNone(quarantine.classify(self.event(end_time=None, duration=None, start_time=fresh)))

    def test_manifest_and_camera_dir_sit_next_to_db(self):
        self.assertEqual(quarantine.manifest_path('/s/recordings/visionbox.db'),
                         Path('/s/recordings/_quarantine_manifest.jsonl'))
        self.assertEqual(quarantine.cam_out('/s/recordings/visionbox.db', 'front_door'),
                         Path('/s/recordings/front_door'))


class QuarantineRoundTrip(unittest.TestCase):
    def test_apply_then_restore_round_trips_files_and_rows(self):
        rec = TMP / 'recordings'
        (rec / 'front_garage').mkdir(parents=True)
        db_path = rec / 'visionbox.db'
        db = RecordingDatabase(db_path)
        start = datetime(2026, 9, 1, 12, 0, 0)
        for event_id, count in (('spam', 900), ('ok', 30)):
            db.insert_event(event_id, start, camera='front_garage', clean_clip=f'{event_id}.mp4')
            db.update_event_end(event_id, start + timedelta(seconds=30), 30.0,
                                detection_count=count, top_label='car')
            (rec / 'front_garage' / f'{event_id}.mp4').write_bytes(b'x')
        db.close()
        spam_clip = rec / 'front_garage' / 'spam.mp4'

        def flags():
            with sqlite3.connect(db_path) as conn:
                return dict(conn.execute('SELECT event_id, quarantined FROM events').fetchall())

        quarantine.quarantine(str(db_path), apply=False)
        self.assertTrue(spam_clip.exists())
        self.assertEqual(flags(), {'spam': 0, 'ok': 0})

        quarantine.quarantine(str(db_path), apply=True)
        self.assertFalse(spam_clip.exists())
        self.assertTrue((rec / 'front_garage' / '_quarantine' / 'spam.mp4').exists())
        self.assertTrue((rec / 'front_garage' / 'ok.mp4').exists())
        self.assertEqual(flags(), {'spam': 1, 'ok': 0})

        quarantine.restore(str(db_path), reason=None)
        self.assertTrue(spam_clip.exists())
        self.assertEqual(flags(), {'spam': 0, 'ok': 0})


class DedupeCrops(unittest.TestCase):
    def test_conf_of(self):
        self.assertEqual(dedupe.conf_of(Path('track3_20260101_120000_000000_0.87.jpg')), 0.87)
        self.assertEqual(dedupe.conf_of(Path('snapshot.jpg')), -1.0)

    def test_cluster_keeps_best_of_near_identical_crops(self):
        root = TMP / 'crops'
        folder = root / 'front_door' / 'car'
        folder.mkdir(parents=True)
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        cv2.imwrite(str(folder / 'track1_a_0.60.png'), gradient)
        cv2.imwrite(str(folder / 'track1_b_0.90.png'), gradient)
        cv2.imwrite(str(folder / 'track2_c_0.50.png'), np.zeros((64, 64), dtype=np.uint8))
        (folder / 'track9_broken_0.99.png').write_bytes(b'not an image')

        self.assertEqual(dedupe.dhash(folder / 'track1_a_0.60.png'), dedupe.dhash(folder / 'track1_b_0.90.png'))
        keep, drop = dedupe.cluster(folder, threshold=6)
        self.assertEqual(sorted(p.name for p in keep),
                         ['track1_b_0.90.png', 'track2_c_0.50.png', 'track9_broken_0.99.png'])
        self.assertEqual([p.name for p in drop], ['track1_a_0.60.png'])

        dedupe.run(root, 6, apply=False)   # dry-run moves nothing
        self.assertFalse((root / '_trash').exists())
        self.assertFalse(dedupe.manifest_path(root).exists())


if __name__ == '__main__':
    unittest.main()
