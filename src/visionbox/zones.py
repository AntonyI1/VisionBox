"""Detection zones — Frigate-style motion masks and required zones.

Exclude zones act as motion masks: motion in these areas is suppressed
so YOLO never runs there. Include zones are required zones: detections
must fall inside one to trigger recording.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    type: str  # 'include' or 'exclude'
    points: list[list[float]]  # normalized 0-1 coordinates [[x,y], ...]


class ZoneFilter:
    def __init__(self, path: str = 'zones.json'):
        self._path = Path(path)
        self._zones: list[Zone] = []
        self._lock = threading.Lock()
        self._cache_resolution: tuple[int, int] | None = None
        self._cache_contours: dict[str, np.ndarray] = {}
        self.load()

    def load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._zones = [Zone(**z) for z in data]
            self._invalidate_cache()
        except (json.JSONDecodeError, TypeError, KeyError):
            self._zones = []

    def save(self):
        data = [{'name': z.name, 'type': z.type, 'points': z.points} for z in self._zones]
        self._path.write_text(json.dumps(data, indent=2))

    def add_zone(self, zone: Zone):
        with self._lock:
            self._zones = [z for z in self._zones if z.name != zone.name]
            self._zones.append(zone)
            self._invalidate_cache()
            self.save()

    def remove_zone(self, name: str) -> bool:
        with self._lock:
            before = len(self._zones)
            self._zones = [z for z in self._zones if z.name != name]
            if len(self._zones) < before:
                self._invalidate_cache()
                self.save()
                return True
            return False

    def get_zones(self) -> list[dict]:
        return [{'name': z.name, 'type': z.type, 'points': z.points} for z in self._zones]

    @property
    def has_exclude(self) -> bool:
        return any(z.type == 'exclude' for z in self._zones)

    @property
    def has_include(self) -> bool:
        return any(z.type == 'include' for z in self._zones)

    def filter_motion_regions(
        self, regions: list[tuple], frame_shape: tuple
    ) -> list[tuple]:
        """Motion mask — drop motion regions whose center falls in an exclude zone."""
        if not self.has_exclude:
            return regions

        h, w = frame_shape[:2]
        contours = self._get_contours(w, h)
        excludes = [contours[z.name] for z in self._zones
                    if z.type == 'exclude' and z.name in contours]

        filtered = []
        for region in regions:
            x1, y1, x2, y2 = region[:4]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            if any(cv2.pointPolygonTest(c, (cx, cy), False) >= 0 for c in excludes):
                continue
            filtered.append(region)
        return filtered

    def filter_detections(
        self, detections: list[dict], frame_shape: tuple
    ) -> list[dict]:
        """Drop detections whose center falls in an exclude zone."""
        if not self.has_exclude:
            return detections

        h, w = frame_shape[:2]
        contours = self._get_contours(w, h)
        excludes = [contours[z.name] for z in self._zones
                    if z.type == 'exclude' and z.name in contours]

        filtered = []
        for det in detections:
            box = det['box']
            cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
            if any(cv2.pointPolygonTest(c, (cx, cy), False) >= 0 for c in excludes):
                continue
            filtered.append(det)
        return filtered

    def check_required_zones(
        self, detections: list[dict], frame_shape: tuple
    ) -> bool:
        """Required zones — return True if any detection is inside an include zone.

        If no include zones are defined, returns True (no restriction).
        """
        if not self.has_include:
            return True
        if not detections:
            return False

        h, w = frame_shape[:2]
        contours = self._get_contours(w, h)
        includes = [contours[z.name] for z in self._zones
                    if z.type == 'include' and z.name in contours]

        for det in detections:
            box = det['box']
            cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
            if any(cv2.pointPolygonTest(c, (cx, cy), False) >= 0 for c in includes):
                return True
        return False

    def get_pixel_contours(self, w: int, h: int) -> list[tuple[str, str, np.ndarray]]:
        contours = self._get_contours(w, h)
        return [(z.name, z.type, contours[z.name]) for z in self._zones if z.name in contours]

    def _get_contours(self, w: int, h: int) -> dict[str, np.ndarray]:
        res = (w, h)
        if self._cache_resolution == res and self._cache_contours:
            return self._cache_contours

        contours = {}
        for z in self._zones:
            pts = np.array([[p[0] * w, p[1] * h] for p in z.points], dtype=np.float32)
            contours[z.name] = pts
        self._cache_resolution = res
        self._cache_contours = contours
        return contours

    def _invalidate_cache(self):
        self._cache_resolution = None
        self._cache_contours = {}
