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
        self._path.write_text(json.dumps(self.get_zones(), indent=2))

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
        excludes = self._contours_of('exclude', frame_shape)
        return [r for r in regions if not self._center_in_any(r[:4], excludes)]

    def filter_detections(
        self, detections: list[dict], frame_shape: tuple
    ) -> list[dict]:
        """Drop detections whose center falls in an exclude zone."""
        if not self.has_exclude:
            return detections
        excludes = self._contours_of('exclude', frame_shape)
        return [d for d in detections if not self._center_in_any(d['box'], excludes)]

    def check_required_zones(
        self, detections: list[dict], frame_shape: tuple
    ) -> bool:
        """Required zones — return True if any detection is inside an include zone.

        If no include zones are defined, returns True (no restriction).
        """
        if not self.has_include:
            return True
        includes = self._contours_of('include', frame_shape)
        return any(self._center_in_any(d['box'], includes) for d in detections)

    def get_pixel_contours(self, w: int, h: int) -> list[tuple[str, str, np.ndarray]]:
        contours = self._get_contours(w, h)
        return [(z.name, z.type, contours[z.name]) for z in self._zones if z.name in contours]

    def _contours_of(self, zone_type: str, frame_shape: tuple) -> list[np.ndarray]:
        h, w = frame_shape[:2]
        contours = self._get_contours(w, h)
        return [contours[z.name] for z in self._zones if z.type == zone_type and z.name in contours]

    @staticmethod
    def _center_in_any(box, contours: list[np.ndarray]) -> bool:
        cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
        return any(cv2.pointPolygonTest(c, (cx, cy), False) >= 0 for c in contours)

    def _get_contours(self, w: int, h: int) -> dict[str, np.ndarray]:
        if self._cache_resolution == (w, h):
            return self._cache_contours
        self._cache_contours = {
            z.name: np.array([[p[0] * w, p[1] * h] for p in z.points], dtype=np.float32)
            for z in self._zones
        }
        self._cache_resolution = (w, h)
        return self._cache_contours

    def _invalidate_cache(self):
        self._cache_resolution = None
        self._cache_contours = {}
