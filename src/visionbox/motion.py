"""Motion detection via MOG2 background subtraction."""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MotionRegion:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class MotionDetector:
    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        min_area: int = 500,
        learning_rate: float = -1,
        min_area_frac: float = 0.0,
    ):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.min_area_frac = min_area_frac
        self.learning_rate = learning_rate
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Set by detect(): full-frame foreground fraction and the min across the 4 quadrants.
        # Used by the caller's whole-frame illumination guard.
        self.last_coverage = 0.0
        self.quadrant_min = 0.0

    def relearn(self, frame: np.ndarray, rate: float = -1):
        """Fold one frame into the background model without returning regions."""
        self.bg_subtractor.apply(frame, learningRate=rate)

    def detect(self, frame: np.ndarray) -> list[MotionRegion]:
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        h_m, w_m = fg_mask.shape[:2]
        self.last_coverage = float(cv2.countNonZero(fg_mask)) / float(h_m * w_m)
        hh, hw = h_m // 2, w_m // 2
        quads = [fg_mask[:hh, :hw], fg_mask[:hh, hw:], fg_mask[hh:, :hw], fg_mask[hh:, hw:]]
        self.quadrant_min = min(cv2.countNonZero(q) / float(max(1, q.size)) for q in quads)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area
        if self.min_area_frac > 0:
            min_area = max(min_area, self.min_area_frac * h_m * w_m)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append(MotionRegion(x=x, y=y, w=w, h=h, area=int(area)))

        return regions

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        return fg_mask

    def reset(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16.0, detectShadows=False,
        )


def merge_overlapping_regions(
    regions: list[MotionRegion],
    padding: int = 20
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping motion regions into larger bounding boxes."""
    if not regions:
        return []

    boxes = []
    for r in regions:
        boxes.append([
            max(0, r.x - padding), max(0, r.y - padding),
            r.x + r.w + padding, r.y + r.h + padding,
        ])

    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue

        x1, y1, x2, y2 = box
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(boxes):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2 = other
                if not (x2 < ox1 or ox2 < x1 or y2 < oy1 or oy2 < y1):
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    used[j] = True
                    changed = True

        merged.append((x1, y1, x2, y2))

    return merged
