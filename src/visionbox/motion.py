"""
Motion detection using background subtraction.

How it works:
1. MOG2 learns what the "background" looks like over time
2. Each new frame is compared to the background model
3. Pixels that differ significantly = "foreground" (motion)
4. We find contours around motion regions
5. Only run expensive detection where motion occurs

This is how Frigate and other smart cameras avoid burning CPU on static scenes.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MotionRegion:
    """A detected region of motion."""
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class MotionDetector:
    """
    Detects motion using MOG2 background subtraction.

    MOG2 = Mixture of Gaussians (2 components per pixel)
    - Each pixel is modeled as a probability distribution
    - "Background" = pixels that match the learned distribution
    - "Foreground" = pixels that don't match (something moved)

    The model adapts over time, so gradual changes (lighting) are absorbed
    into the background, but sudden changes (person walking) trigger motion.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        min_area: int = 500,
        learning_rate: float = -1,
    ):
        """
        Args:
            history: How many frames to use for background model.
                     More = slower to adapt, but more stable.
            var_threshold: How different a pixel must be to count as foreground.
                          Lower = more sensitive to small changes.
            detect_shadows: Whether to detect shadows separately (slower).
            min_area: Minimum contour area to count as motion (filters noise).
            learning_rate: How fast to update background. -1 = auto.
                          0 = never update, 1 = instant update.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.learning_rate = learning_rate

        # Morphological kernel for cleaning up the mask
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[MotionRegion]:
        """
        Detect motion regions in frame.

        Returns list of MotionRegion objects, or empty list if no motion.
        """
        # Apply background subtraction
        # This returns a mask: 255 = foreground (motion), 0 = background
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Clean up the mask with morphological operations:
        # - erode: removes small noise (isolated pixels)
        # - dilate: fills gaps and connects nearby regions
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Find contours (outlines of motion regions)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area and convert to MotionRegion
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append(MotionRegion(x=x, y=y, w=w, h=h, area=int(area)))

        return regions

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get the raw foreground mask (for visualization)."""
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)
        return fg_mask

    def reset(self):
        """Reset the background model (start fresh)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16.0,
            detectShadows=False,
        )


def merge_overlapping_regions(
    regions: list[MotionRegion],
    padding: int = 20
) -> list[tuple[int, int, int, int]]:
    """
    Merge overlapping motion regions into larger bounding boxes.

    This is useful because:
    - A person walking might create multiple small motion regions
    - We want to run detection on ONE area covering all of them
    - Padding ensures we don't crop too tightly

    Returns list of (x1, y1, x2, y2) boxes.
    """
    if not regions:
        return []

    # Convert to boxes with padding
    boxes = []
    for r in regions:
        x1 = max(0, r.x - padding)
        y1 = max(0, r.y - padding)
        x2 = r.x + r.w + padding
        y2 = r.y + r.h + padding
        boxes.append([x1, y1, x2, y2])

    # Simple merge: combine all overlapping boxes
    # This is greedy but works well for motion regions
    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue

        # Start with this box
        x1, y1, x2, y2 = box
        used[i] = True

        # Keep merging until no more overlaps
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(boxes):
                if used[j]:
                    continue

                ox1, oy1, ox2, oy2 = other

                # Check overlap
                if not (x2 < ox1 or ox2 < x1 or y2 < oy1 or oy2 < y1):
                    # Merge: expand to cover both
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    used[j] = True
                    changed = True

        merged.append((x1, y1, x2, y2))

    return merged
