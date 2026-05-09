"""OpenCV MOG2 background subtraction fallback for person detection."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MOG2Detector:
    """OpenCV MOG2 background subtraction for person detection.

    This is a fallback for environments without PyTorch/ultralytics.
    It uses background subtraction to detect foreground (likely the teacher)
    and produces a binary mask.

    Limitations compared to YOLO:
    - Less accurate mask boundaries
    - Sensitive to lighting changes
    - Requires a learning period to model the background
    - Cannot distinguish "person" from other moving objects
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        min_area_ratio: float = 0.01,
        morph_kernel_size: int = 5,
    ) -> None:
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._min_area_ratio = min_area_ratio
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._frame_count = 0
        self._learning_frames = max(history // 5, 30)  # warm-up period

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect foreground (person) in a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Binary mask (H, W) where 1=foreground/person, or None if no person.
        """
        h, w = frame.shape[:2]

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(frame)

        # During warm-up, always apply with high learning rate
        if self._frame_count < self._learning_frames:
            self._bg_subtractor.apply(frame, learningRate=0.1)
        self._frame_count += 1

        # Shadow values are 127 in MOG2, threshold them out
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        fg_mask = cv2.dilate(fg_mask, self._morph_kernel, iterations=2)

        # Check if foreground area is significant enough
        fg_pixels = np.count_nonzero(fg_mask)
        total_pixels = h * w
        area_ratio = fg_pixels / total_pixels

        if area_ratio < self._min_area_ratio:
            return None

        # Normalize to 0/1
        mask = (fg_mask > 0).astype(np.uint8)
        return mask

    def reset(self) -> None:
        """Reset the background model."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self._frame_count = 0
