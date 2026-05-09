"""PPT slide change detection using masked SSIM."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..types import ChangeType, ExtractionConfig, FrameInfo
from ..utils.image_utils import compute_masked_ssim

logger = logging.getLogger(__name__)


class SlideDetector:
    """Detect PPT slide changes using SSIM on masked frames.

    The key innovation: by computing SSIM on frames with teacher pixels
    zeroed out, we can detect PPT changes even when the teacher is
    moving in front of the slide.

    Change classification:
    - NO_CHANGE: SSIM > ssim_threshold (same slide)
    - ANIMATION_STEP: animation_ssim_threshold < SSIM <= ssim_threshold
      (small change, likely PPT animation like text appearing)
    - SLIDE_CHANGE: SSIM <= animation_ssim_threshold
      (large change, likely a completely different slide)
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config

    def classify_change(self, current: FrameInfo, previous: FrameInfo) -> ChangeType:
        """Classify the type of change between two frames.

        Args:
            current: Current frame info.
            previous: Previous frame info.

        Returns:
            ChangeType indicating the nature of the change.
        """
        ssim_score = current.ssim_score

        if ssim_score >= self._config.ssim_threshold:
            return ChangeType.NO_CHANGE
        elif ssim_score >= self._config.animation_ssim_threshold:
            return ChangeType.ANIMATION_STEP
        else:
            return ChangeType.SLIDE_CHANGE

    def compute_ssim(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask1: Optional[np.ndarray] = None,
        mask2: Optional[np.ndarray] = None,
    ) -> float:
        """Compute masked SSIM between two frames.

        Args:
            frame1: First BGR image.
            frame2: Second BGR image.
            mask1: Person mask for frame1.
            mask2: Person mask for frame2.

        Returns:
            SSIM score between 0 and 1.
        """
        return compute_masked_ssim(frame1, frame2, mask1, mask2)

    def is_slide_change(self, current: FrameInfo, previous: FrameInfo) -> bool:
        """Check if a real slide change occurred (not just animation).

        Args:
            current: Current frame info.
            previous: Previous frame info.

        Returns:
            True if this is a real slide change.
        """
        change_type = self.classify_change(current, previous)
        return change_type == ChangeType.SLIDE_CHANGE
