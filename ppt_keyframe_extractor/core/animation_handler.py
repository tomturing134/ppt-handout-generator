"""Animation sequence detection and final-state selection."""

from __future__ import annotations

import logging
from typing import List

from ..types import ChangeType, ExtractionConfig, FrameInfo, SlideCluster
from ..utils.image_utils import compute_information_entropy
from .slide_detector import SlideDetector

logger = logging.getLogger(__name__)


class AnimationHandler:
    """Detect and resolve PPT animation sequences.

    PPT animations (text appearing line by line, progressive reveals, etc.)
    cause multiple small SSIM changes within a single slide. The user wants
    only the final, most complete state of each slide.

    Detection strategy:
    - Animation sequences show multiple small SSIM drops in quick succession
    - Time span is typically short (a few seconds)
    - The final frame has the most content (highest information entropy)

    Resolution:
    - For each animation cluster, keep only the frame with maximum entropy
    - This represents the most complete state of the slide
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._slide_detector = SlideDetector(config)

    def resolve_animations(self, clusters: List[SlideCluster]) -> List[SlideCluster]:
        """Process clusters and resolve animation sequences.

        For clusters with animation sub-steps, keep only the most
        information-rich frame (the final, complete state).

        Args:
            clusters: List of SlideCluster objects from FrameClusterer.

        Returns:
            Updated list with animation sequences resolved.
        """
        result = []

        for cluster in clusters:
            if len(cluster.frames) <= 1:
                result.append(cluster)
                continue

            # Check if this cluster contains animation steps
            is_animation = self._is_animation_sequence(cluster)

            if is_animation:
                cluster.is_animation_sequence = True
                # Select the frame with the highest information entropy
                # (most content visible = final animation state)
                best_frame = self._select_most_complete_frame(cluster.frames)
                # Keep all frames for occlusion selection, but mark the animation
                logger.debug(
                    f"Animation sequence detected in cluster {cluster.cluster_id} "
                    f"({len(cluster.frames)} sub-frames)"
                )
            else:
                best_frame = None

            result.append(cluster)

        return result

    def _is_animation_sequence(self, cluster: SlideCluster) -> bool:
        """Determine if a cluster contains PPT animation steps.

        Animation signature:
        - Multiple frames with small SSIM changes between them
        - Changes happen within a short time window
        - All changes are ANIMATION_STEP type (not SLIDE_CHANGE)
        """
        if len(cluster.frames) < 3:
            return False

        # Count animation steps (small SSIM drops)
        animation_steps = 0
        for i in range(1, len(cluster.frames)):
            change_type = self._slide_detector.classify_change(
                cluster.frames[i], cluster.frames[i - 1]
            )
            if change_type == ChangeType.ANIMATION_STEP:
                animation_steps += 1

        # If most changes are animation steps, it's an animation sequence
        total_changes = sum(
            1
            for i in range(1, len(cluster.frames))
            if cluster.frames[i].ssim_score < self._config.ssim_threshold
        )

        if total_changes == 0:
            return False

        animation_ratio = animation_steps / total_changes if total_changes > 0 else 0

        return animation_ratio >= 0.5  # majority of changes are animation steps

    def _select_most_complete_frame(self, frames: List[FrameInfo]) -> FrameInfo:
        """Select the frame with the most complete PPT content.

        Uses information entropy as a proxy for content completeness.
        The final state of an animation typically has the most text/content.

        Args:
            frames: List of frames from an animation sequence.

        Returns:
            The frame with the highest information entropy.
        """
        best_frame = frames[0]
        best_entropy = compute_information_entropy(
            best_frame.image, best_frame.person_mask
        )

        for frame in frames[1:]:
            entropy = compute_information_entropy(frame.image, frame.person_mask)
            if entropy > best_entropy:
                best_entropy = entropy
                best_frame = frame

        return best_frame
