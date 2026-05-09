"""Keyframe selection - pick the best frame for each slide cluster."""

from __future__ import annotations

import logging
from typing import List

from ..types import ExtractionConfig, FrameInfo, SlideCluster
from ..utils.image_utils import compute_masked_ssim

logger = logging.getLogger(__name__)


class KeyframeSelector:
    """Select the best keyframe for each slide cluster.

    For each cluster:
    1. Select the frame with minimal occlusion (handled by OcclusionScorer)
    2. For animation sequences, prefer the most complete state

    After selection:
    3. Final deduplication: remove near-duplicate keyframes across clusters
       (can happen if the same slide appears at different times)
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config

    def select_keyframes(self, clusters: List[SlideCluster]) -> List[FrameInfo]:
        """Select one keyframe per cluster and deduplicate.

        Args:
            clusters: List of SlideCluster with best_frame already set.

        Returns:
            List of FrameInfo objects (one per unique slide).
        """
        keyframes = []
        for cluster in clusters:
            if cluster.best_frame is not None:
                keyframes.append(cluster.best_frame)
            elif cluster.frames:
                # Fallback: use the last frame in the cluster
                keyframes.append(cluster.frames[-1])

        # Final deduplication: remove near-duplicate keyframes
        keyframes = self._deduplicate(keyframes)

        logger.info(f"Selected {len(keyframes)} keyframes from {len(clusters)} clusters")
        return keyframes

    def _deduplicate(self, keyframes: List[FrameInfo]) -> List[FrameInfo]:
        """Remove near-duplicate keyframes using SSIM comparison.

        If two consecutive keyframes have SSIM > 0.98, they are likely
        the same slide appearing at different times. Keep the one with
        lower occlusion.

        Args:
            keyframes: List of keyframes to deduplicate.

        Returns:
            Deduplicated list of keyframes.
        """
        if len(keyframes) <= 1:
            return keyframes

        dedup_threshold = 0.92
        result = [keyframes[0]]

        for i in range(1, len(keyframes)):
            # Compare with the last kept keyframe
            ssim_score = compute_masked_ssim(
                keyframes[i].image,
                result[-1].image,
                keyframes[i].person_mask,
                result[-1].person_mask,
            )

            if ssim_score > dedup_threshold:
                # Near-duplicate: keep the one with lower occlusion
                if keyframes[i].occlusion_score < result[-1].occlusion_score:
                    result[-1] = keyframes[i]
                    logger.debug(
                        f"Dedup: replaced keyframe at {result[-1].timestamp:.1f}s "
                        f"with less occluded one at {keyframes[i].timestamp:.1f}s"
                    )
            else:
                result.append(keyframes[i])

        removed = len(keyframes) - len(result)
        if removed > 0:
            logger.info(f"Deduplication: removed {removed} duplicate keyframes")

        return result
