"""Occlusion scoring - quantify how much of the PPT is blocked by the teacher."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..types import ExtractionConfig, FrameInfo, SlideCluster
from ..utils.image_utils import compute_occlusion_score

logger = logging.getLogger(__name__)


class OcclusionScorer:
    """Score and rank frames by how much the teacher occludes the PPT.

    For each frame, the occlusion score is:
        occlusion_score = (person_mask ∩ ppt_region) / ppt_region_area

    Lower score = less occlusion = better frame.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config

    def score_frame(self, frame: FrameInfo) -> float:
        """Compute occlusion score for a single frame.

        Args:
            frame: FrameInfo with person_mask already computed.

        Returns:
            Occlusion score between 0.0 and 1.0.
        """
        return compute_occlusion_score(frame.person_mask, self._config.ppt_region)

    def score_all(self, frames: List[FrameInfo]) -> None:
        """Compute and update occlusion scores for all frames.

        Modifies frames in-place by updating their occlusion_score field.

        Args:
            frames: List of FrameInfo objects.
        """
        for frame in frames:
            frame.occlusion_score = self.score_frame(frame)

    def find_least_occluded(self, cluster: SlideCluster) -> FrameInfo:
        """Find the least occluded frame in a cluster.

        For animation sequences, also considers information richness:
        prefer the frame with lowest occlusion among the most recent
        (final) animation sub-frames.

        Args:
            cluster: SlideCluster to search.

        Returns:
            The FrameInfo with the lowest occlusion score.
        """
        if not cluster.frames:
            raise ValueError(f"Cluster {cluster.cluster_id} has no frames")

        # For animation sequences, prefer later frames (more complete content)
        # among those with low occlusion
        if cluster.is_animation_sequence and len(cluster.frames) > 1:
            # Take the latter half of frames (likely the complete state)
            midpoint = len(cluster.frames) // 2
            candidates = cluster.frames[midpoint:]
        else:
            candidates = cluster.frames

        # Select the frame with the lowest occlusion
        best = min(candidates, key=lambda f: f.occlusion_score)
        return best

    def check_warnings(self, clusters: List[SlideCluster]) -> List[str]:
        """Check for high-occlusion slides and generate warnings.

        Args:
            clusters: List of SlideCluster with best_frame already selected.

        Returns:
            List of warning messages for heavily occluded slides.
        """
        warnings = []
        for cluster in clusters:
            if cluster.best_frame and cluster.best_frame.occlusion_score > self._config.max_occlusion_threshold:
                warnings.append(
                    f"Slide {cluster.cluster_id + 1} "
                    f"(t={cluster.best_frame.timestamp:.1f}s): "
                    f"best frame is {cluster.best_frame.occlusion_score:.0%} occluded"
                )
        return warnings
