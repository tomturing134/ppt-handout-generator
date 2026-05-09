"""Frame clustering - group frames belonging to the same PPT slide."""

from __future__ import annotations

import logging
from typing import List

from ..types import ChangeType, ExtractionConfig, FrameInfo, SlideCluster
from .slide_detector import SlideDetector

logger = logging.getLogger(__name__)


class FrameClusterer:
    """Group frames into clusters, where each cluster represents one PPT slide.

    Uses the SlideDetector to determine when a slide change occurs.
    Frames between two slide changes belong to the same cluster.
    Animation steps create sub-clusters that are later merged by AnimationHandler.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._slide_detector = SlideDetector(config)

    def cluster(self, frames: List[FrameInfo]) -> List[SlideCluster]:
        """Group frames into slide clusters.

        Args:
            frames: List of FrameInfo objects from sampling phase.

        Returns:
            List of SlideCluster objects.
        """
        if not frames:
            return []

        clusters: List[SlideCluster] = []
        current_cluster = SlideCluster(
            cluster_id=0,
            frames=[frames[0]],
            start_time=frames[0].timestamp,
            end_time=frames[0].timestamp,
        )

        for i in range(1, len(frames)):
            change_type = self._slide_detector.classify_change(frames[i], frames[i - 1])

            if change_type == ChangeType.NO_CHANGE:
                # Same slide, add to current cluster
                current_cluster.frames.append(frames[i])
                current_cluster.end_time = frames[i].timestamp
            elif change_type == ChangeType.ANIMATION_STEP:
                # Animation step - still same slide but content changed
                # Keep in same cluster, will be resolved by AnimationHandler
                current_cluster.frames.append(frames[i])
                current_cluster.end_time = frames[i].timestamp
            else:
                # SLIDE_CHANGE - new slide, finalize current cluster
                clusters.append(current_cluster)

                # Start new cluster
                current_cluster = SlideCluster(
                    cluster_id=len(clusters),
                    frames=[frames[i]],
                    start_time=frames[i].timestamp,
                    end_time=frames[i].timestamp,
                )

        # Don't forget the last cluster
        clusters.append(current_cluster)

        logger.info(f"Clustered {len(frames)} frames into {len(clusters)} slides")
        return clusters
