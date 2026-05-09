"""Main extraction pipeline orchestrator."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..types import ExtractionConfig, ExtractionResult, FrameInfo, SlideCluster
from .animation_handler import AnimationHandler
from .frame_clusterer import FrameClusterer
from .frame_sampler import AdaptiveFrameSampler
from .keyframe_selector import KeyframeSelector
from .occlusion_scorer import OcclusionScorer
from .person_detector import PersonDetectorFactory
from ..utils.pdf_builder import build_pdf, save_keyframe_images
from ..utils.video_io import get_video_info

logger = logging.getLogger(__name__)


class LecturePPTExtractor:
    """Main entry point for extracting PPT keyframes from lecture videos.

    This class orchestrates the full 5-phase pipeline:
    1. Adaptive sampling (double-jump with person detection)
    2. PPT change detection and frame clustering
    3. Animation resolution (keep only final state)
    4. Keyframe selection (min occlusion per cluster)
    5. Output generation (images + PDF)

    Usage:
        extractor = LecturePPTExtractor()
        result = extractor.extract("lecture.mp4", output_dir="./output")
        print(f"Extracted {result.slide_count} slides")
        print(f"PDF saved to {result.pdf_path}")
    """

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self._config = config or ExtractionConfig()

        # Initialize components
        self._person_detector = PersonDetectorFactory.create(self._config)
        self._sampler = AdaptiveFrameSampler(self._config, self._person_detector)
        self._clusterer = FrameClusterer(self._config)
        self._animation_handler = AnimationHandler(self._config)
        self._occlusion_scorer = OcclusionScorer(self._config)
        self._keyframe_selector = KeyframeSelector(self._config)

    def extract(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract PPT keyframes from a lecture video.

        Args:
            video_path: Path to the lecture video file.
            output_dir: Directory for output files. Defaults to config.output_dir.

        Returns:
            ExtractionResult with keyframes, PDF path, and warnings.
        """
        output_dir = output_dir or self._config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting extraction: {video_path}")

        # Get video info
        fps, total_frames, duration, width, height = get_video_info(video_path)
        logger.info(
            f"Video info: {width}x{height} @ {fps:.1f}fps, "
            f"{duration:.1f}s ({total_frames} frames)"
        )

        # Phase 1: Adaptive sampling with person detection
        logger.info("Phase 1: Adaptive frame sampling...")
        frames = self._sampler.sample(video_path, show_progress=True)

        if not frames:
            logger.warning("No frames extracted from video")
            return ExtractionResult()

        # Phase 2: Frame clustering
        logger.info("Phase 2: Clustering frames by slide...")
        clusters = self._clusterer.cluster(frames)

        # Phase 3: Animation resolution
        logger.info("Phase 3: Resolving animations...")
        clusters = self._animation_handler.resolve_animations(clusters)

        # Phase 4: Keyframe selection (occlusion-based)
        logger.info("Phase 4: Selecting keyframes (minimal occlusion)...")
        for cluster in clusters:
            if cluster.frames:
                cluster.best_frame = self._occlusion_scorer.find_least_occluded(cluster)

        # Check for high-occlusion warnings
        warnings = self._occlusion_scorer.check_warnings(clusters)

        # Select final keyframes with deduplication
        keyframes = self._keyframe_selector.select_keyframes(clusters)

        # Phase 5: Output generation
        logger.info("Phase 5: Generating output...")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        images_dir = os.path.join(output_dir, video_name)

        image_paths = save_keyframe_images(
            keyframes,
            images_dir,
            image_format=self._config.image_format,
            quality=self._config.image_quality,
        )

        pdf_path = os.path.join(output_dir, f"{video_name}.pdf")
        build_pdf(image_paths, pdf_path)

        result = ExtractionResult(
            keyframes=keyframes,
            pdf_path=pdf_path,
            image_paths=image_paths,
            warnings=warnings,
            slide_count=len(keyframes),
        )

        logger.info(
            f"Extraction complete: {result.slide_count} slides, "
            f"PDF: {result.pdf_path}"
        )
        if warnings:
            logger.warning(f"{len(warnings)} slides have high occlusion")

        return result

    def extract_frames(self, video_path: str) -> List[SlideCluster]:
        """Run the pipeline but return cluster data without writing files.

        Useful for programmatic access to frame data.

        Args:
            video_path: Path to the lecture video file.

        Returns:
            List of SlideCluster with best_frame selected.
        """
        # Phase 1: Sampling
        frames = self._sampler.sample(video_path, show_progress=False)
        if not frames:
            return []

        # Phase 2: Clustering
        clusters = self._clusterer.cluster(frames)

        # Phase 3: Animation resolution
        clusters = self._animation_handler.resolve_animations(clusters)

        # Phase 4: Keyframe selection
        for cluster in clusters:
            if cluster.frames:
                cluster.best_frame = self._occlusion_scorer.find_least_occluded(cluster)

        return clusters
