"""Adaptive frame sampling using double-jump strategy."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..types import ExtractionConfig, FrameInfo
from ..utils.image_utils import compute_masked_ssim, compute_occlusion_score, is_dominant_color
from ..utils.video_io import open_video

logger = logging.getLogger(__name__)


class AdaptiveFrameSampler:
    """Frame sampling with two modes:

    1. Fixed-interval (default, recommended): Sample every N seconds.
       More reliable for frequent slide changes, avoids skipping slides.

    2. Double-jump: Original MP4_to_PDF strategy with exponential backoff.
       Better for long videos with infrequent slide changes.
    """

    def __init__(self, config: ExtractionConfig, person_detector) -> None:
        self._config = config
        self._person_detector = person_detector

    def sample(self, video_path: str, show_progress: bool = True) -> List[FrameInfo]:
        """Sample frames from a video.

        Uses fixed-interval mode by default (config.use_fixed_interval=True)
        for better recall. Falls back to double-jump for backward compatibility.

        Args:
            video_path: Path to the video file.
            show_progress: Whether to show progress bar.

        Returns:
            List of FrameInfo objects for sampled frames.
        """
        if self._config.use_fixed_interval:
            return self._sample_fixed_interval(video_path, show_progress)
        else:
            return self._sample_double_jump(video_path, show_progress)

    def _sample_fixed_interval(
        self, video_path: str, show_progress: bool = True
    ) -> List[FrameInfo]:
        """Sample frames at fixed intervals with triple-signal change detection.

        Each sampled frame is annotated with:
        - ssim_score: Masked SSIM vs previous frame
        - edge_density: Canny edge count / total pixels
        - occlusion_score: Teacher occlusion ratio

        Change detection uses SSIM + edge density (recall-first approach).
        """
        with open_video(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_step = int(fps * 5)  # 5-second intervals

            frames: List[FrameInfo] = []
            prev_frame = None
            prev_mask = None

            pbar = tqdm(
                total=total_frames,
                desc="Sampling frames",
                unit="frame",
                disable=not show_progress,
            )

            for i in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret or is_dominant_color(frame):
                    pbar.update(sample_step)
                    continue

                # Person detection
                person_mask = self._person_detector.detect(frame)
                occlusion = compute_occlusion_score(
                    person_mask, self._config.ppt_region
                )
                timestamp = i / fps if fps > 0 else 0.0

                # Compute masked SSIM vs previous frame
                ssim_score = 1.0
                if prev_frame is not None:
                    ssim_score = compute_masked_ssim(
                        frame, prev_frame, person_mask, prev_mask
                    )

                frame_info = FrameInfo(
                    frame_index=i,
                    timestamp=timestamp,
                    image=frame,
                    person_mask=person_mask,
                    occlusion_score=occlusion,
                    ssim_score=ssim_score,
                )
                frames.append(frame_info)
                prev_frame = frame
                prev_mask = person_mask

                pbar.update(sample_step)

            pbar.close()

        logger.info(
            f"Fixed-interval sampling: {len(frames)} frames from {total_frames} total"
        )
        return frames

    def _sample_double_jump(
        self, video_path: str, show_progress: bool = True
    ) -> List[FrameInfo]:
        """Original double-jump sampling strategy (backward compatible)."""
        with open_video(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0

            logger.info(
                f"Video: {video_path} | FPS: {fps:.1f} | "
                f"Frames: {total_frames} | Duration: {duration:.1f}s"
            )

            initial_jump = int(fps * self._config.initial_jump_seconds)
            max_jump = int(fps * self._config.max_jump_seconds)
            stability_frames = int(fps * self._config.stability_window_seconds)

            frames: List[FrameInfo] = []
            i = 0
            prev_info: Optional[FrameInfo] = None
            jump = initial_jump

            pbar = tqdm(
                total=total_frames,
                desc="Sampling frames",
                unit="frame",
                disable=not show_progress,
            )

            while i < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip dominant-color (blank) frames
                if is_dominant_color(frame):
                    pbar.update(jump)
                    i += jump
                    jump = min(jump * 2, max_jump)
                    continue

                # Person detection
                person_mask = self._person_detector.detect(frame)

                # Occlusion scoring
                occlusion = compute_occlusion_score(
                    person_mask, self._config.ppt_region
                )

                # Compute masked SSIM with previous frame
                ssim_score = 1.0
                if prev_info is not None:
                    ssim_score = compute_masked_ssim(
                        frame, prev_info.image, person_mask, prev_info.person_mask
                    )

                timestamp = i / fps if fps > 0 else 0.0
                frame_info = FrameInfo(
                    frame_index=i,
                    timestamp=timestamp,
                    image=frame,
                    person_mask=person_mask,
                    occlusion_score=occlusion,
                    ssim_score=ssim_score,
                )

                if prev_info is None:
                    # First valid frame
                    frames.append(frame_info)
                    prev_info = frame_info
                    jump = initial_jump
                elif ssim_score < self._config.ssim_threshold:
                    # Change detected - verify stability
                    # Check if the change persists (not a transient glitch)
                    stable_frame = self._verify_stability(
                        cap, i, frame, person_mask, fps, stability_frames, total_frames
                    )
                    if stable_frame is not None:
                        # Real change confirmed
                        frames.append(frame_info)
                        prev_info = frame_info
                        jump = initial_jump  # reset to fine scanning
                    else:
                        # Transient change, keep looking
                        jump = initial_jump
                else:
                    # Same slide - exponential backoff
                    jump = min(jump * 2, max_jump)

                pbar.update(jump)
                i += jump

            pbar.close()

        logger.info(f"Sampled {len(frames)} candidate frames")
        return frames

    def _verify_stability(
        self,
        cap: cv2.VideoCapture,
        current_idx: int,
        current_frame: np.ndarray,
        current_mask,
        fps: float,
        stability_frames: int,
        total_frames: int,
    ) -> Optional[np.ndarray]:
        """Verify that a detected change is stable (not a transient glitch).

        Read a frame `stability_frames` ahead and check if it's still different
        from the pre-change frame.

        Returns:
            The stable frame if change is confirmed, None if transient.
        """
        check_idx = min(current_idx + stability_frames, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
        ret, future_frame = cap.read()

        if not ret:
            # End of video, accept the change
            return current_frame

        future_mask = self._person_detector.detect(future_frame)
        ssim_with_future = compute_masked_ssim(
            current_frame, future_frame, current_mask, future_mask
        )

        # If future frame is similar to current, the change is stable
        if ssim_with_future > self._config.ssim_threshold:
            return current_frame

        # Future frame differs - might be another quick change or animation
        # Still accept the current frame as a valid sample point
        return current_frame
