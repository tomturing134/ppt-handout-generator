"""Core data types for ppt_keyframe_extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ChangeType(Enum):
    """Type of change detected between two frames."""

    NO_CHANGE = "no_change"
    ANIMATION_STEP = "animation_step"  # small change, likely PPT animation
    SLIDE_CHANGE = "slide_change"  # large change, likely new slide


@dataclass
class FrameInfo:
    """Metadata for a single video frame."""

    frame_index: int
    timestamp: float  # seconds from video start
    image: np.ndarray  # BGR image (H, W, 3)
    person_mask: Optional[np.ndarray] = None  # binary mask (H, W), 1=person
    occlusion_score: float = 0.0  # 0.0=no occlusion, 1.0=fully occluded
    ssim_score: float = 1.0  # similarity to previous masked frame

    def __repr__(self) -> str:
        return (
            f"FrameInfo(idx={self.frame_index}, t={self.timestamp:.1f}s, "
            f"occlusion={self.occlusion_score:.1%}, ssim={self.ssim_score:.3f})"
        )


@dataclass
class SlideCluster:
    """A group of frames showing the same PPT slide."""

    cluster_id: int
    frames: List[FrameInfo] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    is_animation_sequence: bool = False
    best_frame: Optional[FrameInfo] = None  # set after keyframe selection

    def __repr__(self) -> str:
        n = len(self.frames)
        best = f", best={self.best_frame.frame_index}" if self.best_frame else ""
        anim = " [ANIM]" if self.is_animation_sequence else ""
        return (
            f"SlideCluster(id={self.cluster_id}, frames={n}, "
            f"t={self.start_time:.1f}-{self.end_time:.1f}s{anim}{best})"
        )


@dataclass
class ExtractionConfig:
    """All configurable parameters with sensible defaults."""

    # --- Video sampling ---
    initial_jump_seconds: float = 1.0  # initial jump step in seconds
    max_jump_seconds: float = 64.0  # maximum jump step (exponential backoff cap)
    use_fixed_interval: bool = True  # True=fixed 5s intervals, False=double-jump

    # --- Slide detection ---
    ssim_threshold: float = 0.95  # below this = slide change detected (lower = more sensitive)
    animation_ssim_threshold: float = 0.70  # below this = real slide change (vs animation)
    stability_window_seconds: float = 3.0  # seconds to confirm animation end
    edge_change_threshold: float = 0.013  # edge density change threshold

    # --- Person detection ---
    person_detector: str = "onnx"  # "onnx" (recommended) or "mog2"
    yolo_model: str = "yolo11n-seg.onnx"
    yolo_confidence: float = 0.5
    yolo_device: str = "cpu"  # "cpu" only (PyTorch/CUDA unavailable)

    # --- Occlusion ---
    max_occlusion_threshold: float = 0.8  # warn if best frame exceeds this
    ppt_region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) crop

    # --- Output ---
    output_dir: str = "./output"
    image_format: str = "jpg"
    image_quality: int = 95

    # --- Processing ---
    resize_for_detection: int = 640  # resize frame before person detection

    def __post_init__(self) -> None:
        if self.ssim_threshold <= self.animation_ssim_threshold:
            raise ValueError(
                f"ssim_threshold ({self.ssim_threshold}) must be > "
                f"animation_ssim_threshold ({self.animation_ssim_threshold})"
            )
        if self.person_detector not in ("onnx", "mog2", "yolo_seg"):
            raise ValueError(
                f"person_detector must be 'onnx' or 'mog2', got '{self.person_detector}'"
            )


@dataclass
class ExtractionResult:
    """Result of the extraction pipeline."""

    keyframes: List[FrameInfo] = field(default_factory=list)
    pdf_path: str = ""
    image_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    slide_count: int = 0

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(slides={self.slide_count}, "
            f"pdf={self.pdf_path}, warnings={len(self.warnings)})"
        )
