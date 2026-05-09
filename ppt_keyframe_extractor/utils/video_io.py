"""Video I/O utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


@contextmanager
def open_video(video_path: str) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager for opening a video file.

    Args:
        video_path: Path to the video file.

    Yields:
        cv2.VideoCapture object.

    Raises:
        FileNotFoundError: If video file does not exist.
        IOError: If video cannot be opened.
    """
    import os

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        yield cap
    finally:
        cap.release()


def get_video_info(video_path: str) -> Tuple[int, int, float, int]:
    """Get basic video information.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (fps, total_frames, duration_seconds, width, height).
    """
    with open_video(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

    return fps, total_frames, duration, width, height


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[cv2.typing.MatLike]:
    """Read a specific frame by index.

    Args:
        cap: Opened VideoCapture object.
        frame_index: Frame index to read.

    Returns:
        BGR image, or None if read fails.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
