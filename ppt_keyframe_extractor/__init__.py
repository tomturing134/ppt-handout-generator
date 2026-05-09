"""ppt_keyframe_extractor - Extract PPT keyframes from lecture videos.

This library extracts PPT slide keyframes from lecture recording videos
where a teacher may occlude the slide content. It selects the least-occluded
frame for each unique slide and generates a PDF.

Basic usage:
    from ppt_keyframe_extractor import LecturePPTExtractor

    extractor = LecturePPTExtractor()
    result = extractor.extract("lecture.mp4", output_dir="./output")
    print(f"Extracted {result.slide_count} slides")
    print(f"PDF saved to {result.pdf_path}")
"""

from .core.pipeline import LecturePPTExtractor
from .types import (
    ChangeType,
    ExtractionConfig,
    ExtractionResult,
    FrameInfo,
    SlideCluster,
)

__all__ = [
    "LecturePPTExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "FrameInfo",
    "SlideCluster",
    "ChangeType",
]

__version__ = "0.2.0"
