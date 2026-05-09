"""Person detector factory and base class.

Priority order for person detection:
1. ONNX Segmenter (yolo11n-seg via ONNX Runtime) - no PyTorch needed
2. OpenCV MOG2 background subtraction - fallback
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import ExtractionConfig
from ..models.bg_subtractor import MOG2Detector

logger = logging.getLogger(__name__)


class PersonDetectorBase(ABC):
    """Abstract base class for person detection strategies."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect person in a frame and return binary mask.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Binary mask (H, W) where 1=person, or None if no person detected.
        """
        ...


class PersonDetectorFactory:
    """Factory to create the appropriate person detector based on config."""

    @staticmethod
    def create(config: ExtractionConfig) -> PersonDetectorBase:
        """Create a person detector instance.

        Priority: ONNX > MOG2

        Args:
            config: Extraction configuration.

        Returns:
            A PersonDetectorBase implementation.
        """
        if config.person_detector == "onnx":
            return PersonDetectorFactory._try_onnx(config)
        elif config.person_detector == "mog2":
            logger.info("Using MOG2 background subtraction for person detection")
            return MOG2Detector()
        elif config.person_detector == "yolo_seg":
            # Try ONNX first (replaces PyTorch-based YOLO which is incompatible)
            logger.info(
                "yolo_seg mode mapped to ONNX (PyTorch unavailable in this environment)"
            )
            return PersonDetectorFactory._try_onnx(config)
        else:
            raise ValueError(
                f"Unknown person detector: {config.person_detector}. "
                f"Supported: 'onnx', 'mog2'"
            )

    @staticmethod
    def _try_onnx(config: ExtractionConfig) -> PersonDetectorBase:
        """Try to create an ONNX Runtime-based segmenter, fall back to MOG2."""
        try:
            import onnxruntime  # noqa: F401 - verify it's installed

            from ..models.onnx_segmenter import ONNXSegmenter

            detector = ONNXSegmenter(
                model_name=config.yolo_model.replace(".pt", ".onnx"),
                confidence=config.yolo_confidence,
                target_size=config.resize_for_detection,
            )
            # Test model loading
            detector._ensure_model()
            logger.info("Using ONNX Runtime YOLO segmentation for person detection")
            return detector
        except Exception as e:
            logger.warning(
                f"Failed to initialize ONNX segmenter ({e}), falling back to MOG2"
            )
            return MOG2Detector()
