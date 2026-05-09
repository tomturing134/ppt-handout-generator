"""YOLO instance segmentation for person detection."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLOSegmenter:
    """YOLO instance segmentation for precise person masks.

    Uses ultralytics YOLO segmentation models (e.g., yolo11n-seg) to detect
    people in video frames and produce pixel-accurate binary masks.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        confidence: float = 0.5,
        device: str = "auto",
        target_size: int = 640,
    ) -> None:
        self._model = None
        self._model_name = model_name
        self._confidence = confidence
        self._device = device
        self._target_size = target_size
        self._loaded = False

    def _ensure_model(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self._model_name}")
            self._model = YOLO(self._model_name)
            # Determine device
            if self._device == "auto":
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"YOLO running on device: {self._device}")
            self._loaded = True
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO segmentation. "
                "Install with: pip install ultralytics"
            )

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect person in a single frame and return binary mask.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Binary mask (H, W) where 1=person, or None if no person detected.
        """
        self._ensure_model()

        h, w = frame.shape[:2]
        results = self._model(
            frame,
            conf=self._confidence,
            classes=[0],  # class 0 = person in COCO
            device=self._device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None

        result = results[0]
        if result.masks is None:
            return None

        # Combine all person masks into one binary mask
        masks = result.masks.data  # (N, H_mask, W_mask)
        if masks is None or len(masks) == 0:
            return None

        # Resize masks to original frame size and combine
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, mask_resized)

        return combined_mask

    def detect_batch(self, frames: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Detect persons in a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of binary masks (same order as input), None where no person.
        """
        self._ensure_model()

        h, w = frames[0].shape[:2] if frames else (0, 0)
        results = self._model(
            frames,
            conf=self._confidence,
            classes=[0],
            device=self._device,
            verbose=False,
        )

        masks_list: list[Optional[np.ndarray]] = []
        for result in results:
            if result.masks is None or len(result.masks.data) == 0:
                masks_list.append(None)
                continue

            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, mask_resized)

            masks_list.append(combined_mask)

        return masks_list
