"""YOLO instance segmentation via ONNX Runtime (no PyTorch required)."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

YOLO_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx"
)
MODELS_DIR = Path(__file__).parent.parent / "models"


def _download_model(model_path: Path) -> None:
    """Download YOLO ONNX model from Ultralytics release."""
    logger.info(f"Downloading YOLO model from {YOLO_MODEL_URL}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    import requests
    import warnings
    warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

    r = requests.get(YOLO_MODEL_URL, stream=True, verify=False)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(
        f"Model downloaded: {model_path} "
        f"({model_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )


class ONNXSegmenter:
    """YOLO instance segmentation via ONNX Runtime.

    Uses ultralytics' pre-exported YOLO11n-seg ONNX model for person detection.
    No PyTorch required - only onnxruntime and numpy.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-seg.onnx",
        confidence: float = 0.5,
        target_size: int = 640,
    ) -> None:
        self._session = None
        self._model_path = MODELS_DIR / model_name
        self._confidence = confidence
        self._target_size = target_size
        self._loaded = False

        # YOLO metadata: names, colors
        self._class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        }

    def _ensure_model(self) -> None:
        """Lazy-load the ONNX model on first use."""
        if self._loaded:
            return

        if not self._model_path.exists():
            _download_model(self._model_path)

        import onnxruntime

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._session = onnxruntime.InferenceSession(
            str(self._model_path), providers=providers
        )
        self._loaded = True
        logger.info(
            f"ONNX model loaded: {self._model_path.name} "
            f"(provider={self._session.get_providers()[0]})"
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

        # Preprocess: resize to model input size
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self._target_size, self._target_size))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0)  # add batch dim

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_img})
        # outputs[0]: (1, 116, 8400) - detections
        # outputs[1]: (1, 32, 160, 160) - mask coefficients (protos)

        det_output = outputs[0][0]  # (116, 8400)
        proto = outputs[1][0]  # (32, 160, 160)

        # Parse YOLO output format
        # For detection + segmentation, each detection has: cx, cy, w, h, conf, cls, 32 mask coeffs
        # 116 = 4 (bbox) + 1 (conf) + 80 (classes) + 32 (mask coeffs) - 1 = ... depends on model
        # Actually for YOLO11n-seg: 4 + 1 + 80 + 32 = 117... let me check the actual format

        # Simplified: extract person class (index 0) with confidence > threshold
        # Format: [x, y, w, h, conf, cls1...cls80, mask_coeffs*32] = 4+1+80+32 = 117
        # Wait, YOLO11 uses the new format: xywh + conf + cls_scores(N)
        # For YOLO11 onnx export format... let me check
        # Actually YOLO11 output is (84 + mask_dim, num_boxes) = (84 + 32, 8400) = (116, 8400)
        # where 84 = 4(bbox) + 80(class scores)
        # So indices: 0-3=bbox, 4=best class score, 5-83=rewrite... no
        # Let me use a more robust approach

        boxes = []
        for i in range(det_output.shape[1]):
            det = det_output[:, i]
            # Class scores start at index 4
            class_scores = det[4:84]
            best_cls = int(np.argmax(class_scores))
            score = float(class_scores[best_cls])

            if best_cls == 0 and score >= self._confidence:
                # Person detected
                cx, cy, bw, bh = det[0], det[1], det[2], det[3]
                x1 = max(0, (cx - bw / 2) * w / self._target_size)
                y1 = max(0, (cy - bh / 2) * h / self._target_size)
                x2 = min(w, (cx + bw / 2) * w / self._target_size)
                y2 = min(h, (cy + bh / 2) * h / self._target_size)

                # Mask coefficients start at index 84
                mask_coeffs = det[84:116]  # 32 coefficients
                boxes.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "score": score,
                    "coeffs": mask_coeffs,
                })

        if not boxes:
            return None

        # Process masks
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            bbox = box["bbox"]
            coeffs = box["coeffs"]

            # Generate mask from proto * coeffs
            # proto shape: (32, 160, 160)
            mask = np.zeros((proto.shape[1], proto.shape[2]), dtype=np.float32)
            for j in range(len(coeffs)):
                mask += coeffs[j] * proto[j]

            # Apply sigmoid
            mask = 1.0 / (1.0 + np.exp(-mask))

            # Resize mask to original image size
            mask = cv2.resize(mask, (w, h))
            mask = (mask > 0.5).astype(np.uint8)

            # Crop to bounding box area
            x1, y1, x2, y2 = bbox
            crop_mask = np.zeros_like(mask)
            crop_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

            combined_mask = np.maximum(combined_mask, crop_mask)

        return combined_mask

    def detect_batch(self, frames: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Detect persons in a batch of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of binary masks (same order as input), None where no person.
        """
        return [self.detect(f) for f in frames]
