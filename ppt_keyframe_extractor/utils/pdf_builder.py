"""PDF generation from keyframe images."""

from __future__ import annotations

import logging
import os
from typing import List

import cv2
import img2pdf
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_keyframe_images(
    keyframes: list,
    output_dir: str,
    image_format: str = "jpg",
    quality: int = 95,
) -> List[str]:
    """Save keyframe images to disk.

    Args:
        keyframes: List of FrameInfo objects with .image and .frame_index.
        output_dir: Directory to save images.
        image_format: Image format ("jpg" or "png").
        quality: JPEG quality (1-100).

    Returns:
        List of saved image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, kf in enumerate(keyframes):
        ext = image_format if image_format in ("jpg", "png") else "jpg"
        filename = f"slide_{i + 1:03d}.{ext}"
        filepath = os.path.join(output_dir, filename)

        if ext == "jpg":
            cv2.imwrite(filepath, kf.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(filepath, kf.image)

        paths.append(filepath)
        logger.debug(f"Saved keyframe: {filepath}")

    return paths


def build_pdf(
    image_paths: List[str],
    output_path: str,
) -> str:
    """Build a PDF from a list of image files.

    Uses img2pdf for lossless embedding (JPEG data is embedded directly
    without re-encoding).

    Args:
        image_paths: List of image file paths (one per page).
        output_path: Path for the output PDF file.

    Returns:
        Path to the generated PDF file.
    """
    if not image_paths:
        raise ValueError("No images provided for PDF generation")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert images to PDF-compatible format
    # img2pdf handles JPEG directly; for other formats, convert via Pillow
    pdf_images = []
    for path in image_paths:
        if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
            pdf_images.append(path)
        else:
            # Convert to JPEG in memory for img2pdf compatibility
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            temp_jpg = path.rsplit(".", 1)[0] + "_temp.jpg"
            img.save(temp_jpg, "JPEG", quality=95)
            pdf_images.append(temp_jpg)

    # Generate PDF
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(pdf_images))

    # Clean up temp files
    for path in pdf_images:
        if path.endswith("_temp.jpg") and os.path.exists(path):
            os.remove(path)

    logger.info(f"PDF generated: {output_path} ({len(pdf_images)} pages)")
    return output_path
