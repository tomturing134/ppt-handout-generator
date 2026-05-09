"""Image processing utilities - masked SSIM, information entropy, etc."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def compute_masked_ssim(
    frame1: np.ndarray,
    frame2: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
) -> float:
    """Compute SSIM between two frames with person regions masked out.

    This is the core innovation over MP4_to_PDF: by zeroing out teacher pixels
    before computing SSIM, we get a pure PPT-content similarity score that is
    immune to teacher movement.

    Args:
        frame1: First BGR image (H, W, 3).
        frame2: Second BGR image (H, W, 3).
        mask1: Binary person mask for frame1 (H, W), 1=person.
        mask2: Binary person mask for frame2 (H, W), 1=person.

    Returns:
        SSIM score between 0 and 1.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).copy()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).copy()

    # Create combined mask: pixels where EITHER frame has a person
    if mask1 is not None or mask2 is not None:
        combined_mask = np.zeros_like(gray1, dtype=bool)
        if mask1 is not None:
            combined_mask |= mask1.astype(bool)
        if mask2 is not None:
            combined_mask |= mask2.astype(bool)

        # Zero out person pixels (set to neutral gray=128)
        gray1[combined_mask] = 128
        gray2[combined_mask] = 128

    return ssim(gray1, gray2)


def compute_information_entropy(
    frame: np.ndarray,
    person_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute information entropy of a frame (higher = more content).

    Used to select the most information-rich frame from an animation sequence.
    The final state of a PPT animation typically has the highest entropy.

    Args:
        frame: BGR image (H, W, 3).
        person_mask: Binary person mask (H, W), 1=person. If provided,
            person region is excluded from entropy calculation.

    Returns:
        Information entropy value (higher = more content).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()

    # Mask out person region for entropy computation
    if person_mask is not None:
        gray[person_mask.astype(bool)] = 128  # neutral value

    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten().astype(np.float64)
    hist = hist / hist.sum()

    # Remove zero entries to avoid log(0)
    hist = hist[hist > 0]

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def is_dominant_color(
    frame: np.ndarray,
    threshold: float = 0.95,
    max_unique_colors: int = 8,
) -> bool:
    """Check if the frame is a blank transition (not a real slide).

    Used to filter out blank slides (all-black, all-white, blue screens)
    that occur during PPT transitions.

    A frame is considered blank if ONE color dominates (>threshold) AND
    the total number of distinct colors is very small. This avoids
    misclassifying content-rich white-background PPT slides as blank.

    Args:
        frame: BGR image (H, W, 3).
        threshold: Dominant color frequency threshold.
        max_unique_colors: Max distinct color count for blank frame.

    Returns:
        True if frame is likely a blank transition, False if real content.
    """
    # Downsample for speed
    small = cv2.resize(frame, (160, 90))
    pixels = small.reshape(-1, small.shape[-1])

    # Quantize to reduce color count (round to nearest 8)
    quantized = (pixels // 8) * 8

    # Count unique colors
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
    frequencies = counts / counts.sum()

    # Must have both: very few colors AND one dominates
    few_colors = len(unique_colors) <= max_unique_colors
    one_dominates = bool(np.any(frequencies > threshold))

    return few_colors and one_dominates


def compute_occlusion_score(
    person_mask: Optional[np.ndarray],
    ppt_region: Optional[tuple] = None,
) -> float:
    """Compute how much of the PPT region is occluded by a person.

    Args:
        person_mask: Binary mask (H, W), 1=person. None means no person.
        ppt_region: Optional (x, y, w, h) defining the PPT area.
            If None, the entire frame is the PPT region.

    Returns:
        Occlusion score between 0.0 (no occlusion) and 1.0 (fully occluded).
    """
    if person_mask is None:
        return 0.0

    h, w = person_mask.shape[:2]

    # Create PPT region mask
    if ppt_region is not None:
        x, y, rw, rh = ppt_region
        ppt_mask = np.zeros((h, w), dtype=np.uint8)
        ppt_mask[y : y + rh, x : x + rw] = 1
    else:
        ppt_mask = np.ones((h, w), dtype=np.uint8)

    # Compute intersection
    ppt_pixels = np.count_nonzero(ppt_mask)
    if ppt_pixels == 0:
        return 0.0

    occluded_pixels = np.count_nonzero(person_mask.astype(bool) & ppt_mask.astype(bool))
    return float(occluded_pixels / ppt_pixels)


def compute_histogram_correlation(
    frame1: np.ndarray,
    frame2: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
) -> float:
    """Compute histogram correlation between two frames with person pixels excluded.

    Histogram correlation complements SSIM: it measures color/intensity distribution
    changes that SSIM might miss when PPT slides share the same template background.

    Args:
        frame1: First BGR image.
        frame2: Second BGR image.
        mask1: Person mask for frame1.
        mask2: Person mask for frame2.

    Returns:
        Correlation score between 0 (very different) and 1 (identical distribution).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).copy()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).copy()

    # Create combined mask and zero out teacher pixels
    if mask1 is not None or mask2 is not None:
        combined_mask = np.zeros_like(gray1, dtype=bool)
        if mask1 is not None:
            combined_mask |= mask1.astype(bool)
        if mask2 is not None:
            combined_mask |= mask2.astype(bool)
        gray1[combined_mask] = 128
        gray2[combined_mask] = 128

    # Compute histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compute correlation
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, corr)))  # clamp to [0, 1]
