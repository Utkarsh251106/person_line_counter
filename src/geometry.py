from __future__ import annotations

import cv2
import numpy as np


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return (cx, cy) for a binary/boolean mask, or None for an empty mask."""
    binary = (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary, binaryImage=True)

    if moments["m00"] <= 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return float(cx), float(cy)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a model mask to frame resolution without changing its binary nature."""
    if mask.shape[:2] == (height, width):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)
