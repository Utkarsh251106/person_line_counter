"""
Classical Computer Vision - Plate Inspection
============================================

Tasks:
1. Segment the plate
2. Detect the 4 holes correctly
3. Detect the 6 screws correctly
4. Calculate plate dimensions in pixels

Input:
    test.jpeg

Output:
    plate_inspection_result.jpg

Dependencies:
    pip install opencv-python numpy

Run:
    python plate_inspection.py

This solution uses classical computer vision only.
No YOLO / deep learning is used.
"""

import argparse
import cv2
import numpy as np


# ---------------------------------------------------------------------
# 1. PLATE SEGMENTATION
# ---------------------------------------------------------------------
def segment_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The plate is brighter than the green background.
    _, binary = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("Could not detect the plate.")

    plate_contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(plate_contour) < 10000:
        raise RuntimeError(
            "Detected object is too small to be the plate."
        )

    plate_mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        plate_mask,
        [plate_contour],
        -1,
        255,
        thickness=-1
    )

    return plate_contour, plate_mask


# ---------------------------------------------------------------------
# Utility: distance from point to a line segment
# ---------------------------------------------------------------------
def point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    vector = end - start
    denominator = np.dot(vector, vector)

    if denominator == 0:
        return float(np.linalg.norm(point - start))

    t = np.dot(point - start, vector) / denominator
    t = np.clip(t, 0.0, 1.0)

    projection = start + t * vector

    return float(np.linalg.norm(point - projection))


# ---------------------------------------------------------------------
# 2. HOLE DETECTION
# ---------------------------------------------------------------------
def detect_holes(image, plate_contour):
    """
    Detect the four actual through-holes.

    The plate contains several circular/irregular objects, including
    screws and marks. Therefore, simply thresholding dark pixels can
    produce false holes.

    The actual holes have three useful classical-CV properties:
        1. They are approximately circular.
        2. They are located close to the LEFT or RIGHT edge of the plate.
        3. Their centers are dark.

    Hough Circle Transform is used to generate circular candidates.
    Plate bounding-box geometry and center intensity are then used
    to reject the middle objects and screws.

    No hard-coded hole coordinates are used.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Median filtering reduces noise while preserving circular edges.
    blurred = cv2.medianBlur(
        gray,
        5
    )

    # Generate circular candidates.
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=19,
        minRadius=7,
        maxRadius=16
    )

    if circles is None:
        return []

    circles = np.round(
        circles[0]
    ).astype(int)

    # Axis-aligned bounding box of the detected plate.
    # This is deliberately used instead of assuming a particular
    # minAreaRect edge ordering.
    px, py, pw, ph = cv2.boundingRect(
        plate_contour
    )

    candidates = []

    for circle in circles:
        cx, cy, radius = circle

        # Candidate must be inside the plate.
        if cv2.pointPolygonTest(
            plate_contour,
            (float(cx), float(cy)),
            False
        ) < 0:
            continue

        # Actual holes are close to the left/right sides.
        # Use a percentage of the plate width so the method scales
        # with different image resolutions.
        normalized_x = (cx - px) / max(pw, 1)

        is_left_side = (
            0.02 <= normalized_x <= 0.09
        )

        is_right_side = (
            0.91 <= normalized_x <= 0.98
        )

        if not (is_left_side or is_right_side):
            continue

        # Check the center intensity.
        # Holes are dark; the nearby screws are brighter/reflective.
        r = 5

        y1 = max(0, cy - r)
        y2 = min(gray.shape[0], cy + r + 1)

        x1 = max(0, cx - r)
        x2 = min(gray.shape[1], cx + r + 1)

        center_mean = float(
            np.mean(
                gray[y1:y2, x1:x2]
            )
        )

        if center_mean > 125:
            continue

        candidates.append(
            {
                "center": (
                    int(cx),
                    int(cy)
                ),
                "radius": float(radius),
                "center_mean": center_mean
            }
        )

    # Remove duplicate circle detections.
    holes = remove_duplicate_detections(
        candidates,
        min_distance=30
    )

    holes.sort(
        key=lambda h: (
            h["center"][1],
            h["center"][0]
        )
    )

    return holes


# ---------------------------------------------------------------------
# 3. SCREW DETECTION
# ---------------------------------------------------------------------
def detect_screws(image, plate_mask, holes):
    """
    Detect screws using local intensity variation.

    Screws have strong local texture/contrast compared with the
    relatively smooth plate.

    The four holes are explicitly excluded using their detected
    centers.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    # Local mean.
    local_mean = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=3
    )

    # Local squared mean.
    local_mean_sq = cv2.GaussianBlur(
        gray * gray,
        (0, 0),
        sigmaX=3
    )

    local_variance = np.maximum(
        local_mean_sq - local_mean * local_mean,
        0
    )

    local_std = np.sqrt(
        local_variance
    )

    _, variation_mask = cv2.threshold(
        local_std.astype(np.uint8),
        28,
        255,
        cv2.THRESH_BINARY
    )

    variation_mask = cv2.bitwise_and(
        variation_mask,
        plate_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    variation_mask = cv2.morphologyEx(
        variation_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            variation_mask,
            connectivity=8
        )
    )

    candidates = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if not (20 <= area <= 1000):
            continue

        if not (5 <= w <= 80 and 5 <= h <= 80):
            continue

        aspect_ratio = w / max(h, 1)

        if not (0.45 <= aspect_ratio <= 2.2):
            continue

        cx, cy = centroids[i]

        cx = int(round(cx))
        cy = int(round(cy))

        # Average intensity at the center.
        r = 5

        y1 = max(0, cy - r)
        y2 = min(gray.shape[0], cy + r + 1)

        x1 = max(0, cx - r)
        x2 = min(gray.shape[1], cx + r + 1)

        center_mean = float(
            np.mean(
                gray[y1:y2, x1:x2]
            )
        )

        candidates.append(
            {
                "center": (cx, cy),
                "radius": max(w, h) / 2.0,
                "area": int(area),
                "center_mean": center_mean
            }
        )

    screws = []

    for candidate in candidates:
        cx, cy = candidate["center"]

        # Exclude anything close to a detected hole.
        near_hole = False

        for hole in holes:
            hx, hy = hole["center"]

            distance = np.sqrt(
                (cx - hx) ** 2 +
                (cy - hy) ** 2
            )

            if distance < 30:
                near_hole = True
                break

        if near_hole:
            continue

        # Screws have a brighter metal/reflection center.
        if candidate["center_mean"] < 120:
            continue

        screws.append(candidate)

    screws = remove_duplicate_detections(
        screws,
        min_distance=35
    )

    screws.sort(
        key=lambda s: (
            s["center"][1],
            s["center"][0]
        )
    )

    return screws


# ---------------------------------------------------------------------
# Utility: duplicate suppression
# ---------------------------------------------------------------------
def remove_duplicate_detections(
    detections,
    min_distance=35
):
    result = []

    detections = sorted(
        detections,
        key=lambda d: (
            d.get("area", 0),
            d.get("center_mean", 0)
        ),
        reverse=True
    )

    for detection in detections:
        cx, cy = detection["center"]

        duplicate = False

        for selected in result:
            sx, sy = selected["center"]

            distance = np.sqrt(
                (cx - sx) ** 2 +
                (cy - sy) ** 2
            )

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            result.append(detection)

    return result


# ---------------------------------------------------------------------
# 4. PLATE DIMENSION CALCULATION
# ---------------------------------------------------------------------
def calculate_plate_dimensions(
    plate_contour
):
    rect = cv2.minAreaRect(
        plate_contour
    )

    width, height = rect[1]

    length_px = max(
        width,
        height
    )

    width_px = min(
        width,
        height
    )

    return {
        "length_px": float(length_px),
        "width_px": float(width_px),
        "angle": float(rect[2]),
        "rectangle": rect
    }


# ---------------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------------
def draw_results(
    image,
    plate_contour,
    holes,
    screws,
    dimensions
):
    result = image.copy()

    # Plate contour.
    cv2.drawContours(
        result,
        [plate_contour],
        -1,
        (255, 0, 0),
        4
    )

    # Minimum-area rectangle.
    rect = dimensions["rectangle"]

    box = cv2.boxPoints(
        rect
    ).astype(np.int32)

    cv2.drawContours(
        result,
        [box.reshape(-1, 1, 2)],
        -1,
        (255, 0, 255),
        3
    )

    # Draw holes in red.
    for index, hole in enumerate(
        holes,
        start=1
    ):
        x, y = hole["center"]

        radius = int(
            hole["radius"] + 4
        )

        cv2.circle(
            result,
            (x, y),
            radius,
            (0, 0, 255),
            3
        )

        cv2.putText(
            result,
            f"H{index}",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    # Draw screws in orange.
    for index, screw in enumerate(
        screws,
        start=1
    ):
        x, y = screw["center"]

        radius = int(
            screw["radius"]
        )

        cv2.circle(
            result,
            (x, y),
            radius,
            (0, 165, 255),
            3
        )

        cv2.putText(
            result,
            f"S{index}",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 165, 255),
            2
        )

    # Information panel.
    panel_height = 125

    cv2.rectangle(
        result,
        (15, 15),
        (760, panel_height),
        (255, 255, 255),
        -1
    )

    lines = [
        (
            f"Plate: "
            f"{dimensions['length_px']:.1f} "
            f"x "
            f"{dimensions['width_px']:.1f} px"
        ),
        f"Holes: {len(holes)}",
        f"Screws: {len(screws)}"
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (30, 45 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2
        )

    return result


# ---------------------------------------------------------------------
# MAIN INSPECTION PIPELINE
# ---------------------------------------------------------------------
def inspect_plate(
    image_path,
    output_path
):
    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # 1. Segment plate.
    plate_contour, plate_mask = (
        segment_plate(image)
    )

    # 2. Detect holes.
    holes = detect_holes(
        image,
        plate_contour
    )

    # 3. Detect screws.
    screws = detect_screws(
        image,
        plate_mask,
        holes
    )

    # 4. Calculate dimensions.
    dimensions = calculate_plate_dimensions(
        plate_contour
    )

    # 5. Draw results.
    annotated = draw_results(
        image,
        plate_contour,
        holes,
        screws,
        dimensions
    )

    # Save result.
    cv2.imwrite(
        output_path,
        annotated
    )

    # Terminal output.
    print()
    print("=" * 55)
    print("             PLATE INSPECTION RESULT")
    print("=" * 55)
    print("Plate detected       : YES")
    print(
        f"Plate length         : "
        f"{dimensions['length_px']:.2f} px"
    )
    print(
        f"Plate width          : "
        f"{dimensions['width_px']:.2f} px"
    )
    print(
        f"Holes detected       : "
        f"{len(holes)}"
    )
    print(
        f"Screws detected      : "
        f"{len(screws)}"
    )
    print("-" * 55)
    print("Dimension unit       : pixels")
    print(
        f"Annotated output     : "
        f"{output_path}"
    )
    print("=" * 55)
    print()

    return {
        "plate_detected": True,
        "plate_length_px": dimensions["length_px"],
        "plate_width_px": dimensions["width_px"],
        "holes": len(holes),
        "screws": len(screws)
    }


# ---------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classical CV plate inspection"
    )

    parser.add_argument(
        "--image",
        default="test.jpeg",
        help="Input image path"
    )

    parser.add_argument(
        "--output",
        default="plate_inspection_result.jpg",
        help="Output image path"
    )

    args = parser.parse_args()

    inspect_plate(
        args.image,
        args.output
    )
