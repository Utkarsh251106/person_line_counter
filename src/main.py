from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .counter import LineCounter
from .geometry import mask_centroid, resize_mask
from .tracker import CentroidTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Person segmentation + mask-centroid tracking + line counting"
    )
    parser.add_argument("--source", default="0", help="Video path, RTSP URL, or webcam index")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--person-class-id", type=int, default=0)

    parser.add_argument("--line-x", type=float, default=None)
    parser.add_argument("--line-ratio", type=float, default=0.50)
    parser.add_argument("--buffer-half-width", type=float, default=25.0)

    parser.add_argument("--max-distance", type=float, default=100.0)
    parser.add_argument("--max-missed", type=int, default=20)
    parser.add_argument("--min-confirmed-frames", type=int, default=3)
    parser.add_argument("--trajectory-length", type=int, default=30)

    parser.add_argument("--output", default="outputs/processed.mp4")
    parser.add_argument("--events", default="outputs/events.csv")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a simple translucent mask overlay."""
    overlay = frame.copy()
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 255, 0])).astype(
        np.uint8
    )
    return overlay


def draw_track(frame, track):
    cx, cy = map(int, track.centroid)

    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    if len(track.history) >= 2:
        points = np.array(
            [(int(x), int(y)) for x, y in track.history],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], False, (255, 255, 0), 2)

    label = f"ID {track.track_id}"
    cv2.putText(
        frame,
        label,
        (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    output_path = Path(args.output)
    events_path = Path(args.events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    source = parse_source(args.source)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not determine video dimensions.")

    line_x = args.line_x
    if line_x is None:
        line_x = width * args.line_ratio

    counter = LineCounter(
        line_x=line_x,
        buffer_half_width=args.buffer_half_width,
    )

    tracker = CentroidTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        min_confirmed_frames=args.min_confirmed_frames,
        history_length=args.trajectory_length,
    )

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

    events_file = None
    events_writer = None
    if not args.no_save:
        events_file = open(events_path, "w", newline="", encoding="utf-8")
        events_writer = csv.writer(events_file)
        events_writer.writerow(
            ["timestamp", "frame", "track_id", "direction", "counter"]
        )

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            results = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
            )

            result = results[0]

            centroids = []
            masks_for_frame = []

            if result.masks is not None and result.boxes is not None:
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                masks = result.masks.data.detach().cpu().numpy()

                for class_id, confidence, raw_mask in zip(
                    classes, confidences, masks
                ):
                    if class_id != args.person_class_id:
                        continue

                    if confidence < args.conf:
                        continue

                    mask = resize_mask(
                        raw_mask > args.mask_threshold,
                        width=width,
                        height=height,
                    )

                    centroid = mask_centroid(mask)
                    if centroid is None:
                        continue

                    centroids.append(centroid)
                    masks_for_frame.append(mask)

            # Update tracker from mask centroids only.
            tracks = tracker.update(centroids)

            # Draw masks.
            for mask in masks_for_frame:
                frame = overlay_mask(frame, mask)

            # Update crossing state.
            for track in tracks.values():
                if track.missed > 0:
                    continue

                event = counter.update_track(track)
                if event is not None and events_writer is not None:
                    events_writer.writerow(
                        [
                            datetime.now().isoformat(timespec="milliseconds"),
                            frame_number,
                            event.track_id,
                            event.direction,
                            event.counter,
                        ]
                    )
                    events_file.flush()

            # Draw active tracks.
            for track in tracks.values():
                if track.missed == 0:
                    draw_track(frame, track)

            # Draw line and buffer.
            x = int(line_x)
            left_buffer = int(line_x - args.buffer_half_width)
            right_buffer = int(line_x + args.buffer_half_width)

            cv2.line(
                frame,
                (x, 0),
                (x, height),
                (0, 0, 255),
                3,
            )
            cv2.line(
                frame,
                (left_buffer, 0),
                (left_buffer, height),
                (100, 100, 100),
                1,
            )
            cv2.line(
                frame,
                (right_buffer, 0),
                (right_buffer, height),
                (100, 100, 100),
                1,
            )

            # Dashboard.
            cv2.rectangle(frame, (10, 10), (310, 105), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Net Count: {counter.count}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Active Tracks: {sum(t.missed == 0 for t in tracks.values())}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Q = quit",
                (25, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Person Segmentation + Line Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if events_file is not None:
            events_file.close()

        cv2.destroyAllWindows()

    print(f"Final counter: {counter.count}")
    if not args.no_save:
        print(f"Processed video: {output_path}")
        print(f"Events CSV:      {events_path}")


if __name__ == "__main__":
    main()
