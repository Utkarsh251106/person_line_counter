from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    previous_centroid: tuple[float, float] | None = None
    missed: int = 0
    hits: int = 1
    confirmed: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)
    side: str = "unknown"
    counted: bool = False


class CentroidTracker:
    """Simple greedy centroid tracker.

    This deliberately does not require bounding boxes. It is a good baseline
    for moderate-density scenes. For heavy occlusion, replace this component
    with a stronger MOT implementation.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missed: int = 20,
        min_confirmed_frames: int = 3,
        history_length: int = 30,
    ):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_confirmed_frames = min_confirmed_frames
        self.history_length = history_length

        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(
        self,
        centroids: list[tuple[float, float]],
    ) -> dict[int, Track]:
        """Update tracks and return active tracks.

        Greedy nearest-neighbor matching is intentionally simple and transparent.
        """
        track_ids = list(self.tracks.keys())
        candidates: list[tuple[float, int, int]] = []

        for track_id in track_ids:
            old = self.tracks[track_id].centroid
            for detection_idx, new in enumerate(centroids):
                d = self._distance(old, new)
                if d <= self.max_distance:
                    candidates.append((d, track_id, detection_idx))

        candidates.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_detections = set()

        for _, track_id, detection_idx in candidates:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue

            track = self.tracks[track_id]
            new_centroid = centroids[detection_idx]

            track.previous_centroid = track.centroid
            track.centroid = new_centroid
            track.missed = 0
            track.hits += 1
            track.confirmed = track.hits >= self.min_confirmed_frames

            track.history.append(new_centroid)
            if len(track.history) > self.history_length:
                track.history.pop(0)

            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].missed += 1

        for detection_idx, centroid in enumerate(centroids):
            if detection_idx in matched_detections:
                continue

            track = Track(
                track_id=self.next_id,
                centroid=centroid,
                history=[centroid],
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        dead_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_missed
        ]

        for track_id in dead_ids:
            del self.tracks[track_id]

        return self.tracks
