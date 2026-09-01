from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossingEvent:
    track_id: int
    direction: str
    counter: int


class LineCounter:
    """State machine for a vertical line with a dead-band buffer."""

    def __init__(self, line_x: float, buffer_half_width: float = 25.0):
        self.line_x = line_x
        self.buffer_half_width = buffer_half_width
        self.count = 0

    def side_of_line(self, x: float) -> str:
        if x < self.line_x - self.buffer_half_width:
            return "left"
        if x > self.line_x + self.buffer_half_width:
            return "right"
        return "buffer"

    def update_track(self, track) -> CrossingEvent | None:
        """Update a track's side state and return a crossing event if confirmed.

        The track stores the last stable side. We count only:
            left -> right
            right -> left

        A buffer around the line avoids repeated crossings due to jitter.
        """
        if not track.confirmed:
            return None

        current_side = self.side_of_line(track.centroid[0])

        if current_side == "buffer":
            return None

        if track.side == "unknown":
            track.side = current_side
            return None

        if track.side == current_side:
            return None

        previous_side = track.side
        track.side = current_side

        if previous_side == "left" and current_side == "right":
            self.count += 1
            return CrossingEvent(track.track_id, "left_to_right", self.count)

        if previous_side == "right" and current_side == "left":
            self.count -= 1
            return CrossingEvent(track.track_id, "right_to_left", self.count)

        return None
