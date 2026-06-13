from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackingSession:
    """
    Snapshot of tracker state for one frame, returned by TrackerPipeline.tick().

    Frontends read this to drive rendering and UI updates without touching
    any pipeline internals directly.
    """

    people: list = field(default_factory=list)
    target_track_id: int | None = None
    tracked_person: dict | None = None
    reid_matches: dict | None = None
    action_prediction: Any = None
    command: str | None = None
    enroll_status: str | None = None
    gallery_updated: bool = False
