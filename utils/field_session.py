from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

YARDS_TO_METERS = 0.9144
GROUPING_DISTANCE_M = 10 * YARDS_TO_METERS  # 10 yards
GROUPING_TIME_S = 60.0

_SESSIONS_DIR = Path("RecordedTestSwings")
_VIDEOS_DIR = Path("RecordedTestVideos")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class FieldSessionManager:
    def __init__(self, data: dict):
        self._data = data
        self._active_hole: int | None = None

    # ------------------------------------------------------------------ factory

    @classmethod
    def create(cls, name: str) -> "FieldSessionManager":
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "name": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "holes": {},
        }
        mgr = cls(data)
        mgr.save()
        return mgr

    @classmethod
    def resume(cls, name: str) -> "FieldSessionManager":
        path = _SESSIONS_DIR / f"{name}.json"
        with path.open() as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def list_sessions(cls) -> list[str]:
        if not _SESSIONS_DIR.exists():
            return []
        return sorted(p.stem for p in _SESSIONS_DIR.glob("*.json"))

    # ------------------------------------------------------------------ hole

    def start_hole(self, hole_number: int) -> None:
        self._active_hole = hole_number
        key = str(hole_number)
        if key not in self._data["holes"]:
            self._data["holes"][key] = {"shots": {}}

    def end_hole(self) -> None:
        self._active_hole = None

    # ------------------------------------------------------------------ swing

    def add_swing(
        self,
        lat: float | None,
        lon: float | None,
        ts: datetime,
        video_path: str | None,
    ) -> tuple[int, int]:
        """Add a detection. Returns (shot_num, detection_num) (1-indexed)."""
        hole_num = self._active_hole
        if hole_num is None:
            raise RuntimeError("No active hole — call start_hole() first")

        hole_key = str(hole_num)
        shots: dict = self._data["holes"][hole_key]["shots"]

        shot_id = self._find_matching_shot(shots, lat, lon, ts)
        if shot_id is None:
            shot_id = self._next_shot_id(shots)
            shots[str(shot_id)] = {"detections": []}

        detections: list = shots[str(shot_id)]["detections"]
        det_num = len(detections) + 1
        detections.append(
            {
                "location": {
                    "lat": lat,
                    "lon": lon,
                },
                "time": ts.isoformat(timespec="seconds"),
                "video_path": str(video_path) if video_path else None,
            }
        )
        return shot_id, det_num

    def update_detection_video(
        self, hole_num: int, shot_num: int, det_num: int, video_path: Path
    ) -> None:
        det = self._data["holes"][str(hole_num)]["shots"][str(shot_num)]["detections"][
            det_num - 1
        ]
        det["video_path"] = str(video_path)

    def shot_count(self, hole_num: int) -> int:
        return len(self._data["holes"].get(str(hole_num), {}).get("shots", {}))

    def video_path_for(self, hole_num: int, shot_num: int, det_num: int) -> Path:
        name = self._data["name"]
        return (
            _VIDEOS_DIR
            / name
            / f"hole_{hole_num}"
            / f"shot_{shot_num}"
            / f"detection_{det_num}.mp4"
        )

    # ------------------------------------------------------------------ persist

    @property
    def name(self) -> str:
        return self._data["name"]

    def save(self) -> None:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _SESSIONS_DIR / f"{self.name}.json"
        with path.open("w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------ private

    def _find_matching_shot(
        self,
        shots: dict,
        lat: float | None,
        lon: float | None,
        ts: datetime,
    ) -> int | None:
        if not shots:
            return None
        for shot_id_str, shot in shots.items():
            for det in shot["detections"]:
                dlat = det["location"]["lat"]
                dlon = det["location"]["lon"]
                dtime = datetime.fromisoformat(det["time"])
                time_diff = abs((ts - dtime).total_seconds())
                if time_diff > GROUPING_TIME_S:
                    continue
                if lat is None or lon is None or dlat is None or dlon is None:
                    # Can't measure distance; don't merge
                    continue
                if _haversine_m(lat, lon, dlat, dlon) <= GROUPING_DISTANCE_M:
                    return int(shot_id_str)
        return None

    @staticmethod
    def _next_shot_id(shots: dict) -> int:
        if not shots:
            return 1
        return max(int(k) for k in shots) + 1
