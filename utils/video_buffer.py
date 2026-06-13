from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class RollingVideoBuffer:
    """
    Keeps a rolling pre-detection window of raw frames. On trigger_save(), snapshots
    the pre-buffer and then collects post-detection frames before writing to disk
    in a background thread.
    """

    def __init__(self, pre_seconds: float = 5.0, post_seconds: float = 3.0, fps: float = 30.0):
        self._fps = fps
        self._pre: deque[np.ndarray] = deque(maxlen=int(pre_seconds * fps))
        self._post_target = int(post_seconds * fps)
        self._state = "idle"  # "idle" | "capturing"
        self._post_frames: list[np.ndarray] = []
        self._pre_snapshot: list[np.ndarray] = []
        self._pending_path: Path | None = None

    # ------------------------------------------------------------------ public

    def push(self, frame: np.ndarray) -> None:
        """Call every tick with the raw (unannotated) frame."""
        if self._state == "capturing":
            self._post_frames.append(frame.copy())
            if len(self._post_frames) >= self._post_target:
                self._flush()
        self._pre.append(frame.copy())

    def trigger_save(self, output_path: Path) -> bool:
        """
        Snapshot the current pre-buffer and begin capturing post-detection frames.
        Returns False if a capture is already in progress (the trigger is ignored).
        """
        if self._state == "capturing":
            return False
        self._pre_snapshot = list(self._pre)
        self._post_frames = []
        self._pending_path = output_path
        self._state = "capturing"
        return True

    @property
    def is_capturing(self) -> bool:
        return self._state == "capturing"

    # ------------------------------------------------------------------ private

    def _flush(self) -> None:
        frames = self._pre_snapshot + self._post_frames
        path = self._pending_path
        fps = self._fps
        self._state = "idle"
        self._post_frames = []
        self._pre_snapshot = []
        self._pending_path = None
        threading.Thread(target=self._write, args=(frames, path, fps), daemon=True).start()

    @staticmethod
    def _write(frames: list[np.ndarray], path: Path, fps: float) -> None:
        if not frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
