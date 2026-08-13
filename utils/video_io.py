from __future__ import annotations

import cv2


class SequentialFrameReader:
    """
    Read frames at ascending indices without seeking for each one.

    cv2's POS_FRAMES seek on H.264 rewinds to the nearest keyframe and decodes
    forward, so it is far more expensive than stepping. grab() advances the
    decoder while skipping the color conversion and copy that read() pays for.
    Sampling 15fps out of 60fps source means most steps are +4 frames, where
    grabbing forward beats seeking; only large jumps between scattered indices
    are worth a real seek.
    """

    SEEK_GAP_FRAMES = 60

    def __init__(self, cap, seek_gap=SEEK_GAP_FRAMES):
        self._cap = cap
        self._seek_gap = seek_gap
        self._next_index = None  # index the decoder will return next, if known

    def read(self, frame_index):
        if (
            self._next_index is None
            or frame_index < self._next_index
            or frame_index - self._next_index > self._seek_gap
        ):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._next_index = frame_index

        while self._next_index < frame_index:
            if not self._cap.grab():
                self._next_index = None
                return None
            self._next_index += 1

        ok, frame = self._cap.read()
        if not ok:
            self._next_index = None
            return None

        self._next_index = frame_index + 1
        return frame
