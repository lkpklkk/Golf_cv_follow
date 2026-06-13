from __future__ import annotations

from collections import deque

import numpy as np


class ActionSequenceBuffer:
    def __init__(
        self,
        sequence_length,
        target_fps,
        classify_stride_sec=0.25,
        max_gap_sec=0.5,
    ):
        self.sequence_length = int(sequence_length)
        self.target_fps = float(target_fps)
        self.classify_stride_sec = float(classify_stride_sec)
        self.max_gap_sec = float(max_gap_sec)
        self.window_sec = self.sequence_length / self.target_fps
        self._samples = deque()
        self._target_id = None
        self._last_prediction_time = None
        self.frame_width = None
        self.frame_height = None

    def reset(self):
        self._samples.clear()
        self._target_id = None
        self._last_prediction_time = None
        self.frame_width = None
        self.frame_height = None

    def add(self, target_id, timestamp, keypoints, frame_width, frame_height):
        if target_id is None:
            self.reset()
            return None

        timestamp = float(timestamp)
        if self._target_id != target_id:
            self.reset()
            self._target_id = target_id

        # Person temporarily undetected: pause without resetting the history.
        # The timestamp-gap check below handles genuine long disappearances.
        if keypoints is None:
            return None

        if self._samples and timestamp - self._samples[-1][0] > self.max_gap_sec:
            self.reset()
            self._target_id = target_id

        sequence = np.asarray(keypoints, dtype=np.float32)
        if sequence.shape != (17, 3):
            self.reset()
            return None

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self._samples.append((timestamp, sequence.copy()))
        oldest_needed = timestamp - self.window_sec - self.max_gap_sec
        while self._samples and self._samples[0][0] < oldest_needed:
            self._samples.popleft()

        if self._last_prediction_time is not None:
            if timestamp - self._last_prediction_time < self.classify_stride_sec:
                return None

        resampled = self._resample(timestamp)
        if resampled is not None:
            self._last_prediction_time = timestamp
        return resampled

    def _resample(self, end_time):
        start_time = end_time - (self.sequence_length - 1) / self.target_fps
        if not self._samples or self._samples[0][0] > start_time:
            return None

        timestamps = np.asarray(
            [sample[0] for sample in self._samples], dtype=np.float64
        )
        poses = [sample[1] for sample in self._samples]
        targets = start_time + np.arange(self.sequence_length) / self.target_fps
        indices = np.searchsorted(timestamps, targets)
        output = []
        for target, right in zip(targets, indices):
            candidates = []
            if right < len(timestamps):
                candidates.append(right)
            if right > 0:
                candidates.append(right - 1)
            if not candidates:
                return None
            selected = min(
                candidates, key=lambda index: abs(timestamps[index] - target)
            )
            if abs(timestamps[selected] - target) > self.max_gap_sec:
                return None
            output.append(poses[selected])
        return np.asarray(output, dtype=np.float32)
