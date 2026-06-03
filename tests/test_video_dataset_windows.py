import copy
import unittest
from pathlib import Path

from video_test_runner import (
    create_windows,
    load_dataset_config,
    sample_frame_timestamps,
)


class DatasetWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_dataset_config(Path("dataset_config.yaml"))

    def test_unlabeled_regions_are_excluded_when_disabled(self):
        config = copy.deepcopy(self.config)
        config["windowing"]["include_unlabeled_as_other"] = False
        intervals = [
            {"label": "swing", "start_sec": 4.0, "end_sec": 5.0},
        ]

        windows = create_windows(10.0, intervals, config)

        self.assertTrue(windows)
        self.assertTrue(all(window.source_interval is not None for window in windows))
        self.assertFalse(
            any(window.start_sec < 1.0 or window.start_sec > 5.0 for window in windows)
        )

    def test_explicit_other_interval_still_generates_other_windows(self):
        config = copy.deepcopy(self.config)
        config["windowing"]["include_unlabeled_as_other"] = False
        intervals = [
            {"label": "other", "start_sec": 2.0, "end_sec": 5.0},
        ]

        windows = create_windows(8.0, intervals, config)

        self.assertTrue(windows)
        self.assertTrue(all(window.label_name == "other" for window in windows))
        self.assertTrue(all(window.source_interval["label"] == "other" for window in windows))

    def test_unlabeled_regions_can_be_enabled_for_backward_compatibility(self):
        config = copy.deepcopy(self.config)
        config["windowing"]["include_unlabeled_as_other"] = True

        windows = create_windows(5.0, [], config)

        self.assertTrue(windows)
        self.assertTrue(all(window.label_name == "other" for window in windows))
        self.assertTrue(all(window.source_interval is None for window in windows))

    def test_timestamp_jitter_is_deterministic_ordered_and_selects_different_frames(self):
        jittered_config = copy.deepcopy(self.config)
        jittered_config["sampling"]["timestamp_jitter_ratio"] = 0.35
        regular_config = copy.deepcopy(self.config)
        regular_config["sampling"]["timestamp_jitter_ratio"] = 0.0

        jittered_indices, jittered_timestamps = sample_frame_timestamps(
            start_sec=10.0,
            source_fps=60.0,
            total_frames=10_000,
            dataset_config=jittered_config,
            seed=123,
        )
        repeated_indices, repeated_timestamps = sample_frame_timestamps(
            start_sec=10.0,
            source_fps=60.0,
            total_frames=10_000,
            dataset_config=jittered_config,
            seed=123,
        )
        regular_indices, regular_timestamps = sample_frame_timestamps(
            start_sec=10.0,
            source_fps=60.0,
            total_frames=10_000,
            dataset_config=regular_config,
            seed=123,
        )

        self.assertEqual(jittered_indices, repeated_indices)
        self.assertEqual(jittered_timestamps, repeated_timestamps)
        self.assertTrue(
            all(a < b for a, b in zip(jittered_timestamps, jittered_timestamps[1:]))
        )
        self.assertAlmostEqual(jittered_timestamps[0], regular_timestamps[0], places=6)
        self.assertAlmostEqual(jittered_timestamps[-1], regular_timestamps[-1], places=6)
        self.assertNotEqual(jittered_indices, regular_indices)


if __name__ == "__main__":
    unittest.main()
