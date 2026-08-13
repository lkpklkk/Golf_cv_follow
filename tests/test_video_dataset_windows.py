import copy
import unittest
from pathlib import Path

from action.generate_dataset import (
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
        # Interval is 2.0 s, not the 1.0 s this test used originally. A window is
        # 2.0 s, so a 1.0 s interval can reach at most 0.50 overlap and produces
        # no positives at all once swing_overlap_threshold goes to 0.70; the
        # ignore band then drops the remainder and the assertion below fails on
        # an empty list. See test_short_interval_yields_no_positives.
        intervals = [
            {"label": "swing", "start_sec": 4.0, "end_sec": 6.0},
        ]

        windows = create_windows(10.0, intervals, config)

        self.assertTrue(windows)
        self.assertTrue(all(window.source_interval is not None for window in windows))
        self.assertFalse(
            any(window.start_sec < 2.0 or window.start_sec > 6.0 for window in windows)
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

    # ------------------------------------------------------------------
    # Ignore band (Stage 2). These set the thresholds explicitly rather than
    # relying on dataset_config.yaml, so they hold both before and after the
    # Stage 3 config edit.
    # ------------------------------------------------------------------

    def _ignore_band_config(self):
        config = copy.deepcopy(self.config)
        config["windowing"]["include_unlabeled_as_other"] = True
        config["windowing"]["swing_overlap_threshold"] = 0.70
        config["windowing"]["swing_ignore_below"] = 0.10
        config["windowing"]["max_background_windows"] = None
        return config

    @staticmethod
    def _overlap_ratio(window, interval, window_sec=2.0):
        overlap = max(
            0.0,
            min(window.end_sec, interval["end_sec"])
            - max(window.start_sec, interval["start_sec"]),
        )
        return overlap / window_sec

    def test_ignore_band_emits_nothing(self):
        """
        The regression test for the Stage 2 hazard: with a raised threshold and
        include_unlabeled_as_other on, a partially-overlapping swing window must
        vanish, not fall through to `other`. Training it as `other` would teach
        the model to suppress real swings.
        """
        config = self._ignore_band_config()
        interval = {"label": "swing", "start_sec": 4.0, "end_sec": 5.0}

        windows = create_windows(20.0, [interval], config)

        self.assertTrue(windows)  # background windows still exist
        for window in windows:
            ratio = self._overlap_ratio(window, interval)
            self.assertFalse(
                0.10 < ratio < 0.70,
                f"ambiguous window emitted as {window.label_name} at "
                f"{window.start_sec:.2f}s (ratio {ratio:.3f})",
            )
        self.assertNotIn("swing", {window.label_name for window in windows})

    def test_low_overlap_window_is_background(self):
        config = self._ignore_band_config()
        interval = {"label": "swing", "start_sec": 4.0, "end_sec": 5.0}

        windows = create_windows(20.0, [interval], config)
        grazing = [
            window
            for window in windows
            if 0.0 < self._overlap_ratio(window, interval) <= 0.10
        ]

        self.assertTrue(grazing)
        self.assertTrue(all(window.label_name == "other" for window in grazing))

    def test_short_interval_yields_no_positives(self):
        config = self._ignore_band_config()
        # 1.0 s of swing in a 2.0 s window caps the ratio at 0.50, below 0.70.
        intervals = [{"label": "swing", "start_sec": 4.0, "end_sec": 5.0}]

        windows = create_windows(20.0, intervals, config)

        self.assertEqual([], [w for w in windows if w.label_name == "swing"])

    def test_ignore_band_is_inert_when_keys_absent(self):
        config = copy.deepcopy(self.config)
        config["windowing"].pop("swing_ignore_below", None)
        config["windowing"]["swing_overlap_threshold"] = 0.30
        intervals = [{"label": "swing", "start_sec": 4.0, "end_sec": 6.0}]

        windows = create_windows(10.0, intervals, config)

        self.assertTrue(any(window.label_name == "swing" for window in windows))

    def test_background_windows_are_capped_and_deterministic(self):
        config = self._ignore_band_config()
        config["windowing"]["max_background_windows"] = 50

        uncapped = create_windows(600.0, [], self._ignore_band_config())
        first = create_windows(600.0, [], config)
        second = create_windows(600.0, [], copy.deepcopy(config))

        self.assertGreater(len(uncapped), 50)
        self.assertLessEqual(len(first), 50)
        self.assertEqual(
            [w.start_sec for w in first], [w.start_sec for w in second]
        )

    def test_background_cap_never_drops_interval_windows(self):
        config = self._ignore_band_config()
        config["windowing"]["max_background_windows"] = 1
        intervals = [{"label": "swing", "start_sec": 100.0, "end_sec": 103.0}]

        windows = create_windows(600.0, intervals, config)
        swing_windows = [w for w in windows if w.label_name == "swing"]

        self.assertTrue(swing_windows)

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
