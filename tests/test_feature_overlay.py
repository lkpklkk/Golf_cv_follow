"""Tests for the debug-video feature panel: stats, aggregation, rendering."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from action.feature_stats import (
    compute_feature_stats,
    load_feature_stats,
    save_feature_stats,
)
from action.preprocessing import (
    FEATURE_NAMES,
    PER_KEYPOINT_FEATURE_COUNT,
    frame_feature_vector,
    keypoint_valid_mask,
    preprocess_pose_sequence,
)
from ui.feature_overlay import _fit_text, _layout, draw_feature_panel, short_name


def _pose_sequence(frames=6, seed=0):
    """A plausible moving pose: all keypoints visible, drifting each frame."""
    rng = np.random.default_rng(seed)
    sequence = np.zeros((frames, 17, 3), dtype=np.float32)
    base = rng.uniform(100, 500, size=(17, 2)).astype(np.float32)
    for index in range(frames):
        sequence[index, :, :2] = base + index * 4.0
        sequence[index, :, 2] = 0.9
    return sequence


class FeatureStatsTest(unittest.TestCase):
    def test_covers_every_feature_with_usable_scale(self):
        stats = _stats_from_sequences()
        self.assertEqual(list(stats["features"]), list(FEATURE_NAMES))
        for name, entry in stats["features"].items():
            self.assertGreater(entry["scale"], 0.0, name)
            self.assertLessEqual(entry["min"], entry["max"], name)

    def test_signed_flag_matches_observed_minimum(self):
        stats = _stats_from_sequences()
        for name, entry in stats["features"].items():
            self.assertEqual(entry["signed"], entry["min"] < -1e-6, name)
        # confidence is a probability and can never go negative.
        self.assertFalse(stats["features"]["confidence"]["signed"])

    def test_stats_describe_the_displayed_values_not_the_raw_spread(self):
        """
        The panel draws the per-keypoint half as a mean of absolute values, so
        its ranges must be measured on that. Measuring the raw per-keypoint
        column instead marks non-negative rows as signed (drawing a bipolar bar
        that can only ever fill one side) and oversizes their scale.
        """
        stats = _stats_from_sequences()
        for index in range(PER_KEYPOINT_FEATURE_COUNT):
            name = FEATURE_NAMES[index]
            entry = stats["features"][name]
            self.assertGreaterEqual(entry["min"], 0.0, name)
            self.assertFalse(entry["signed"], name)

        # And the values fed to the panel must fall inside those measured bounds.
        sequence = _pose_sequence(seed=1)
        values = frame_feature_vector(
            preprocess_pose_sequence(sequence, 1920, 1080)[-1],
            keypoint_valid_mask(sequence[-1]),
        )
        for index in range(PER_KEYPOINT_FEATURE_COUNT):
            entry = stats["features"][FEATURE_NAMES[index]]
            self.assertLessEqual(float(values[index]), entry["max"] + 1e-5)

    def test_scale_ignores_a_lone_outlier(self):
        """p99 scaling exists so one extreme frame cannot flatten every bar."""
        column = np.concatenate([np.full(999, 1.0), np.array([500.0])])
        self.assertLess(float(np.percentile(np.abs(column), 99.0)), 2.0)
        self.assertEqual(column.max(), 500.0)

    def test_json_round_trip(self):
        stats = _stats_from_sequences()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "stats.json"
            save_feature_stats(stats, path)
            self.assertEqual(load_feature_stats(path), json.loads(path.read_text()))

    def test_missing_file_loads_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_feature_stats(Path(tmp) / "absent.json"))

    def test_missing_samples_file_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compute_feature_stats(Path(tmp) / "samples.npz")


class FrameFeatureVectorTest(unittest.TestCase):
    def setUp(self):
        self.sequence = _pose_sequence()
        self.features = preprocess_pose_sequence(self.sequence, 1920, 1080)

    def test_returns_one_value_per_feature(self):
        values = frame_feature_vector(self.features[-1])
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))

    def test_global_features_are_passed_through_unchanged(self):
        """Indices 9-18 are per-frame scalars repeated across all 17 keypoints,
        so the panel must show the exact model input, not an average of it."""
        frame = self.features[-1]
        values = frame_feature_vector(frame)
        for index in range(PER_KEYPOINT_FEATURE_COUNT, len(FEATURE_NAMES)):
            self.assertAlmostEqual(
                float(values[index]), float(frame[0, index]), places=6
            )
            # Verify the repetition assumption the pass-through relies on.
            self.assertTrue(np.allclose(frame[:, index], frame[0, index]))

    def test_per_keypoint_features_average_absolute_values(self):
        frame = self.features[-1]
        valid = keypoint_valid_mask(self.sequence[-1])
        values = frame_feature_vector(frame, valid)
        for index in range(PER_KEYPOINT_FEATURE_COUNT):
            expected = np.abs(frame[valid, index]).mean()
            self.assertAlmostEqual(float(values[index]), float(expected), places=5)

    def test_invalid_keypoints_are_excluded_from_the_average(self):
        """Invalid rows are zeroed upstream, so averaging over all 17 would
        drag the value toward zero in proportion to how many joints were lost."""
        frame = self.features[-1].copy()
        valid = np.zeros(17, dtype=bool)
        valid[:2] = True
        values = frame_feature_vector(frame, valid)
        expected = np.abs(frame[:2, 0]).mean()
        self.assertAlmostEqual(float(values[0]), float(expected), places=5)
        self.assertNotAlmostEqual(
            float(values[0]), float(np.abs(frame[:, 0]).mean()), places=5
        )

    def test_no_valid_keypoints_returns_zeros_for_the_per_keypoint_half(self):
        values = frame_feature_vector(self.features[-1], np.zeros(17, dtype=bool))
        self.assertTrue(np.all(values[:PER_KEYPOINT_FEATURE_COUNT] == 0.0))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            frame_feature_vector(np.zeros((17, 3), dtype=np.float32))


class DrawFeaturePanelTest(unittest.TestCase):
    def setUp(self):
        self.stats = _stats_from_sequences()
        self.values = np.linspace(-1.0, 1.0, len(FEATURE_NAMES)).astype(np.float32)

    def _frame(self, width=1280, height=720):
        return np.full((height, width, 3), 40, dtype=np.uint8)

    def test_draws_only_inside_the_panel(self):
        frame = self._frame()
        original = frame.copy()
        draw_feature_panel(frame, self.values, self.stats)
        panel_left = frame.shape[1] - max(180, int(frame.shape[1] * 0.26))
        # Left of the panel must be untouched; the panel itself must change.
        self.assertTrue(np.array_equal(frame[:, :panel_left], original[:, :panel_left]))
        self.assertFalse(np.array_equal(frame[:, panel_left:], original[:, panel_left:]))

    def test_returns_same_frame_and_shape(self):
        frame = self._frame()
        result = draw_feature_panel(frame, self.values, self.stats)
        self.assertIs(result, frame)
        self.assertEqual(result.shape, (720, 1280, 3))

    def test_survives_edge_case_inputs(self):
        # Each of these has crashed a naive implementation: no stats at all, a
        # stats dict missing entries, values far past the measured range,
        # all-zero values, and the no-data path.
        for kwargs in (
            {"stats": None},
            {"stats": {"features": {}}},
            {"stats": self.stats, "no_data": True},
            {"stats": self.stats, "title": "swing @1.25s"},
        ):
            with self.subTest(kwargs=sorted(kwargs)):
                draw_feature_panel(self._frame(), self.values, **kwargs)

        draw_feature_panel(self._frame(), np.zeros(len(FEATURE_NAMES)), self.stats)
        draw_feature_panel(
            self._frame(), np.full(len(FEATURE_NAMES), 1e6), self.stats
        )
        draw_feature_panel(
            self._frame(), np.full(len(FEATURE_NAMES), -1e6), self.stats
        )

    def test_handles_small_and_large_frames(self):
        for width, height in ((320, 240), (3840, 2160)):
            with self.subTest(size=(width, height)):
                frame = self._frame(width, height)
                draw_feature_panel(frame, self.values, self.stats)
                self.assertEqual(frame.shape, (height, width, 3))

    def test_label_and_value_never_collide(self):
        """
        Regression: font was sized from row height alone, so a portrait frame
        (tall rows, narrow panel) ran the label straight into its value —
        "L_wrist_dist+0.888" with no gap.
        """
        for width, height in ((1080, 1920), (320, 240), (1280, 720), (3840, 2160)):
            with self.subTest(size=(width, height)):
                layout = _layout(width, height, has_title=True)
                widest = max((short_name(n) for n in FEATURE_NAMES), key=len)
                (label_width, _), _ = cv2.getTextSize(
                    widest,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    layout["font_scale"],
                    layout["thickness"],
                )
                (value_width, _), _ = cv2.getTextSize(
                    "-0.000",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    layout["font_scale"],
                    layout["thickness"],
                )
                self.assertLessEqual(label_width + value_width, layout["bar_width"])

    def test_bar_clears_the_label_descenders(self):
        """Regression: the bar sat a fixed fraction below the text baseline and
        sliced through the descenders of g, p and y."""
        for width, height in ((1080, 1920), (1280, 720), (320, 240)):
            with self.subTest(size=(width, height)):
                layout = _layout(width, height, has_title=False)
                (_, _), descent = cv2.getTextSize(
                    "gpy",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    layout["font_scale"],
                    layout["thickness"],
                )
                self.assertGreater(layout["label_gap"], descent)
                # And the bar must clear the next row's ascenders too.
                stride = layout["row_height"] + layout["bar_height"] + layout["row_gap"]
                bar_bottom = layout["label_gap"] + layout["bar_height"]
                self.assertLess(bar_bottom, stride - layout["row_height"] + 1)

    def test_long_title_is_trimmed_to_the_panel(self):
        frame = self._frame(1080, 1920)
        draw_feature_panel(
            frame, self.values, self.stats, title="features (last classified window)"
        )
        layout = _layout(1080, 1920, has_title=True)
        trimmed = _fit_text("features (last classified window)", layout)
        (text_width, _), _ = cv2.getTextSize(
            trimmed, cv2.FONT_HERSHEY_SIMPLEX, layout["font_scale"], layout["thickness"]
        )
        self.assertLessEqual(text_width, layout["bar_width"])

    def test_rejects_wrong_value_count(self):
        with self.assertRaises(ValueError):
            draw_feature_panel(self._frame(), np.zeros(5), self.stats)

    def test_short_names_stay_short(self):
        for name in FEATURE_NAMES:
            self.assertLessEqual(len(short_name(name)), 15, name)


def _stats_from_sequences():
    """Build real stats through compute_feature_stats() via a temp samples.npz."""
    sequences = np.stack([_pose_sequence(seed=seed) for seed in range(4)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "samples.npz"
        np.savez_compressed(
            path,
            X=sequences,
            y=np.zeros(len(sequences), dtype=np.int64),
            video_ids=np.asarray(["a.mp4"] * len(sequences)),
            start_sec=np.zeros(len(sequences), dtype=np.float32),
            end_sec=np.ones(len(sequences), dtype=np.float32),
            frame_width=np.full(len(sequences), 1920, dtype=np.int32),
            frame_height=np.full(len(sequences), 1080, dtype=np.int32),
        )
        return compute_feature_stats(path)


if __name__ == "__main__":
    unittest.main()
