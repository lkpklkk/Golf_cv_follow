"""Tests for the swing-threshold sweep replay."""

from __future__ import annotations

import unittest

from action.classifier import ActionSequenceClassifier
from action.sweep_swing_threshold import (
    build_thresholds,
    swing_events_from_trace,
)


def _classifier(confidence_threshold=0.75, swing_threshold=0.79):
    """A real classifier, built small; only classify_probabilities is used."""
    return ActionSequenceClassifier(
        model_config={
            "sequence_length": 30,
            "keypoint_count": 17,
            "channels": [8],
            "hidden_dim": 8,
        },
        label_to_id={"other": 0, "swing": 1},
        decision_config={
            "confidence_threshold": confidence_threshold,
            "fallback_label": "other",
            "class_thresholds": {"swing": swing_threshold},
        },
        device="cpu",
    )


def _trace(swing_probabilities):
    return [
        (float(index), {"other": 1.0 - p, "swing": p})
        for index, p in enumerate(swing_probabilities)
    ]


class BuildThresholdsTest(unittest.TestCase):
    def test_inclusive_range_without_float_drift(self):
        self.assertEqual(
            build_thresholds(0.60, 0.95, 0.05),
            [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        )

    def test_single_value_range(self):
        self.assertEqual(build_thresholds(0.8, 0.8, 0.05), [0.8])

    def test_rejects_bad_arguments(self):
        with self.assertRaises(SystemExit):
            build_thresholds(0.6, 0.9, 0.0)
        with self.assertRaises(SystemExit):
            build_thresholds(0.9, 0.6, 0.05)


class SwingEventsFromTraceTest(unittest.TestCase):
    def setUp(self):
        self.classifier = _classifier()

    def _events(self, swing_probabilities, threshold):
        return swing_events_from_trace(
            _trace(swing_probabilities),
            self.classifier,
            confidence_threshold=0.75,
            fallback_label="other",
            swing_threshold=threshold,
        )

    def test_only_rising_edges_count(self):
        # One sustained burst is one event, not four.
        self.assertEqual(self._events([0.1, 0.95, 0.96, 0.97, 0.98, 0.1], 0.90), [1.0])

    def test_separate_bursts_are_separate_events(self):
        self.assertEqual(
            self._events([0.95, 0.1, 0.95, 0.1, 0.95], 0.90), [0.0, 2.0, 4.0]
        )

    def test_leading_edge_at_index_zero(self):
        self.assertEqual(self._events([0.95, 0.95], 0.90), [0.0])

    def test_raising_threshold_removes_events(self):
        probabilities = [0.1, 0.82, 0.1, 0.95, 0.1]
        self.assertEqual(self._events(probabilities, 0.80), [1.0, 3.0])
        self.assertEqual(self._events(probabilities, 0.90), [3.0])
        self.assertEqual(self._events(probabilities, 0.99), [])

    def test_config_confidence_threshold_is_a_floor(self):
        # Effective gate is max(confidence_threshold, swing threshold), so a
        # swing threshold below 0.75 cannot admit a 0.70 prediction.
        self.assertEqual(self._events([0.70], 0.60), [])
        self.assertEqual(self._events([0.76], 0.60), [0.0])

    def test_argmax_must_be_swing(self):
        # p(swing)=0.45 clears no gate and is not the top class anyway.
        trace = [(0.0, {"other": 0.55, "swing": 0.45})]
        self.assertEqual(
            swing_events_from_trace(
                trace, self.classifier, 0.40, "other", 0.40
            ),
            [],
        )

    def test_replay_matches_live_rising_edge_logic(self):
        """
        The replay must agree with RealisticVideoRun._update_action, which
        tracks `is_confident and label == swing` against the previous
        prediction. Reimplemented here from that method so the sweep is
        checked against the live rule rather than against itself.
        """
        probabilities = [0.1, 0.95, 0.93, 0.2, 0.99, 0.5, 0.85, 0.95]
        threshold = 0.90

        live_events = []
        previous = None
        for timestamp, probs in _trace(probabilities):
            prediction = self.classifier.classify_probabilities(
                probs,
                confidence_threshold=0.75,
                fallback_label="other",
                class_thresholds={"swing": threshold},
            )
            if (
                prediction.is_confident
                and prediction.label == "swing"
                and (
                    previous is None
                    or not previous.is_confident
                    or previous.label != "swing"
                )
            ):
                live_events.append(timestamp)
            previous = prediction

        self.assertEqual(self._events(probabilities, threshold), live_events)


if __name__ == "__main__":
    unittest.main()
