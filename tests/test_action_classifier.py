import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from action.classifier import ActionSequenceClassifier
from action.live_buffer import ActionSequenceBuffer
from action.preprocessing import FEATURE_NAMES, horizontal_flip_sequence, preprocess_pose_sequence
from action.review_export import select_false_positive_samples
from action.training import (
    calibrate_class_threshold,
    grouped_stratified_split,
    make_weighted_sampler,
    temporal_jitter_sequence,
)


LABELS = {"other": 0, "walk": 1, "swing": 2}
MODEL_CONFIG = {
    "sequence_length": 30,
    "keypoint_count": 17,
    "channels": [16, 24],
    "kernel_size": 3,
    "pool_size": 2,
    "dropout": 0.0,
    "hidden_dim": 16,
}
FEATURE_CONFIG = {"confidence_threshold": 0.25}


def sample_pose(sequence_length=30, width=100, height=200):
    pose = np.zeros((sequence_length, 17, 3), dtype=np.float32)
    for frame in range(sequence_length):
        pose[frame, :, 0] = 20 + frame
        pose[frame, :, 1] = 50 + np.arange(17)
        pose[frame, :, 2] = 1.0
    pose[:, 5, :2] = [40, 60]
    pose[:, 6, :2] = [60, 60]
    pose[:, 11, :2] = [42, 100]
    pose[:, 12, :2] = [58, 100]
    return pose


class PreprocessingTests(unittest.TestCase):
    def test_normalization_masking_and_velocity(self):
        pose = sample_pose()
        pose[0, 0] = [90, 180, 0.1]
        features = preprocess_pose_sequence(pose, 100, 200, confidence_threshold=0.25)
        self.assertEqual(features.shape, (30, 17, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(features[0, 0, :9], np.zeros(9, dtype=np.float32))
        self.assertAlmostEqual(float(features[0, 5, 0]), 0.4, places=5)
        self.assertAlmostEqual(float(features[0, 5, 1]), 0.3, places=5)
        self.assertAlmostEqual(float(features[1, 0, 5]), 0.0, places=5)
        self.assertAlmostEqual(float(features[1, 1, 5]), 0.01, places=5)
        hip_midpoint_x = float((features[0, 11, 2] + features[0, 12, 2]) * 0.5)
        self.assertAlmostEqual(hip_midpoint_x, 0.0, places=5)

    def test_swing_features_capture_wrist_motion_and_rotation(self):
        pose = sample_pose(sequence_length=2)
        pose[:, 9, :2] = [[40, 80], [70, 70]]
        pose[:, 10, :2] = [[60, 80], [80, 90]]
        pose[1, 5, :2] = [45, 55]
        pose[1, 6, :2] = [55, 65]
        features = preprocess_pose_sequence(pose, 100, 200)
        feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        self.assertGreater(
            float(features[1, 0, feature_index["left_wrist_relative_speed"]]),
            0.0,
        )
        self.assertNotEqual(
            float(features[1, 0, feature_index["shoulder_rotation_delta"]]),
            0.0,
        )
        self.assertGreater(
            float(features[0, 0, feature_index["left_wrist_shoulder_distance"]]),
            0.0,
        )

    def test_horizontal_flip_swaps_left_and_right(self):
        pose = sample_pose(sequence_length=1)
        pose[0, 5, 0] = 10
        pose[0, 6, 0] = 80
        flipped = horizontal_flip_sequence(pose, 100)
        self.assertEqual(float(flipped[0, 5, 0]), 20.0)
        self.assertEqual(float(flipped[0, 6, 0]), 90.0)


class ClassifierTests(unittest.TestCase):
    def test_model_supports_different_sequence_lengths(self):
        for length in (20, 30):
            config = dict(MODEL_CONFIG)
            config["sequence_length"] = length
            classifier = ActionSequenceClassifier(config, LABELS, FEATURE_CONFIG, device="cpu")
            features = torch.zeros(2, length, 17, len(FEATURE_NAMES))
            logits = classifier.model(features)
            self.assertEqual(tuple(logits.shape), (2, 3))

    def test_checkpoint_round_trip_and_confidence_fallback(self):
        classifier = ActionSequenceClassifier(
            MODEL_CONFIG,
            LABELS,
            FEATURE_CONFIG,
            decision_config={"class_thresholds": {"swing": 0.85}},
            device="cpu",
        )
        pose = sample_pose()
        before = classifier.predict_proba(pose, 100, 200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            classifier.save(path)
            loaded = ActionSequenceClassifier.load(path, device="cpu")
            after = loaded.predict_proba(pose, 100, 200)
        for label in LABELS:
            self.assertAlmostEqual(before[label], after[label], places=6)
        self.assertEqual(loaded.decision_config["class_thresholds"]["swing"], 0.85)
        prediction = classifier.classify(
            pose,
            100,
            200,
            confidence_threshold=1.1,
            fallback_label="other",
        )
        self.assertEqual(prediction.label, "other")
        self.assertFalse(prediction.is_confident)

    def test_swing_specific_threshold_falls_back_to_other(self):
        classifier = ActionSequenceClassifier(
            MODEL_CONFIG,
            LABELS,
            FEATURE_CONFIG,
            decision_config={
                "confidence_threshold": 0.60,
                "fallback_label": "other",
                "class_thresholds": {"swing": 0.80},
            },
            device="cpu",
        )
        prediction = classifier.classify_probabilities(
            {"other": 0.10, "walk": 0.15, "swing": 0.75}
        )
        self.assertEqual(prediction.raw_label, "swing")
        self.assertEqual(prediction.label, "other")
        self.assertFalse(prediction.is_confident)


class TrainingUtilityTests(unittest.TestCase):
    def test_grouped_split_has_no_video_overlap_and_all_classes(self):
        video_ids = []
        labels = []
        for video_index in range(15):
            video = f"video_{video_index}"
            for label in range(3):
                video_ids.extend([video] * (label + 1))
                labels.extend([label] * (label + 1))
        splits, videos = grouped_stratified_split(
            np.asarray(video_ids),
            np.asarray(labels),
            seed=7,
        )
        self.assertFalse(set(videos["train"]) & set(videos["validation"]))
        self.assertFalse(set(videos["train"]) & set(videos["test"]))
        self.assertFalse(set(videos["validation"]) & set(videos["test"]))
        for indices in splits.values():
            self.assertEqual(set(np.asarray(labels)[indices]), {0, 1, 2})

    def test_weighted_sampler_balances_exposure(self):
        labels = np.asarray([0] * 100 + [1] * 10 + [2] * 5)
        sampler = make_weighted_sampler(labels)
        sampled_labels = labels[list(iter(sampler))]
        counts = np.bincount(sampled_labels, minlength=3)
        self.assertLess(float(counts.max()) / max(float(counts.min()), 1.0), 3.0)

    def test_temporal_jitter_preserves_length_and_order(self):
        sequence = np.arange(10, dtype=np.float32)[:, None, None]
        jittered = temporal_jitter_sequence(
            sequence,
            max_jitter_frames=1,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(jittered.shape, sequence.shape)
        values = jittered[:, 0, 0]
        self.assertTrue(np.all(np.diff(values) >= 0))
        self.assertTrue(np.all(np.abs(values - np.arange(10)) <= 1))

    def test_calibration_selects_threshold_meeting_precision(self):
        targets = np.asarray([2, 2, 0, 0], dtype=np.int64)
        probabilities = np.asarray(
            [
                [0.02, 0.03, 0.95],
                [0.05, 0.10, 0.85],
                [0.10, 0.10, 0.80],
                [0.80, 0.10, 0.10],
            ],
            dtype=np.float32,
        )
        calibration = calibrate_class_threshold(
            targets,
            probabilities,
            LABELS,
            label="swing",
            minimum_precision=1.0,
            min_threshold=0.60,
            max_threshold=0.90,
            step=0.05,
        )
        self.assertGreater(calibration["selected"]["threshold"], 0.80)
        self.assertEqual(calibration["selected"]["precision"], 1.0)

    def test_false_positive_selection_suppresses_overlapping_windows(self):
        metadata = [
            {"video": "a.mp4", "start_sec": 1.0, "end_sec": 3.0},
            {"video": "a.mp4", "start_sec": 1.2, "end_sec": 3.2},
            {"video": "b.mp4", "start_sec": 5.0, "end_sec": 7.0},
        ]
        selected = select_false_positive_samples(
            global_indices=np.asarray([0, 1, 2]),
            targets=np.asarray([0, 0, 0]),
            predictions=np.asarray([2, 2, 2]),
            probabilities=np.asarray(
                [[0.01, 0.01, 0.98], [0.02, 0.01, 0.97], [0.03, 0.01, 0.96]]
            ),
            metadata=metadata,
            label_to_id=LABELS,
            overlap_suppression_sec=1.0,
        )
        self.assertEqual([item["global_index"] for item in selected], [0, 2])


class LiveBufferTests(unittest.TestCase):
    def test_buffer_resamples_and_resets_on_target_change(self):
        buffer = ActionSequenceBuffer(
            sequence_length=5,
            target_fps=2,
            classify_stride_sec=0.0,
            max_gap_sec=0.6,
        )
        pose = sample_pose(sequence_length=1)[0]
        result = None
        for index in range(5):
            result = buffer.add(1, index * 0.5, pose + index, 100, 200)
        self.assertEqual(result.shape, (5, 17, 3))
        self.assertIsNone(buffer.add(2, 2.5, pose, 100, 200))


if __name__ == "__main__":
    unittest.main()
