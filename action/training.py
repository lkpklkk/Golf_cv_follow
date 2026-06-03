from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from action.classifier import ActionSequenceClassifier
from action.preprocessing import horizontal_flip_sequence, preprocess_pose_sequence


REQUIRED_ARRAYS = {
    "X",
    "y",
    "video_ids",
    "frame_width",
    "frame_height",
}


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        arrays,
        indices,
        confidence_threshold,
        horizontal_flip_probability=0.0,
        temporal_jitter_probability=0.0,
        max_temporal_jitter_frames=1,
    ):
        self.x = arrays["X"]
        self.y = arrays["y"]
        self.frame_width = arrays["frame_width"]
        self.frame_height = arrays["frame_height"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.confidence_threshold = float(confidence_threshold)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.temporal_jitter_probability = float(temporal_jitter_probability)
        self.max_temporal_jitter_frames = int(max_temporal_jitter_frames)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        keypoints = self.x[index]
        width = int(self.frame_width[index])
        height = int(self.frame_height[index])
        if self.temporal_jitter_probability > 0.0:
            if random.random() < self.temporal_jitter_probability:
                keypoints = temporal_jitter_sequence(
                    keypoints,
                    max_jitter_frames=self.max_temporal_jitter_frames,
                )
        if self.horizontal_flip_probability > 0.0:
            if random.random() < self.horizontal_flip_probability:
                keypoints = horizontal_flip_sequence(keypoints, width)
        features = preprocess_pose_sequence(
            keypoints,
            frame_width=width,
            frame_height=height,
            confidence_threshold=self.confidence_threshold,
        )
        return torch.from_numpy(features), torch.tensor(int(self.y[index]), dtype=torch.long)


def temporal_jitter_sequence(keypoints, max_jitter_frames=1, rng=None):
    sequence = np.asarray(keypoints, dtype=np.float32)
    if len(sequence) <= 1 or int(max_jitter_frames) <= 0:
        return sequence.copy()

    rng = rng or np.random.default_rng()
    base_indices = np.arange(len(sequence), dtype=np.int64)
    offsets = rng.integers(
        -int(max_jitter_frames),
        int(max_jitter_frames) + 1,
        size=len(sequence),
    )
    jittered_indices = np.clip(base_indices + offsets, 0, len(sequence) - 1)
    jittered_indices = np.maximum.accumulate(jittered_indices)
    return sequence[jittered_indices].copy()


def load_action_config(path):
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def load_processed_arrays(path):
    data = np.load(path)
    missing = REQUIRED_ARRAYS - set(data.files)
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")

    arrays = {key: data[key] for key in data.files}
    sample_count = len(arrays["X"])
    for key in REQUIRED_ARRAYS - {"X"}:
        if len(arrays[key]) != sample_count:
            raise ValueError(
                f"Dataset array {key} has {len(arrays[key])} rows, expected {sample_count}"
            )
    if arrays["X"].ndim != 4 or arrays["X"].shape[2:] != (17, 3):
        raise ValueError(f"Expected X shape (N, T, 17, 3), got {arrays['X'].shape}")
    return arrays


def grouped_stratified_split(
    video_ids,
    labels,
    ratios=(0.70, 0.15, 0.15),
    seed=42,
):
    split_names = ("train", "validation", "test")
    if len(ratios) != len(split_names) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must contain train/validation/test values summing to 1")

    video_ids = np.asarray(video_ids)
    labels = np.asarray(labels, dtype=np.int64)
    classes = sorted(int(value) for value in np.unique(labels))
    videos = sorted(str(value) for value in np.unique(video_ids))
    group_counts = {}
    for video in videos:
        group_labels = labels[video_ids == video]
        group_counts[video] = np.asarray(
            [np.sum(group_labels == label) for label in classes],
            dtype=np.float64,
        )

    rng = random.Random(seed)
    rng.shuffle(videos)
    videos.sort(
        key=lambda video: (
            np.count_nonzero(group_counts[video]),
            group_counts[video].sum(),
            group_counts[video].max(),
        ),
        reverse=True,
    )

    total_counts = np.asarray([np.sum(labels == label) for label in classes], dtype=np.float64)
    total_samples = float(len(labels))
    target_counts = {
        name: total_counts * ratio for name, ratio in zip(split_names, ratios)
    }
    target_samples = {
        name: total_samples * ratio for name, ratio in zip(split_names, ratios)
    }
    assigned = {name: [] for name in split_names}
    assigned_counts = {
        name: np.zeros(len(classes), dtype=np.float64) for name in split_names
    }
    assigned_samples = {name: 0.0 for name in split_names}

    for video in videos:
        counts = group_counts[video]
        sample_count = counts.sum()
        best_split = None
        best_score = None
        for name in split_names:
            projected_counts = {
                split: assigned_counts[split] + (counts if split == name else 0.0)
                for split in split_names
            }
            projected_samples = {
                split: assigned_samples[split] + (sample_count if split == name else 0.0)
                for split in split_names
            }
            class_error = np.mean(
                [
                    np.mean(
                        np.square(
                            (projected_counts[split] - target_counts[split])
                            / np.maximum(target_counts[split], 1.0)
                        )
                    )
                    for split in split_names
                ]
            )
            sample_error = np.mean(
                [
                    (
                        (projected_samples[split] - target_samples[split])
                        / max(target_samples[split], 1.0)
                    )
                    ** 2
                    for split in split_names
                ]
            )
            overfill_penalty = max(
                0.0,
                (projected_samples[name] - target_samples[name])
                / max(target_samples[name], 1.0),
            )
            score = class_error + 0.35 * sample_error + 2.0 * overfill_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_split = name

        assigned[best_split].append(video)
        assigned_counts[best_split] += counts
        assigned_samples[best_split] += sample_count

    _ensure_each_split_has_all_classes(
        assigned,
        group_counts,
        classes,
        assigned_counts,
        assigned_samples,
    )

    result = {}
    for name in split_names:
        mask = np.isin(video_ids, assigned[name])
        result[name] = np.flatnonzero(mask)
    return result, assigned


def _ensure_each_split_has_all_classes(
    assigned,
    group_counts,
    classes,
    assigned_counts,
    assigned_samples,
):
    for split_name in ("validation", "test", "train"):
        for class_index, _ in enumerate(classes):
            if assigned_counts[split_name][class_index] > 0:
                continue
            candidates = []
            for donor_name, donor_videos in assigned.items():
                if donor_name == split_name:
                    continue
                for video in donor_videos:
                    counts = group_counts[video]
                    donor_after = assigned_counts[donor_name] - counts
                    if counts[class_index] <= 0 or np.any(donor_after <= 0):
                        continue
                    candidates.append((counts.sum(), donor_name, video))
            if not candidates:
                raise ValueError(
                    f"Could not create grouped split containing class index {class_index}"
                )
            _, donor_name, video = min(candidates)
            counts = group_counts[video]
            assigned[donor_name].remove(video)
            assigned[split_name].append(video)
            assigned_counts[donor_name] -= counts
            assigned_counts[split_name] += counts
            assigned_samples[donor_name] -= counts.sum()
            assigned_samples[split_name] += counts.sum()


def make_weighted_sampler(labels):
    labels = np.asarray(labels, dtype=np.int64)
    counts = Counter(int(label) for label in labels)
    weights = np.asarray([1.0 / counts[int(label)] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def evaluate_classifier(classifier, loader):
    targets, probabilities = collect_classifier_outputs(classifier, loader)
    predictions = np.argmax(probabilities, axis=1)
    return classification_metrics(targets, predictions, classifier.label_to_id)


def collect_classifier_outputs(classifier, loader):
    classifier.model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for features, labels in loader:
            logits = classifier.model(features.to(classifier.device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            targets.extend(labels.tolist())
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
    )


def predictions_from_probabilities(
    probabilities,
    label_to_id,
    confidence_threshold=0.60,
    fallback_label="other",
    class_thresholds=None,
):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    raw_predictions = np.argmax(probabilities, axis=1)
    predictions = raw_predictions.copy()
    fallback_id = int(label_to_id[fallback_label])
    id_to_label = {int(value): key for key, value in label_to_id.items()}
    thresholds = class_thresholds or {}
    for index, raw_id in enumerate(raw_predictions):
        label = id_to_label[int(raw_id)]
        required_threshold = max(
            float(confidence_threshold),
            float(thresholds.get(label, 0.0)),
        )
        if float(probabilities[index, raw_id]) < required_threshold:
            predictions[index] = fallback_id
    return predictions


def evaluate_probabilities(
    targets,
    probabilities,
    label_to_id,
    confidence_threshold=0.60,
    fallback_label="other",
    class_thresholds=None,
):
    predictions = predictions_from_probabilities(
        probabilities,
        label_to_id=label_to_id,
        confidence_threshold=confidence_threshold,
        fallback_label=fallback_label,
        class_thresholds=class_thresholds,
    )
    return classification_metrics(targets, predictions, label_to_id), predictions


def calibrate_class_threshold(
    targets,
    probabilities,
    label_to_id,
    label,
    confidence_threshold=0.60,
    fallback_label="other",
    minimum_precision=0.65,
    min_threshold=0.60,
    max_threshold=0.99,
    step=0.01,
):
    label_id = int(label_to_id[label])
    candidates = np.arange(
        float(min_threshold),
        float(max_threshold) + float(step) * 0.5,
        float(step),
    )
    results = []
    for threshold in candidates:
        metrics, _ = evaluate_probabilities(
            targets,
            probabilities,
            label_to_id=label_to_id,
            confidence_threshold=confidence_threshold,
            fallback_label=fallback_label,
            class_thresholds={label: float(threshold)},
        )
        class_metrics = metrics["per_class"][label]
        results.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": float(class_metrics["precision"]),
                "recall": float(class_metrics["recall"]),
                "f1": float(class_metrics["f1"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )

    meeting_precision = [
        result for result in results if result["precision"] >= float(minimum_precision)
    ]
    if meeting_precision:
        selected = max(
            meeting_precision,
            key=lambda result: (
                result["recall"],
                result["f1"],
                result["precision"],
                result["threshold"],
            ),
        )
        selection_reason = "minimum_precision_met"
    else:
        selected = max(
            results,
            key=lambda result: (
                result["f1"],
                result["precision"],
                result["recall"],
                result["threshold"],
            ),
        )
        selection_reason = "best_f1_fallback"

    return {
        "label": label,
        "label_id": label_id,
        "minimum_precision": float(minimum_precision),
        "selection_reason": selection_reason,
        "selected": selected,
        "candidates": results,
    }


def classification_metrics(targets, predictions, label_to_id):
    labels = sorted(label_to_id.values())
    id_to_label = {value: key for key, value in label_to_id.items()}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1

    per_class = {}
    f1_values = []
    for label_id in labels:
        tp = int(matrix[label_id, label_id])
        fp = int(matrix[:, label_id].sum() - tp)
        fn = int(matrix[label_id, :].sum() - tp)
        support = int(matrix[label_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[id_to_label[label_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = float(np.mean(targets == predictions)) if len(targets) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def label_counts(labels, label_to_id):
    labels = np.asarray(labels, dtype=np.int64)
    return {
        label: int(np.sum(labels == label_id))
        for label, label_id in label_to_id.items()
    }


def save_json(path, value):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def build_loaders(arrays, splits, config):
    feature_config = config["features"]
    training_config = config["training"]
    datasets = {}
    for name, indices in splits.items():
        datasets[name] = PoseSequenceDataset(
            arrays=arrays,
            indices=indices,
            confidence_threshold=feature_config["confidence_threshold"],
            horizontal_flip_probability=(
                training_config["horizontal_flip_probability"] if name == "train" else 0.0
            ),
            temporal_jitter_probability=(
                training_config.get("temporal_jitter_probability", 0.0)
                if name == "train"
                else 0.0
            ),
            max_temporal_jitter_frames=training_config.get(
                "max_temporal_jitter_frames",
                1,
            ),
        )

    train_labels = arrays["y"][splits["train"]]
    sampler = make_weighted_sampler(train_labels)
    common = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config.get("num_workers", 0)),
    }
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders, sampler
