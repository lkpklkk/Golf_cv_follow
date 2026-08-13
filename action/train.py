from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from action.classifier import ActionSequenceClassifier
from action.review_export import (
    export_false_positive_videos,
    load_metadata,
    select_false_positive_samples,
)
from action.training import (
    build_loaders,
    calibrate_class_threshold,
    collect_classifier_outputs,
    evaluate_classifier,
    evaluate_probabilities,
    grouped_stratified_split,
    label_counts,
    load_action_config,
    load_processed_arrays,
    save_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the golf action sequence classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("action_classifier_config.toml"),
        help="Action classifier TOML config.",
    )
    args = parser.parse_args(argv)
    train_from_config(args.config)


def train_from_config(config_path):
    config = load_action_config(config_path)
    training_config = config["training"]
    seed = int(training_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    arrays = load_processed_arrays(config["data"]["samples_file"])
    sequence_length = int(config["model"]["sequence_length"])
    if arrays["X"].shape[1] != sequence_length:
        raise ValueError(
            f"Dataset sequence length is {arrays['X'].shape[1]}, "
            f"but model config expects {sequence_length}"
        )

    label_to_id = config["labels"]["label_to_id"]
    ratios = (
        float(training_config["train_ratio"]),
        float(training_config["validation_ratio"]),
        float(training_config["test_ratio"]),
    )
    splits, split_videos = grouped_stratified_split(
        arrays["video_ids"],
        arrays["y"],
        ratios=ratios,
        seed=seed,
    )
    _, loaders, _ = build_loaders(arrays, splits, config)

    split_manifest = {
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "videos": split_videos,
        "sample_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "label_counts": {
            name: label_counts(arrays["y"][indices], label_to_id)
            for name, indices in splits.items()
        },
    }
    save_json(config["data"]["split_manifest_file"], split_manifest)
    print(f"[action split] {split_manifest['sample_counts']}")
    print(f"[action split labels] {split_manifest['label_counts']}")

    classifier = ActionSequenceClassifier(
        model_config=config["model"],
        label_to_id=label_to_id,
        feature_config=config["features"],
        decision_config=config["inference"],
        device=training_config.get("device", "auto"),
    )
    training_result = classifier.fit(
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        patience=training_config["patience"],
        evaluate_fn=evaluate_classifier,
    )

    validation_targets, validation_probabilities = collect_classifier_outputs(
        classifier,
        loaders["validation"],
    )
    test_targets, test_probabilities = collect_classifier_outputs(classifier, loaders["test"])
    calibration_config = config["calibration"]
    calibration = calibrate_class_threshold(
        validation_targets,
        validation_probabilities,
        label_to_id=label_to_id,
        label=calibration_config["label"],
        confidence_threshold=config["inference"]["confidence_threshold"],
        fallback_label=config["inference"]["fallback_label"],
        minimum_precision=calibration_config["minimum_precision"],
        min_threshold=calibration_config["min_threshold"],
        max_threshold=calibration_config["max_threshold"],
        step=calibration_config["step"],
    )
    calibrated_label = calibration["label"]
    selected_threshold = calibration["selected"]["threshold"]
    classifier.decision_config["class_thresholds"][calibrated_label] = selected_threshold
    print(
        f"[action calibration] label={calibrated_label} threshold={selected_threshold:.2f} "
        f"precision={calibration['selected']['precision']:.4f} "
        f"recall={calibration['selected']['recall']:.4f} "
        f"reason={calibration['selection_reason']}"
    )

    raw_test_metrics = evaluate_classifier(classifier, loaders["test"])
    validation_metrics, _ = evaluate_probabilities(
        validation_targets,
        validation_probabilities,
        label_to_id=label_to_id,
        confidence_threshold=classifier.decision_config["confidence_threshold"],
        fallback_label=classifier.decision_config["fallback_label"],
        class_thresholds=classifier.decision_config["class_thresholds"],
    )
    test_metrics, test_predictions = evaluate_probabilities(
        test_targets,
        test_probabilities,
        label_to_id=label_to_id,
        confidence_threshold=classifier.decision_config["confidence_threshold"],
        fallback_label=classifier.decision_config["fallback_label"],
        class_thresholds=classifier.decision_config["class_thresholds"],
    )

    metadata = load_metadata(config["data"]["metadata_file"])
    if len(metadata) != len(arrays["X"]):
        raise ValueError(
            f"Metadata has {len(metadata)} rows, expected {len(arrays['X'])}"
        )
    review_config = config["review_export"]
    selected_false_positives = select_false_positive_samples(
        global_indices=splits["test"],
        targets=test_targets,
        predictions=test_predictions,
        probabilities=test_probabilities,
        metadata=metadata,
        label_to_id=label_to_id,
        actual_label=review_config["actual_label"],
        predicted_label=review_config["predicted_label"],
        max_videos=review_config["max_videos"],
        overlap_suppression_sec=review_config["overlap_suppression_sec"],
    )
    review_manifest = export_false_positive_videos(
        selected_samples=selected_false_positives,
        arrays=arrays,
        raw_video_dir=review_config["raw_video_dir"],
        output_dir=review_config["output_dir"],
        target_fps=config["inference"]["target_fps"],
        pose_confidence_threshold=config["features"]["confidence_threshold"],
    )

    metrics = {
        "raw_label_counts": label_counts(arrays["y"], label_to_id),
        "split": split_manifest,
        "best_validation": training_result["best_validation"],
        "calibration": calibration,
        "validation_calibrated": validation_metrics,
        "test_raw": raw_test_metrics,
        "test": test_metrics,
        "false_positive_review_count": len(review_manifest),
        "model": classifier.checkpoint_summary(),
    }
    classifier.save(
        config["data"]["checkpoint_file"],
        extra={
            "best_validation": training_result["best_validation"],
            "calibration": calibration,
            "test": test_metrics,
        },
    )
    save_json(config["data"]["metrics_file"], metrics)
    print(f"[action save] {config['data']['checkpoint_file']}")
    print(f"[action save] {config['data']['metrics_file']}")
    print(
        f"[action test] accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"{'label':<8} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")
    for label, class_metrics in test_metrics["per_class"].items():
        print(
            f"{label:<8} "
            f"{class_metrics['precision']:>9.4f} "
            f"{class_metrics['recall']:>9.4f} "
            f"{class_metrics['f1']:>9.4f} "
            f"{class_metrics['support']:>9d}"
        )
    return metrics


if __name__ == "__main__":
    main()
