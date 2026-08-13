from __future__ import annotations

from pathlib import Path

from action.classifier import ActionSequenceClassifier
from action.live_buffer import ActionSequenceBuffer
from action.training import load_action_config


def create_live_action_components(config_path="action_classifier_config.toml"):
    path = Path(config_path)
    if not path.exists():
        print(f"[Action] config not found: {path}")
        return None, None, None

    config = load_action_config(path)
    checkpoint_path = Path(config["data"]["checkpoint_file"])
    if not checkpoint_path.exists():
        print(f"[Action] checkpoint not found: {checkpoint_path}")
        return None, None, config

    classifier = ActionSequenceClassifier.load(
        checkpoint_path,
        device=config["training"].get("device", "auto"),
    )
    inference = config["inference"]
    if not classifier.decision_config.get("class_thresholds"):
        classifier.decision_config["class_thresholds"] = dict(
            inference.get("class_thresholds", {})
        )
    buffer = ActionSequenceBuffer(
        sequence_length=classifier.sequence_length,
        target_fps=inference["target_fps"],
        classify_stride_sec=inference["classify_stride_sec"],
        max_gap_sec=inference["max_gap_sec"],
    )
    # Both sources are printed because they disagree silently: the checkpoint
    # carries whatever calibration selected during training, the config carries
    # whatever was hand-tuned since, and only the latter is applied per call.
    print(
        f"[Action] loaded {checkpoint_path} "
        f"window={classifier.sequence_length} frames"
    )
    print(
        f"[Action] thresholds: confidence={inference['confidence_threshold']} "
        f"config class_thresholds={dict(inference.get('class_thresholds') or {})} "
        f"checkpoint class_thresholds="
        f"{dict(classifier.decision_config.get('class_thresholds') or {})}"
    )
    return classifier, buffer, config


def update_live_action(
    classifier,
    buffer,
    config,
    target_id,
    tracked_person,
    timestamp,
    frame_width,
    frame_height,
):
    if classifier is None or buffer is None or config is None:
        return None

    keypoints = tracked_person.get("keypoints") if tracked_person is not None else None
    sequence = buffer.add(
        target_id=target_id,
        timestamp=timestamp,
        keypoints=keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if sequence is None:
        return None

    inference = config["inference"]
    return classifier.classify(
        sequence,
        frame_width=buffer.frame_width,
        frame_height=buffer.frame_height,
        confidence_threshold=inference["confidence_threshold"],
        fallback_label=inference["fallback_label"],
        # Passed explicitly, like the two above: without it the config section
        # loses to whatever calibration baked into the checkpoint at train
        # time, and editing the TOML silently does nothing. Omitting the
        # section from the config still falls back to the calibrated value.
        class_thresholds=inference.get("class_thresholds"),
    )
