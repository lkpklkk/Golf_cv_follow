"""
Generate fixed-length pose-sequence training samples from annotated videos.

Entry point: golf_cv_generate_dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random

import cv2
import numpy as np

import config
from reid.embedder import PersonEmbedder
from reid.matcher import ReIDMatcher
from ui.overlay import _COCO_SKELETON
from utils.annotations import (
    discover_videos,
    load_annotations,
    load_reference_embeddings,
    resolve_reference_image_path,
)
from utils.device import select_torch_device, use_half_precision
from utils.geometry import box_area

COCO_KEYPOINT_COUNT = 17
KEYPOINT_DIMS = 3
DEFAULT_DATASET_CONFIG = Path("dataset_config.yaml")
DEFAULT_ANNOTATION_FILE = Path("data/annotations/video_intervals.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetWindow:
    start_sec: float
    end_sec: float
    label_name: str
    label_id: int
    source_interval: dict | None


# ---------------------------------------------------------------------------
# Pose estimator
# ---------------------------------------------------------------------------


class DatasetPoseEstimator:
    def __init__(self, dataset_config: dict, embedder=None):
        from ultralytics import YOLO

        pose_config = dataset_config.get("pose", {})
        model_name = pose_config.get("model_name", config.MODEL_NAME)
        self.device = select_torch_device(config.INFERENCE_DEVICE)
        self.half = use_half_precision(self.device, config.USE_HALF_ON_CUDA)
        self.model = YOLO(model_name)
        self.embedder = embedder
        print(f"[YOLO dataset] model={model_name} device={self.device}")

    def predict_keypoints(self, frame, reference_embeddings=None):
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
            device=self.device,
            half=self.half,
        )
        if not results:
            return _empty_keypoints(), True

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return _empty_keypoints(), True

        keypoints = _select_pose_keypoints(
            result,
            frame,
            reference_embeddings=reference_embeddings,
            embedder=self.embedder,
        )
        if keypoints is None:
            return _empty_keypoints(), True

        return keypoints, False


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def generate_pose_dataset(dataset_config: dict):
    data_config = dataset_config["data"]
    processed_dir = Path(data_config["processed_dir"]).expanduser()
    raw_video_dir = Path(data_config["raw_video_dir"]).expanduser()
    annotation_file = Path(data_config["annotation_file"]).expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotation_file)
    annotation_by_video = {entry.get("video"): entry for entry in annotations}
    videos = discover_videos(raw_video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {raw_video_dir}")

    embedder = PersonEmbedder()
    estimator = DatasetPoseEstimator(dataset_config, embedder=embedder)
    samples = []
    labels = []
    video_ids = []
    start_times = []
    end_times = []
    frame_widths = []
    frame_heights = []
    metadata = []
    debug_samples = {}

    label_names = dataset_config["labels"]["classes"]
    label_counts = Counter({label: 0 for label in label_names})
    videos_processed = 0
    videos_skipped = 0
    windows_skipped_missing_pose = 0
    pose_frame_count = 0
    missing_pose_frame_count = 0

    for video_path in videos:
        annotation = annotation_by_video.get(
            video_path.name,
            {"video": video_path.name, "skipped": False, "intervals": []},
        )
        if annotation.get("skipped", False):
            videos_skipped += 1
            print(f"[dataset skip] {video_path.name} marked skipped")
            continue

        result = process_dataset_video(
            video_path=video_path,
            annotation=annotation,
            dataset_config=dataset_config,
            estimator=estimator,
            sample_offset=len(samples),
            debug_samples=debug_samples,
            annotation_dir=annotation_file.parent,
            embedder=embedder,
        )
        if result is None:
            videos_skipped += 1
            continue

        videos_processed += 1
        (
            video_samples,
            video_labels,
            video_video_ids,
            video_start_times,
            video_end_times,
            video_frame_widths,
            video_frame_heights,
            video_metadata,
            video_label_counts,
            video_pose_frames,
            video_missing_pose_frames,
            video_skipped_windows,
        ) = result

        samples.extend(video_samples)
        labels.extend(video_labels)
        video_ids.extend(video_video_ids)
        start_times.extend(video_start_times)
        end_times.extend(video_end_times)
        frame_widths.extend(video_frame_widths)
        frame_heights.extend(video_frame_heights)
        metadata.extend(video_metadata)
        label_counts.update(video_label_counts)
        pose_frame_count += video_pose_frames
        missing_pose_frame_count += video_missing_pose_frames
        windows_skipped_missing_pose += video_skipped_windows

    save_npz(
        processed_dir / "samples.npz",
        samples,
        labels,
        video_ids,
        start_times,
        end_times,
        frame_widths,
        frame_heights,
        dataset_config,
    )
    save_metadata_json(processed_dir / "metadata.json", metadata)
    save_label_counts_json(
        processed_dir / "label_counts.json", label_counts, label_names
    )
    write_debug_videos(processed_dir / "debug_videos", debug_samples, dataset_config)

    print("[dataset done]")
    print(f"  videos processed: {videos_processed}")
    print(f"  videos skipped: {videos_skipped}")
    print(f"  samples generated: {len(samples)}")
    print(f"  samples per label: {dict(label_counts)}")
    print(f"  windows skipped for missing pose: {windows_skipped_missing_pose}")
    print(
        "  missing/low-confidence pose frames: "
        f"{missing_pose_frame_count}/{pose_frame_count}"
    )


def process_dataset_video(
    video_path: Path,
    annotation: dict,
    dataset_config: dict,
    estimator: DatasetPoseEstimator,
    sample_offset: int,
    debug_samples: dict,
    annotation_dir: Path | None = None,
    embedder=None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[dataset skip] Could not open {video_path}")
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
    if total_frames <= 0:
        cap.release()
        print(f"[dataset skip] {video_path.name} has no frames")
        return None

    duration_sec = total_frames / source_fps
    intervals = annotation.get("intervals") or []
    reference_embeddings = load_reference_embeddings(
        annotation,
        embedder,
        annotation_dir or Path("."),
    )
    if reference_embeddings:
        print(
            f"[dataset refs] {video_path.name}: "
            f"using {len(reference_embeddings)} reference embedding(s)"
        )
    else:
        print(f"[dataset refs] {video_path.name}: using largest detected person")
    windows = create_windows(duration_sec, intervals, dataset_config)
    print(
        f"[dataset video] {video_path.name}: {len(windows)} windows "
        f"from {duration_sec:.2f}s"
    )

    sampling_config = dataset_config["sampling"]
    pose_config = dataset_config.get("pose", {})
    sequence_length = int(sampling_config["sequence_length"])
    max_missing_ratio = float(pose_config.get("max_missing_frame_ratio", 0.3))

    samples = []
    labels = []
    video_ids = []
    start_times = []
    end_times = []
    frame_widths = []
    frame_heights = []
    metadata = []
    label_counts = Counter()
    pose_frame_count = 0
    missing_pose_frame_count = 0
    skipped_windows = 0
    pose_cache = {}
    window_frame_indices = [
        sample_frame_timestamps(
            window.start_sec,
            source_fps,
            total_frames,
            dataset_config,
            seed=_dataset_sampling_seed(
                video_path.name,
                window.start_sec,
                sampling_config.get("timestamp_jitter_seed", 42),
            ),
        )[0]
        for window in windows
    ]
    unique_frame_indices = sorted(
        {frame_index for indices in window_frame_indices for frame_index in indices}
    )
    populate_pose_cache(
        cap,
        unique_frame_indices,
        estimator,
        dataset_config,
        reference_embeddings=reference_embeddings,
        pose_cache=pose_cache,
    )
    print(
        f"[dataset poses] {video_path.name}: inferred {len(unique_frame_indices)} "
        f"unique frames for {len(windows)} windows"
    )

    for window, frame_indices in zip(windows, window_frame_indices):
        keep_debug_frames = window.label_name not in debug_samples
        keypoints, missing_frames, debug_frames = run_yolo_pose_on_frames(
            cap,
            frame_indices,
            estimator,
            dataset_config,
            keep_frames=keep_debug_frames,
            reference_embeddings=reference_embeddings,
            pose_cache=pose_cache,
        )
        pose_frame_count += sequence_length
        missing_pose_frame_count += missing_frames

        missing_ratio = missing_frames / sequence_length
        if missing_ratio > max_missing_ratio:
            skipped_windows += 1
            continue

        sample_index = sample_offset + len(samples)
        samples.append(keypoints)
        labels.append(window.label_id)
        video_ids.append(video_path.name)
        start_times.append(window.start_sec)
        end_times.append(window.end_sec)
        frame_widths.append(frame_width)
        frame_heights.append(frame_height)
        label_counts[window.label_name] += 1
        metadata.append(
            {
                "sample_index": sample_index,
                "video": video_path.name,
                "label_name": window.label_name,
                "label_id": window.label_id,
                "start_sec": round(window.start_sec, 3),
                "end_sec": round(window.end_sec, 3),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "timestamp_jitter_ratio": float(
                    sampling_config.get("timestamp_jitter_ratio", 0.0)
                ),
                "source_interval": window.source_interval,
            }
        )

        if keep_debug_frames and debug_frames:
            debug_samples[window.label_name] = {
                "frames": debug_frames,
                "keypoints": keypoints.copy(),
                "video": video_path.name,
                "start_sec": window.start_sec,
            }

    cap.release()
    print(
        f"[dataset video done] {video_path.name}: {len(samples)} samples, "
        f"skipped_windows={skipped_windows}, labels={dict(label_counts)}"
    )
    return (
        samples,
        labels,
        video_ids,
        start_times,
        end_times,
        frame_widths,
        frame_heights,
        metadata,
        label_counts,
        pose_frame_count,
        missing_pose_frame_count,
        skipped_windows,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_dataset_config(config_path: Path) -> dict:
    path = config_path.expanduser()
    if not path.exists():
        raise SystemExit(f"Dataset config not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ImportError:
        loaded = _parse_simple_yaml(text)

    if not isinstance(loaded, dict):
        raise SystemExit(f"Dataset config must be a mapping: {path}")

    return _normalize_dataset_config(loaded)


# ---------------------------------------------------------------------------
# Window / sampling helpers
# ---------------------------------------------------------------------------


def sample_frame_timestamps(
    start_sec,
    source_fps,
    total_frames,
    dataset_config,
    seed=None,
):
    sampling_config = dataset_config["sampling"]
    target_fps = float(sampling_config["target_fps"])
    sequence_length = int(sampling_config["sequence_length"])
    jitter_ratio = float(sampling_config.get("timestamp_jitter_ratio", 0.0))
    if jitter_ratio < 0.0 or jitter_ratio >= 1.0:
        raise ValueError("sampling.timestamp_jitter_ratio must be in [0.0, 1.0)")

    offsets = _sample_timestamp_offsets(
        sequence_length=sequence_length,
        target_fps=target_fps,
        jitter_ratio=jitter_ratio,
        seed=seed,
    )
    timestamps = [start_sec + offset for offset in offsets]
    frame_indices = [
        min(total_frames - 1, max(0, int(round(timestamp * source_fps))))
        for timestamp in timestamps
    ]
    return frame_indices, timestamps


def run_yolo_pose_on_frames(
    cap,
    frame_indices,
    estimator: DatasetPoseEstimator,
    dataset_config: dict,
    keep_frames=False,
    reference_embeddings=None,
    pose_cache=None,
):
    pose_config = dataset_config.get("pose", {})
    pose_conf_threshold = float(pose_config.get("pose_conf_threshold", 0.25))
    sequence = []
    debug_frames = []
    missing_frames = 0

    for frame_index in frame_indices:
        cached = pose_cache.get(frame_index) if pose_cache is not None else None
        frame = None
        if cached is not None:
            keypoints, missing = cached
            if keep_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    frame = None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                keypoints = _empty_keypoints()
                missing = True
                frame = None
            else:
                keypoints, missing = estimator.predict_keypoints(
                    frame,
                    reference_embeddings=reference_embeddings,
                )
                missing = missing or _is_low_confidence_pose(
                    keypoints, pose_conf_threshold
                )
            if pose_cache is not None:
                pose_cache[frame_index] = (keypoints.copy(), missing)

        if missing:
            missing_frames += 1

        sequence.append(keypoints)
        if keep_frames:
            if frame is None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
                debug_frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            else:
                debug_frames.append(frame.copy())

    return np.asarray(sequence, dtype=np.float32), missing_frames, debug_frames


def populate_pose_cache(
    cap,
    frame_indices,
    estimator: DatasetPoseEstimator,
    dataset_config: dict,
    reference_embeddings=None,
    pose_cache=None,
):
    if pose_cache is None:
        pose_cache = {}

    pose_config = dataset_config.get("pose", {})
    pose_conf_threshold = float(pose_config.get("pose_conf_threshold", 0.25))
    previous_frame_index = None
    for frame_index in frame_indices:
        if frame_index in pose_cache:
            previous_frame_index = frame_index
            continue

        if previous_frame_index is None or frame_index != previous_frame_index + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            keypoints = _empty_keypoints()
            missing = True
        else:
            keypoints, missing = estimator.predict_keypoints(
                frame,
                reference_embeddings=reference_embeddings,
            )
            missing = missing or _is_low_confidence_pose(keypoints, pose_conf_threshold)
        pose_cache[frame_index] = (keypoints.copy(), missing)
        previous_frame_index = frame_index
    return pose_cache


def create_windows(duration_sec, intervals, dataset_config):
    sampling_config = dataset_config["sampling"]
    window_sec = float(sampling_config["window_sec"])
    target_fps = float(sampling_config["target_fps"])

    if duration_sec + 1e-9 < window_sec:
        return []

    max_start = max(0.0, duration_sec - window_sec)
    candidate_ticks = set()

    include_unlabeled_as_other = bool(
        dataset_config["windowing"].get("include_unlabeled_as_other", True)
    )
    if include_unlabeled_as_other:
        other_stride = float(dataset_config["windowing"]["other_stride_sec"])
        _add_window_start_range(
            candidate_ticks, 0.0, max_start, other_stride, target_fps
        )

    for interval in intervals:
        label = interval.get("label")
        stride = _stride_for_label(label, dataset_config)
        if stride is None:
            continue
        interval_start = float(interval.get("start_sec", 0.0))
        interval_end = float(interval.get("end_sec", interval_start))
        start = max(0.0, interval_start - window_sec)
        end = min(max_start, interval_end)
        _add_window_start_range(candidate_ticks, start, end, stride, target_fps)

    windows = []
    for tick in sorted(candidate_ticks):
        start_sec = min(max_start, tick / target_fps)
        end_sec = start_sec + window_sec
        label_name, label_id, source_interval = assign_label_by_overlap(
            start_sec,
            end_sec,
            intervals,
            dataset_config,
        )
        if label_name is None:
            continue
        windows.append(
            DatasetWindow(
                start_sec=start_sec,
                end_sec=end_sec,
                label_name=label_name,
                label_id=label_id,
                source_interval=source_interval,
            )
        )
    return windows


def assign_label_by_overlap(start_sec, end_sec, intervals, dataset_config):
    window_sec = float(dataset_config["sampling"]["window_sec"])
    label_to_id = dataset_config["labels"]["label_to_id"]
    priority = dataset_config["priority"]

    best_by_label = {}
    for interval in intervals:
        label = interval.get("label")
        if label not in label_to_id:
            continue
        interval_start = float(interval.get("start_sec", 0.0))
        interval_end = float(interval.get("end_sec", interval_start))
        overlap = max(0.0, min(end_sec, interval_end) - max(start_sec, interval_start))
        ratio = overlap / window_sec
        if ratio <= 0.0:
            continue

        current = best_by_label.get(label)
        if current is None or ratio > current[0]:
            best_by_label[label] = (ratio, _source_interval_metadata(interval, ratio))

    for label in priority:
        threshold = _overlap_threshold_for_label(label, dataset_config)
        ratio, source_interval = best_by_label.get(label, (0.0, None))
        if ratio >= threshold:
            return label, int(label_to_id[label]), source_interval

    if dataset_config["windowing"].get("include_unlabeled_as_other", True):
        return "other", int(label_to_id["other"]), None
    return None, None, None


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_npz(
    path: Path,
    samples,
    labels,
    video_ids,
    start_times,
    end_times,
    frame_widths,
    frame_heights,
    dataset_config,
):
    sampling_config = dataset_config["sampling"]
    sequence_length = int(sampling_config["sequence_length"])
    if samples:
        x = np.asarray(samples, dtype=np.float32)
    else:
        x = np.empty(
            (0, sequence_length, COCO_KEYPOINT_COUNT, KEYPOINT_DIMS),
            dtype=np.float32,
        )

    np.savez_compressed(
        path,
        X=x,
        y=np.asarray(labels, dtype=np.int64),
        video_ids=np.asarray(video_ids),
        start_sec=np.asarray(start_times, dtype=np.float32),
        end_sec=np.asarray(end_times, dtype=np.float32),
        frame_width=np.asarray(frame_widths, dtype=np.int32),
        frame_height=np.asarray(frame_heights, dtype=np.int32),
    )
    print(f"[dataset save] {path} X={x.shape}")


def save_metadata_json(path: Path, metadata):
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[dataset save] {path}")


def save_label_counts_json(path: Path, label_counts, label_names):
    import json

    counts = {label: int(label_counts.get(label, 0)) for label in label_names}
    with path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"[dataset save] {path}")


def write_debug_videos(output_dir: Path, debug_samples: dict, dataset_config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset_config["labels"]["classes"]:
        sample = debug_samples.get(label)
        if sample is None:
            print(f"[dataset warn] no valid debug sample for label={label}")
            continue

        output_path = output_dir / f"debug_{label}.mp4"
        write_debug_video(
            output_path, sample["frames"], sample["keypoints"], dataset_config
        )
        print(
            f"[dataset debug] {output_path} "
            f"from {sample['video']} @ {sample['start_sec']:.2f}s"
        )


def write_debug_video(path: Path, frames, keypoint_sequence, dataset_config: dict):
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fps = float(dataset_config["sampling"]["target_fps"])
    pose_conf_threshold = float(
        dataset_config.get("pose", {}).get("pose_conf_threshold", 0.25)
    )
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        print(f"[dataset warn] could not create debug video: {path}")
        return

    for frame, keypoints in zip(frames, keypoint_sequence):
        annotated = frame.copy()
        _draw_dataset_pose(
            annotated,
            keypoints,
            color=(0, 255, 0),
            conf_threshold=pose_conf_threshold,
        )
        writer.write(annotated)

    writer.release()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_dataset_config(dataset_config: dict):
    required_sections = ("data", "labels", "sampling", "windowing", "priority")
    for section in required_sections:
        if section not in dataset_config:
            raise ValueError(f"Dataset config missing required section: {section}")

    dataset_config.setdefault("pose", {})
    dataset_config["pose"].setdefault("model_name", config.MODEL_NAME)

    raw_model_name = dataset_config["pose"]["model_name"]
    if not Path(raw_model_name).is_absolute():
        candidate = config.PROJECT_ROOT / raw_model_name
        if candidate.exists():
            dataset_config["pose"]["model_name"] = str(candidate)

    dataset_config["pose"].setdefault("pose_conf_threshold", 0.25)
    dataset_config["pose"].setdefault("max_missing_frame_ratio", 0.3)

    labels = dataset_config["labels"]
    if "other" not in labels.get("label_to_id", {}):
        raise ValueError("Dataset config labels.label_to_id must define 'other'")

    return dataset_config


def _parse_simple_yaml(text: str):
    root = {}
    stack = [(0, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()

        container = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"Unexpected list item in config: {raw_line}")
            container.append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            raise ValueError(f"Could not parse config line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            container[key] = _parse_scalar(value)
            continue

        next_container = [] if key in {"classes", "priority"} else {}
        container[key] = next_container
        stack.append((indent + 2, next_container))

    return root


def _parse_scalar(value: str):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _add_window_start_range(
    candidate_ticks, start_sec, end_sec, stride_sec, target_fps
):
    if stride_sec <= 0 or end_sec < start_sec:
        return

    start_tick = max(0, int(round(start_sec * target_fps)))
    end_tick = max(0, int(round(end_sec * target_fps)))
    step_ticks = max(1, int(round(stride_sec * target_fps)))

    tick = start_tick
    while tick <= end_tick:
        candidate_ticks.add(tick)
        tick += step_ticks


def _stride_for_label(label, dataset_config):
    if not label:
        return None
    key = f"{label}_stride_sec"
    value = dataset_config["windowing"].get(key)
    return float(value) if value is not None else None


def _overlap_threshold_for_label(label, dataset_config):
    key = f"{label}_overlap_threshold"
    return float(dataset_config["windowing"].get(key, 0.0))


def _source_interval_metadata(interval, overlap_ratio):
    metadata = {
        "label": interval.get("label"),
        "start_sec": interval.get("start_sec"),
        "end_sec": interval.get("end_sec"),
        "overlap_ratio": round(float(overlap_ratio), 4),
    }
    for key in ("start_frame", "end_frame"):
        if key in interval:
            metadata[key] = interval[key]
    return metadata


def _select_pose_keypoints(result, frame, reference_embeddings=None, embedder=None):
    if reference_embeddings:
        return _reference_matched_pose_keypoints(
            result,
            frame,
            reference_embeddings,
            embedder,
        )
    return _largest_pose_keypoints(result)


def _reference_matched_pose_keypoints(result, frame, reference_embeddings, embedder):
    if embedder is None or not getattr(embedder, "available", True):
        return None

    boxes = []
    valid_indices = []
    for i, box in enumerate(result.boxes):
        if i >= len(result.keypoints.xy):
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        boxes.append((x1, y1, x2, y2))
        valid_indices.append(i)

    if not boxes:
        return None

    embeddings = embedder.embed_many(frame, boxes)
    scores = {}
    for result_index, embedding in zip(valid_indices, embeddings):
        if embedding is None:
            continue
        score = max(float(np.dot(ref, embedding)) for ref in reference_embeddings)
        scores[result_index] = max(0.0, min(1.0, score))

    selected_index = ReIDMatcher.select_best_match(scores)
    if selected_index is None:
        return None

    return _pose_keypoints_at_index(result, selected_index)


def _largest_pose_keypoints(result):
    kpts_data = result.keypoints
    if kpts_data is None or kpts_data.xy is None:
        return None

    best_index = None
    best_area = -1
    for i, box in enumerate(result.boxes):
        if i >= len(kpts_data.xy):
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        area = box_area((x1, y1, x2, y2))
        if area > best_area:
            best_area = area
            best_index = i

    if best_index is None:
        return None

    return _pose_keypoints_at_index(result, best_index)


def _pose_keypoints_at_index(result, index):
    kpts_data = result.keypoints
    xy = kpts_data.xy[index].cpu().numpy()
    if kpts_data.conf is not None:
        conf = kpts_data.conf[index].cpu().numpy()
    else:
        conf = np.ones(len(xy), dtype=np.float32)

    return _coerce_coco_keypoints(xy, conf)


def _coerce_coco_keypoints(xy, conf):
    keypoints = np.zeros((COCO_KEYPOINT_COUNT, KEYPOINT_DIMS), dtype=np.float32)
    count = min(COCO_KEYPOINT_COUNT, len(xy), len(conf))
    if count:
        keypoints[:count, :2] = xy[:count]
        keypoints[:count, 2] = conf[:count]
    return keypoints


def _empty_keypoints():
    return np.zeros((COCO_KEYPOINT_COUNT, KEYPOINT_DIMS), dtype=np.float32)


def _is_low_confidence_pose(keypoints, conf_threshold):
    return keypoints is None or float(np.max(keypoints[:, 2])) < conf_threshold


def _draw_dataset_pose(frame, keypoints, color, conf_threshold):
    if keypoints is None:
        return

    for a, b in _COCO_SKELETON:
        if keypoints[a, 2] >= conf_threshold and keypoints[b, 2] >= conf_threshold:
            cv2.line(
                frame,
                (int(keypoints[a, 0]), int(keypoints[a, 1])),
                (int(keypoints[b, 0]), int(keypoints[b, 1])),
                color,
                2,
            )

    for index, (x, y, conf) in enumerate(keypoints):
        if conf < conf_threshold:
            continue
        strength = min(1.0, max(0.0, float(conf)))
        point_color = (
            int(color[0] * strength),
            int(80 + 175 * strength),
            int(255 * (1.0 - strength)),
        )
        radius = max(3, int(round(3 + 3 * strength)))
        cv2.circle(frame, (int(x), int(y)), radius, point_color, -1)
        cv2.putText(
            frame,
            str(index),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            point_color,
            1,
        )


def _sample_timestamp_offsets(sequence_length, target_fps, jitter_ratio, seed=None):
    if sequence_length <= 0:
        return []
    if sequence_length == 1 or jitter_ratio <= 0.0:
        return [i / target_fps for i in range(sequence_length)]

    rng = random.Random(seed)
    base_interval = 1.0 / target_fps
    intervals = np.asarray(
        [
            base_interval * rng.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)
            for _ in range(sequence_length - 1)
        ],
        dtype=np.float64,
    )
    regular_span = (sequence_length - 1) / target_fps
    intervals *= regular_span / float(intervals.sum())
    offsets = np.concatenate([[0.0], np.cumsum(intervals)])
    return offsets.tolist()


def _dataset_sampling_seed(video_name, start_sec, base_seed):
    value = f"{base_seed}:{video_name}:{float(start_sec):.6f}"
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate pose-sequence training samples from annotated videos."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help=f"Dataset config YAML. Default: {DEFAULT_DATASET_CONFIG}",
    )
    args = parser.parse_args(argv)
    dataset_config = load_dataset_config(args.dataset_config)
    generate_pose_dataset(dataset_config)


if __name__ == "__main__":
    main()
