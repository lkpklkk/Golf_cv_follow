from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from ui.overlay import _COCO_SKELETON


def load_metadata(path):
    with Path(path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        raise ValueError(f"Expected metadata list in {path}")
    return metadata


def select_false_positive_samples(
    global_indices,
    targets,
    predictions,
    probabilities,
    metadata,
    label_to_id,
    actual_label="other",
    predicted_label="swing",
    max_videos=20,
    overlap_suppression_sec=1.0,
):
    actual_id = int(label_to_id[actual_label])
    predicted_id = int(label_to_id[predicted_label])
    candidates = []
    for local_index, global_index in enumerate(global_indices):
        if int(targets[local_index]) != actual_id:
            continue
        if int(predictions[local_index]) != predicted_id:
            continue
        sample_metadata = metadata[int(global_index)]
        candidates.append(
            {
                "global_index": int(global_index),
                "probability": float(probabilities[local_index, predicted_id]),
                "metadata": sample_metadata,
            }
        )

    candidates.sort(key=lambda item: item["probability"], reverse=True)
    selected = []
    selected_starts_by_video = {}
    for candidate in candidates:
        video = candidate["metadata"]["video"]
        start_sec = float(candidate["metadata"]["start_sec"])
        previous_starts = selected_starts_by_video.setdefault(video, [])
        if any(
            abs(start_sec - previous_start) < float(overlap_suppression_sec)
            for previous_start in previous_starts
        ):
            continue
        previous_starts.append(start_sec)
        selected.append(candidate)
        if len(selected) >= int(max_videos):
            break
    return selected


def export_false_positive_videos(
    selected_samples,
    arrays,
    raw_video_dir,
    output_dir,
    target_fps,
    pose_confidence_threshold=0.25,
):
    raw_video_dir = Path(raw_video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for rank, selected in enumerate(selected_samples, start=1):
        global_index = selected["global_index"]
        metadata = selected["metadata"]
        video_path = raw_video_dir / metadata["video"]
        output_name = (
            f"fp_{rank:03d}_{Path(metadata['video']).stem}_"
            f"{float(metadata['start_sec']):07.3f}s_"
            f"p{selected['probability']:.3f}.mp4"
        )
        output_path = output_dir / output_name
        ok = _write_review_clip(
            video_path=video_path,
            output_path=output_path,
            keypoint_sequence=arrays["X"][global_index],
            start_sec=float(metadata["start_sec"]),
            target_fps=float(target_fps),
            probability=selected["probability"],
            pose_confidence_threshold=float(pose_confidence_threshold),
        )
        if not ok:
            continue
        manifest.append(
            {
                "rank": rank,
                "sample_index": global_index,
                "video": metadata["video"],
                "start_sec": metadata["start_sec"],
                "end_sec": metadata["end_sec"],
                "actual_label": "other",
                "predicted_label": "swing",
                "swing_probability": selected["probability"],
                "output_video": str(output_path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[action review] exported {len(manifest)} clips to {output_dir}")
    return manifest


def _write_review_clip(
    video_path,
    output_path,
    keypoint_sequence,
    start_sec,
    target_fps,
    probability,
    pose_confidence_threshold,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[action review] could not open {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        print(f"[action review] invalid frame size for {video_path}")
        return False

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(target_fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"[action review] could not create {output_path}")
        return False

    for frame_index, keypoints in enumerate(keypoint_sequence):
        timestamp = float(start_sec) + frame_index / float(target_fps)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        _draw_pose(frame, keypoints, pose_confidence_threshold)
        cv2.rectangle(frame, (0, 0), (width, 76), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "False positive review: actual=other predicted=swing",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
        )
        cv2.putText(
            frame,
            f"swing_probability={probability:.3f} time={timestamp:.3f}s",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    return True


def _draw_pose(frame, keypoints, confidence_threshold):
    for a, b in _COCO_SKELETON:
        if keypoints[a, 2] >= confidence_threshold and keypoints[b, 2] >= confidence_threshold:
            cv2.line(
                frame,
                (int(keypoints[a, 0]), int(keypoints[a, 1])),
                (int(keypoints[b, 0]), int(keypoints[b, 1])),
                (0, 255, 0),
                2,
            )
    for x, y, confidence in keypoints:
        if confidence >= confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
