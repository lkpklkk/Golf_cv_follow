from __future__ import annotations

import numpy as np


COCO_KEYPOINT_COUNT = 17
FEATURE_NAMES = (
    "frame_x",
    "frame_y",
    "body_x",
    "body_y",
    "confidence",
    "frame_dx",
    "frame_dy",
    "body_dx",
    "body_dy",
    "left_wrist_shoulder_distance",
    "right_wrist_shoulder_distance",
    "left_wrist_relative_speed",
    "right_wrist_relative_speed",
    "shoulder_axis_x",
    "shoulder_axis_y",
    "hip_axis_x",
    "hip_axis_y",
    "shoulder_rotation_delta",
    "hip_rotation_delta",
)
LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


def preprocess_pose_sequence(
    keypoints,
    frame_width,
    frame_height,
    confidence_threshold=0.25,
):
    sequence = np.asarray(keypoints, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (COCO_KEYPOINT_COUNT, 3):
        raise ValueError(
            "Expected keypoints with shape (sequence_length, 17, 3), "
            f"got {sequence.shape}"
        )

    width = max(float(frame_width), 1.0)
    height = max(float(frame_height), 1.0)
    confidence = np.clip(sequence[..., 2], 0.0, 1.0)
    valid = confidence >= float(confidence_threshold)

    frame_xy = np.zeros(sequence.shape[:2] + (2,), dtype=np.float32)
    frame_xy[..., 0] = sequence[..., 0] / width
    frame_xy[..., 1] = sequence[..., 1] / height
    frame_xy[~valid] = 0.0

    body_xy = np.zeros_like(frame_xy)
    for frame_index in range(sequence.shape[0]):
        center, scale = _body_center_and_scale(frame_xy[frame_index], valid[frame_index])
        if center is None:
            continue
        body_xy[frame_index] = (frame_xy[frame_index] - center) / scale
        body_xy[frame_index, ~valid[frame_index]] = 0.0

    frame_velocity = _masked_velocity(frame_xy, valid)
    body_velocity = _masked_velocity(body_xy, valid)
    swing_features = _swing_features(body_xy, valid)

    keypoint_features = np.concatenate(
        [
            frame_xy,
            body_xy,
            confidence[..., None],
            frame_velocity,
            body_velocity,
        ],
        axis=-1,
    )
    keypoint_features[~valid] = 0.0
    repeated_swing_features = np.repeat(
        swing_features[:, None, :],
        COCO_KEYPOINT_COUNT,
        axis=1,
    )
    features = np.concatenate([keypoint_features, repeated_swing_features], axis=-1)
    return features.astype(np.float32)


def horizontal_flip_sequence(keypoints, frame_width):
    sequence = np.asarray(keypoints, dtype=np.float32).copy()
    width = float(frame_width)
    valid = sequence[..., 2] > 0.0
    sequence[..., 0] = np.where(valid, width - sequence[..., 0], sequence[..., 0])
    for left, right in LEFT_RIGHT_PAIRS:
        sequence[:, [left, right], :] = sequence[:, [right, left], :]
    return sequence


def _body_center_and_scale(frame_xy, valid):
    hip_center = _pair_midpoint(frame_xy, valid, 11, 12)
    shoulder_center = _pair_midpoint(frame_xy, valid, 5, 6)

    center = hip_center if hip_center is not None else shoulder_center
    if center is None:
        points = frame_xy[valid]
        if len(points) == 0:
            return None, 1.0
        center = points.mean(axis=0)

    scale = None
    if hip_center is not None and shoulder_center is not None:
        torso = float(np.linalg.norm(shoulder_center - hip_center))
        if torso > 1e-4:
            scale = torso

    if scale is None:
        points = frame_xy[valid]
        if len(points) >= 2:
            span = points.max(axis=0) - points.min(axis=0)
            candidate = float(max(span[0], span[1]))
            if candidate > 1e-4:
                scale = candidate

    return center, scale or 1.0


def _pair_midpoint(frame_xy, valid, a, b):
    if valid[a] and valid[b]:
        return (frame_xy[a] + frame_xy[b]) * 0.5
    return None


def _masked_velocity(values, valid):
    velocity = np.zeros_like(values)
    if len(values) <= 1:
        return velocity
    pair_valid = valid[1:] & valid[:-1]
    delta = values[1:] - values[:-1]
    velocity[1:] = np.where(pair_valid[..., None], delta, 0.0)
    return velocity


def _swing_features(body_xy, valid):
    left_wrist_vector, left_wrist_valid = _relative_vector(body_xy, valid, 9, 5)
    right_wrist_vector, right_wrist_valid = _relative_vector(body_xy, valid, 10, 6)
    shoulder_axis, shoulder_valid = _unit_axis(body_xy, valid, 5, 6)
    hip_axis, hip_valid = _unit_axis(body_xy, valid, 11, 12)

    left_distance = _masked_norm(left_wrist_vector, left_wrist_valid)
    right_distance = _masked_norm(right_wrist_vector, right_wrist_valid)
    left_speed = _relative_speed(left_wrist_vector, left_wrist_valid)
    right_speed = _relative_speed(right_wrist_vector, right_wrist_valid)
    shoulder_rotation = _rotation_delta(shoulder_axis, shoulder_valid)
    hip_rotation = _rotation_delta(hip_axis, hip_valid)

    return np.column_stack(
        [
            left_distance,
            right_distance,
            left_speed,
            right_speed,
            shoulder_axis,
            hip_axis,
            shoulder_rotation,
            hip_rotation,
        ]
    ).astype(np.float32)


def _relative_vector(values, valid, point_index, anchor_index):
    pair_valid = valid[:, point_index] & valid[:, anchor_index]
    vector = np.zeros((len(values), 2), dtype=np.float32)
    vector[pair_valid] = values[pair_valid, point_index] - values[pair_valid, anchor_index]
    return vector, pair_valid


def _unit_axis(values, valid, left_index, right_index):
    pair_valid = valid[:, left_index] & valid[:, right_index]
    axis = np.zeros((len(values), 2), dtype=np.float32)
    axis[pair_valid] = values[pair_valid, right_index] - values[pair_valid, left_index]
    norms = np.linalg.norm(axis, axis=1)
    usable = pair_valid & (norms > 1e-4)
    axis[usable] /= norms[usable, None]
    axis[~usable] = 0.0
    return axis, usable


def _masked_norm(values, valid):
    output = np.zeros(len(values), dtype=np.float32)
    output[valid] = np.linalg.norm(values[valid], axis=1)
    return output


def _relative_speed(values, valid):
    output = np.zeros(len(values), dtype=np.float32)
    if len(values) <= 1:
        return output
    pair_valid = valid[1:] & valid[:-1]
    delta = values[1:] - values[:-1]
    output[1:][pair_valid] = np.linalg.norm(delta[pair_valid], axis=1)
    return output


def _rotation_delta(unit_axes, valid):
    output = np.zeros(len(unit_axes), dtype=np.float32)
    if len(unit_axes) <= 1:
        return output
    pair_valid = valid[1:] & valid[:-1]
    previous = unit_axes[:-1]
    current = unit_axes[1:]
    cross = previous[:, 0] * current[:, 1] - previous[:, 1] * current[:, 0]
    dot = np.sum(previous * current, axis=1)
    angles = np.arctan2(cross, dot).astype(np.float32)
    output[1:][pair_valid] = angles[pair_valid]
    return output
