"""
Per-feature value ranges used to scale the debug-video feature bars.

The 19 model input features live on wildly different scales — `confidence` is
bounded to [0, 1], `frame_dx` rarely leaves ±0.03, `hip_rotation_delta` spans
±pi. Drawing them against a shared axis would make most bars invisible, so each
feature gets its own denominator measured from the training set.

Ranges come from `data/processed/samples.npz`, which already stores every
sample's raw keypoints and frame dimensions: re-running the real
`preprocess_pose_sequence` over all of it takes about a second, so no extra
video pass is needed to calibrate the bars.

Entry point: golf_cv_feature_stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action.preprocessing import (
    FEATURE_NAMES,
    PER_KEYPOINT_FEATURE_COUNT,
    keypoint_valid_mask,
    preprocess_pose_sequence,
)

DEFAULT_SAMPLES_FILE = Path("data/processed/samples.npz")
# Not under data/, which is gitignored: the bars are only comparable across runs
# if every run scales them the same way, so this file is version controlled
# alongside the other configs.
DEFAULT_STATS_FILE = Path("feature_norm_stats.json")
DEFAULT_PERCENTILE = 99.0
SIGNED_EPSILON = 1e-6


def compute_feature_stats(
    samples_file: Path = DEFAULT_SAMPLES_FILE,
    confidence_threshold: float = 0.25,
    percentile: float = DEFAULT_PERCENTILE,
) -> dict:
    """
    Measure each feature's range across every sample in the dataset.

    Returns {"percentile": p, "sample_count": n, "features": {name: {...}}}.
    """
    path = Path(samples_file)
    if not path.exists():
        raise SystemExit(
            f"Samples file not found: {path}. Run golf_cv_generate_dataset first."
        )

    data = np.load(path, allow_pickle=True)
    keypoints = data["X"]
    if len(keypoints) == 0:
        raise SystemExit(f"{path} contains no samples.")

    widths = data["frame_width"]
    heights = data["frame_height"]
    features = np.concatenate(
        [
            _displayed_vectors(
                preprocess_pose_sequence(
                    keypoints[index],
                    widths[index],
                    heights[index],
                    confidence_threshold=confidence_threshold,
                ),
                keypoint_valid_mask(keypoints[index], confidence_threshold),
            )
            for index in range(len(keypoints))
        ]
    )

    stats = {}
    for index, name in enumerate(FEATURE_NAMES):
        column = features[:, index]
        stats[name] = {
            "min": float(column.min()),
            "max": float(column.max()),
            # Scale to a high percentile of |value| rather than the extreme: a
            # handful of outlier frames (body_y reaches 7.3 against a p99 of
            # 2.1) would otherwise squash every bar into a sliver.
            "scale": float(max(np.percentile(np.abs(column), percentile), SIGNED_EPSILON)),
            "signed": bool(column.min() < -SIGNED_EPSILON),
        }

    return {
        "percentile": float(percentile),
        "sample_count": int(len(keypoints)),
        "frame_count": int(len(features)),
        "confidence_threshold": float(confidence_threshold),
        "features": stats,
    }


def _displayed_vectors(features, valid):
    """
    Reduce (T, 17, 19) features to the (T, 19) values the panel actually draws.

    Vectorised equivalent of calling frame_feature_vector() per frame — the
    ranges must describe the displayed quantity, not the raw per-keypoint
    spread. Those differ: `body_x` is signed and reaches ±2.3 per keypoint, but
    the panel shows its mean absolute value, which is non-negative and much
    narrower. Measuring the wrong one leaves every per-keypoint bar reading low
    and marks it bipolar when it can never go negative.
    """
    valid = np.asarray(valid, dtype=bool)
    counts = valid.sum(axis=1, keepdims=True)
    output = np.zeros((features.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    masked_sum = (
        np.abs(features[:, :, :PER_KEYPOINT_FEATURE_COUNT]) * valid[:, :, None]
    ).sum(axis=1)
    output[:, :PER_KEYPOINT_FEATURE_COUNT] = masked_sum / np.maximum(counts, 1)
    output[:, PER_KEYPOINT_FEATURE_COUNT:] = features[:, 0, PER_KEYPOINT_FEATURE_COUNT:]
    return output


def save_feature_stats(stats: dict, path: Path = DEFAULT_STATS_FILE) -> None:
    output = Path(path)
    if output.parent != Path(""):
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[feature stats] wrote {output}")


def load_feature_stats(path: Path = DEFAULT_STATS_FILE) -> dict | None:
    """
    Load previously measured ranges, or None when the file is absent.

    Returning None rather than raising keeps the overlay optional: a missing
    stats file should degrade to no bars, not break dataset generation.
    """
    stats_path = Path(path)
    if not stats_path.exists():
        return None
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_stats(stats: dict) -> None:
    print(
        f"[feature stats] {stats['sample_count']} samples, "
        f"{stats['frame_count']} frames, p{stats['percentile']:g} scaling"
    )
    header = f"{'feature':>30} {'min':>9} {'max':>9} {'scale':>9} {'signed':>7}"
    print(header)
    print("-" * len(header))
    for name, entry in stats["features"].items():
        print(
            f"{name:>30} {entry['min']:9.3f} {entry['max']:9.3f} "
            f"{entry['scale']:9.3f} {str(entry['signed']):>7}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure per-feature value ranges for the debug-video bars."
    )
    parser.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_FILE)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    stats = compute_feature_stats(
        args.samples_file,
        confidence_threshold=args.confidence_threshold,
        percentile=args.percentile,
    )
    print_stats(stats)
    save_feature_stats(stats, args.output)


if __name__ == "__main__":
    main()
