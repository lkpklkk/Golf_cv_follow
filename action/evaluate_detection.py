"""
Detection-metric harness for swing/impact detection on data/hold_out/.

This is Stage D of ACTION_PIPELINE_REWORK.md. Training/window-level metrics
(accuracy, macro F1 over generated windows) are dominated by how many windows
the sampler happened to emit and are computed against labels produced by the
same generator being evaluated (see doc section 1.4) — they cannot validate
detection quality. This harness instead runs the real inference path
(RealisticVideoRun, same as `golf_cv_video_test`) on hand-labeled holdout
footage and matches predicted swing events to `impact_events.json` with a
tolerance window, greedy nearest-first.

Ground truth positives = impacts_sec UNION practice_swings_sec (see
ACTION_PIPELINE_REWORK.md section 1.5): a fire on a practice swing is an
expected detection, not a false positive.

Reports, pooled across videos and runs:
    - recall            fraction of labeled swing events detected
    - precision         fraction of detections that matched a labeled event
    - fp_per_min_idle   false positives per minute of non-swing footage
    - median_latency_sec  median (detection_time - matched_event_time)

Entry point: golf_cv_evaluate_detection
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from action.runtime import create_live_action_components
from reid.embedder import PersonEmbedder
from utils.annotations import load_annotation_map, load_reference_embeddings
from video_test_runner import (
    DEFAULT_ACTION_CONFIG,
    DEFAULT_ANNOTATION_FILE,
    RealisticVideoRun,
)

DEFAULT_HOLDOUT_DIR = Path("data/hold_out")
DEFAULT_GROUND_TRUTH_FILE = Path("data/hold_out/impact_events.json")
DEFAULT_TOLERANCE_SEC = 1.0
DEFAULT_RUNS = 3
DEFAULT_WORKERS = 1  # see _limit_per_task_threads: measured no win on MPS, GPU-bound


# ---------------------------------------------------------------------------
# Pure matching / metrics logic (no video I/O, unit-testable)
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> dict[str, list[float]]:
    """
    Return {video_filename: sorted positive event timestamps}.

    Positives are impacts_sec union practice_swings_sec — see module docstring.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        entries = json.load(f)

    ground_truth = {}
    for entry in entries:
        video = entry.get("video")
        if not video:
            continue
        positives = list(entry.get("impacts_sec") or []) + list(
            entry.get("practice_swings_sec") or []
        )
        ground_truth[video] = sorted(positives)
    return ground_truth


def match_events(
    event_times: list[float],
    positive_times: list[float],
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
) -> dict:
    """
    Greedy nearest-first matching between detected event timestamps and
    labeled positive timestamps, within +/- tolerance_sec.

    Returns:
        matches: [(event_time, positive_time), ...] sorted by event_time
        false_positives: unmatched event_times
        missed: unmatched positive_times
    """
    candidates = []
    for e_idx, e_time in enumerate(event_times):
        for p_idx, p_time in enumerate(positive_times):
            diff = abs(e_time - p_time)
            if diff <= tolerance_sec:
                candidates.append((diff, e_idx, p_idx))
    candidates.sort(key=lambda c: c[0])

    matched_events = set()
    matched_positives = set()
    matches = []
    for _diff, e_idx, p_idx in candidates:
        if e_idx in matched_events or p_idx in matched_positives:
            continue
        matched_events.add(e_idx)
        matched_positives.add(p_idx)
        matches.append((event_times[e_idx], positive_times[p_idx]))

    false_positives = [
        e_time for i, e_time in enumerate(event_times) if i not in matched_events
    ]
    missed = [
        p_time for i, p_time in enumerate(positive_times) if i not in matched_positives
    ]
    matches.sort(key=lambda pair: pair[0])
    return {
        "matches": matches,
        "false_positives": false_positives,
        "missed": missed,
    }


def idle_seconds(duration_sec: float, positive_count: int, tolerance_sec: float) -> float:
    """
    Approximate non-swing footage duration: total duration minus a
    tolerance-window-sized band around each labeled event. Holdout ground
    truth marks instants, not intervals, so this is the best available
    denominator for a false-positive rate (see ACTION_PIPELINE_REWORK.md
    section on the FP-per-idle-minute denominator being coarse).
    """
    return max(0.0, duration_sec - positive_count * 2 * tolerance_sec)


def aggregate_metrics(video_reports: list[dict]) -> dict:
    """Pool per-run/per-video counts into one summary dict."""
    total_tp = sum(r["tp"] for r in video_reports)
    total_fp = sum(r["fp"] for r in video_reports)
    total_positive_instances = sum(r["positive_instances"] for r in video_reports)
    total_idle_sec = sum(r["idle_sec"] for r in video_reports)
    all_latencies = [lat for r in video_reports for lat in r["latencies"]]

    recall = total_tp / total_positive_instances if total_positive_instances else None
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    fp_per_min_idle = (
        total_fp / (total_idle_sec / 60.0) if total_idle_sec > 0 else None
    )
    median_latency = statistics.median(all_latencies) if all_latencies else None

    return {
        "tp": total_tp,
        "fp": total_fp,
        "positive_instances": total_positive_instances,
        "recall": recall,
        "precision": precision,
        "fp_per_min_idle": fp_per_min_idle,
        "median_latency_sec": median_latency,
    }


# ---------------------------------------------------------------------------
# Video I/O driven evaluation
# ---------------------------------------------------------------------------


def _limit_per_task_threads(workers: int) -> None:
    """
    cv2 and torch default to using every core for a single call. That's fine
    serially, but with N concurrent tasks it means N-way contention for the
    same cores rather than real parallelism. Cap each library to roughly
    cpu_count / workers so concurrent tasks divide the machine instead of
    fighting over it.
    """
    per_task = max(1, (os.cpu_count() or workers) // workers)
    cv2.setNumThreads(per_task)
    try:
        import torch

        torch.set_num_threads(per_task)
    except ImportError:
        pass


def _video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return total_frames / fps if fps else 0.0


def _run_single(
    video_path: Path,
    positives: list[float],
    run_index: int,
    embedder,
    reference_embeddings,
    action_components,
    duration_sec: float,
    tolerance_sec: float,
) -> dict:
    """Execute one (video, run) task. Safe to call concurrently: the tracker
    and action buffer are per-call state; only `embedder` and the loaded
    classifier/config inside `action_components` are shared read-only."""
    run = RealisticVideoRun(
        video_path,
        run_index=run_index,
        embedder=embedder,
        reference_embeddings=reference_embeddings,
        write_video=False,
        action_components=action_components,
    )
    stats = run.process()
    event_times = [e["timestamp"] for e in stats.get("swing_events", [])]
    result = match_events(event_times, positives, tolerance_sec)
    latencies = [event - gt for event, gt in result["matches"]]
    return {
        "tp": len(result["matches"]),
        "fp": len(result["false_positives"]),
        "positive_instances": len(positives),
        "idle_sec": idle_seconds(duration_sec, len(positives), tolerance_sec),
        "latencies": latencies,
        "missed": result["missed"],
        "false_positive_times": result["false_positives"],
        "ended_early": stats.get("ended_early", False),
    }


def _build_video_report(video_name: str, duration_sec: float, positives: list, run_reports: list) -> dict:
    run_reports = sorted(run_reports, key=lambda r: r["_run_index"])
    for r in run_reports:
        del r["_run_index"]
    summary = aggregate_metrics(run_reports)
    return {
        "video": video_name,
        "duration_sec": duration_sec,
        "positives": positives,
        "runs": run_reports,
        # Pooled raw fields so evaluate_holdout can re-aggregate across videos
        # with the same aggregate_metrics() contract used per-video above.
        "tp": summary["tp"],
        "fp": summary["fp"],
        "positive_instances": summary["positive_instances"],
        "idle_sec": sum(r["idle_sec"] for r in run_reports),
        "latencies": [lat for r in run_reports for lat in r["latencies"]],
        **summary,
    }


def evaluate_holdout(
    video_dir: Path,
    ground_truth_file: Path,
    action_config_path: Path = DEFAULT_ACTION_CONFIG,
    annotation_file: Path = DEFAULT_ANNOTATION_FILE,
    runs: int = DEFAULT_RUNS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    if workers > 1:
        _limit_per_task_threads(workers)

    ground_truth = load_ground_truth(ground_truth_file)
    annotation_by_video = load_annotation_map(annotation_file)
    embedder = PersonEmbedder()

    # Load the classifier checkpoint once and share it (read-only inference)
    # across every task instead of re-reading it from disk per run.
    classifier, _unused_buffer, action_config = create_live_action_components(
        action_config_path
    )
    action_components = (classifier, action_config)

    tasks = []  # [(video_name, video_path, positives, duration_sec, run_index), ...]
    for video_name, positives in ground_truth.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"[skip] ground truth references missing video: {video_path}")
            continue

        duration_sec = _video_duration_sec(video_path)
        reference_embeddings = load_reference_embeddings(
            annotation_by_video.get(video_name), embedder, annotation_file.parent
        )
        for run_index in range(1, runs + 1):
            tasks.append(
                (video_name, video_path, positives, duration_sec, reference_embeddings, run_index)
            )

    print(f"[eval] {len(tasks)} (video, run) tasks across {workers} worker(s)")

    results_by_video: dict[str, list[dict]] = {}
    video_meta: dict[str, tuple[float, list[float]]] = {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_task = {
            pool.submit(
                _run_single,
                video_path,
                positives,
                run_index,
                embedder,
                reference_embeddings,
                action_components,
                duration_sec,
                tolerance_sec,
            ): (video_name, run_index)
            for video_name, video_path, positives, duration_sec, reference_embeddings, run_index in tasks
        }
        for future in as_completed(future_to_task):
            video_name, run_index = future_to_task[future]
            run_report = future.result()
            run_report["_run_index"] = run_index
            results_by_video.setdefault(video_name, []).append(run_report)

    for video_name, video_path, positives, duration_sec, _refs, _run_index in tasks:
        video_meta[video_name] = (duration_sec, positives)

    video_reports = []
    for video_name in ground_truth:
        if video_name not in results_by_video:
            continue
        duration_sec, positives = video_meta[video_name]
        report = _build_video_report(
            video_name, duration_sec, positives, results_by_video[video_name]
        )
        video_reports.append(report)
        print(
            f"[eval] {video_name}  positives={len(positives)}  runs={runs}  "
            f"recall={_fmt_pct(report['recall'])}  "
            f"precision={_fmt_pct(report['precision'])}  "
            f"fp/min_idle={_fmt(report['fp_per_min_idle'])}"
        )

    overall = aggregate_metrics(video_reports)
    return {
        "tolerance_sec": tolerance_sec,
        "runs_per_video": runs,
        "workers": workers,
        "videos": video_reports,
        "overall": overall,
    }


def _fmt_pct(value):
    return f"{value:.0%}" if value is not None else "n/a"


def _fmt(value):
    return f"{value:.2f}" if value is not None else "n/a"


def print_report(report: dict):
    overall = report["overall"]
    print()
    print("=== Swing detection eval: data/hold_out ===")
    print(f"tolerance = ±{report['tolerance_sec']:.2f}s   runs/video = {report['runs_per_video']}")
    print(
        f"recall={_fmt_pct(overall['recall'])}  "
        f"precision={_fmt_pct(overall['precision'])}  "
        f"fp_per_min_idle={_fmt(overall['fp_per_min_idle'])}  "
        f"median_latency_sec={_fmt(overall['median_latency_sec'])}  "
        f"(tp={overall['tp']} fp={overall['fp']} positives={overall['positive_instances']})"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score swing detection against data/hold_out/impact_events.json."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_FILE)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Concurrent (video, run) tasks. Default: {DEFAULT_WORKERS} — on an "
            "MPS (Apple GPU) backend the GPU serializes work across threads "
            "anyway, so concurrency added contention with no wall-clock win in "
            "testing. Worth trying >1 on a CPU-only or CUDA machine."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the full JSON report."
    )
    args = parser.parse_args(argv)

    report = evaluate_holdout(
        args.video_dir,
        args.ground_truth,
        action_config_path=args.action_config,
        annotation_file=args.annotation_file,
        runs=args.runs,
        tolerance_sec=args.tolerance_sec,
        workers=args.workers,
    )
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
