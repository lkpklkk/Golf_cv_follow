"""
Sweep the swing decision threshold against data/hold_out/ in one inference pass.

Companion to evaluate_detection.py, which scores a single threshold. Running
that harness once per candidate would re-decode every video and re-run YOLO,
Re-ID and the classifier each time — minutes per threshold, almost all of it
recomputing identical work.

Nothing upstream of the decision depends on the decision: `_update_action` only
reads the classifier's output, and swing events are the rising edge of a
confident swing over the classification sequence (video_test_runner.py). So the
sweep records each classification's probabilities once per (video, run), then
replays the event logic per candidate threshold. The replay calls the real
`classify_probabilities`, so the max(confidence_threshold, class_threshold)
rule and the `other` fallback are production behavior, not a reimplementation.
Results are identical to running evaluate_detection.py per threshold.

Entry point: golf_cv_sweep_swing_threshold
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from action.evaluate_detection import (
    DEFAULT_GROUND_TRUTH_FILE,
    DEFAULT_HOLDOUT_DIR,
    DEFAULT_RUNS,
    DEFAULT_TOLERANCE_SEC,
    DEFAULT_WORKERS,
    _limit_per_task_threads,
    _video_duration_sec,
    aggregate_metrics,
    idle_seconds,
    load_ground_truth,
    match_events,
)
from action.runtime import create_live_action_components
from reid.embedder import PersonEmbedder
from utils.annotations import load_annotation_map, load_reference_embeddings
from video_test_runner import (
    DEFAULT_ACTION_CONFIG,
    DEFAULT_ANNOTATION_FILE,
    RealisticVideoRun,
)

DEFAULT_MIN_THRESHOLD = 0.60
DEFAULT_MAX_THRESHOLD = 0.95
DEFAULT_STEP = 0.05
SWING_LABEL = "swing"


# ---------------------------------------------------------------------------
# Recording the classification trace
# ---------------------------------------------------------------------------


class TracingVideoRun(RealisticVideoRun):
    """
    RealisticVideoRun that also records every classification it makes.

    `trace` is [(timestamp, probabilities), ...] in classification order — one
    entry per call where the buffer had a full sequence ready, which is exactly
    the sequence the live event logic sees.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[tuple[float, dict[str, float]]] = []

    def _update_action(self, frame, timestamp):
        previous = self.action_prediction
        super()._update_action(frame, timestamp)
        current = self.action_prediction
        # A new prediction object means the buffer emitted a sequence this
        # frame; unchanged (including still None) means it was throttled or
        # not yet full, and the live logic saw nothing either.
        if current is not None and current is not previous:
            self.trace.append((timestamp, dict(current.probabilities)))


def swing_events_from_trace(
    trace, classifier, confidence_threshold, fallback_label, swing_threshold
) -> list[float]:
    """
    Replay the live event logic over a recorded trace at one swing threshold.

    Mirrors RealisticVideoRun._update_action: an event is the leading edge of a
    confident `swing`, i.e. the previous prediction was not itself a confident
    swing. Returns event timestamps.
    """
    events = []
    previous_was_swing = False
    for timestamp, probabilities in trace:
        prediction = classifier.classify_probabilities(
            probabilities,
            confidence_threshold=confidence_threshold,
            fallback_label=fallback_label,
            class_thresholds={SWING_LABEL: float(swing_threshold)},
        )
        is_swing = prediction.is_confident and prediction.label == SWING_LABEL
        if is_swing and not previous_was_swing:
            events.append(timestamp)
        previous_was_swing = is_swing
    return events


def build_thresholds(min_threshold, max_threshold, step) -> list[float]:
    if step <= 0:
        raise SystemExit("--step must be positive")
    if max_threshold < min_threshold:
        raise SystemExit("--max-threshold must be >= --min-threshold")

    thresholds = []
    # Rounded at every step: repeated float addition drifts, and a threshold
    # printed as 0.85 must be exactly the 0.85 that was scored.
    value = round(min_threshold, 4)
    while value <= max_threshold + 1e-9:
        thresholds.append(round(value, 4))
        value = round(value + step, 4)
    return thresholds


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _trace_single(
    video_path: Path,
    run_index: int,
    embedder,
    reference_embeddings,
    action_components,
) -> list[tuple[float, dict[str, float]]]:
    run = TracingVideoRun(
        video_path,
        run_index=run_index,
        embedder=embedder,
        reference_embeddings=reference_embeddings,
        write_video=False,
        action_components=action_components,
    )
    stats = run.process()
    return run.trace, bool(stats.get("ended_early", False))


def collect_traces(
    video_dir: Path,
    ground_truth: dict,
    annotation_file: Path,
    embedder,
    action_components,
    runs: int,
    workers: int,
) -> tuple[dict, dict]:
    """
    Run inference once per (video, run). Returns
    (traces, video_meta, ended_early) where traces is
    {video_name: [trace_per_run, ...]} ordered by run index, video_meta is
    {video_name: (duration_sec, positives)}, and ended_early lists the runs
    that stopped before the end of the footage — their idle_sec comes from the
    full duration, so every metric below is optimistic for those runs.
    """
    annotation_by_video = load_annotation_map(annotation_file)

    tasks = []
    video_meta = {}
    for video_name, positives in ground_truth.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"[skip] ground truth references missing video: {video_path}")
            continue

        duration_sec = _video_duration_sec(video_path)
        video_meta[video_name] = (duration_sec, positives)
        reference_embeddings = load_reference_embeddings(
            annotation_by_video.get(video_name), embedder, annotation_file.parent
        )
        for run_index in range(1, runs + 1):
            tasks.append((video_name, video_path, reference_embeddings, run_index))

    print(f"[sweep] {len(tasks)} (video, run) inference tasks across {workers} worker(s)")

    indexed: dict[str, list[tuple[int, list]]] = {}
    ended_early = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_task = {
            pool.submit(
                _trace_single,
                video_path,
                run_index,
                embedder,
                reference_embeddings,
                action_components,
            ): (video_name, run_index)
            for video_name, video_path, reference_embeddings, run_index in tasks
        }
        for future in as_completed(future_to_task):
            video_name, run_index = future_to_task[future]
            trace, run_ended_early = future.result()
            indexed.setdefault(video_name, []).append((run_index, trace))
            if run_ended_early:
                ended_early.append(f"{video_name}#{run_index}")

    traces = {
        video_name: [trace for _run_index, trace in sorted(entries)]
        for video_name, entries in indexed.items()
    }
    return traces, video_meta, sorted(ended_early)


def score_threshold(
    traces: dict,
    video_meta: dict,
    classifier,
    inference: dict,
    swing_threshold: float,
    tolerance_sec: float,
) -> dict:
    """Score every recorded run at one swing threshold, pooled the same way
    evaluate_detection does."""
    video_reports = []
    for video_name, run_traces in traces.items():
        duration_sec, positives = video_meta[video_name]
        run_reports = []
        for trace in run_traces:
            event_times = swing_events_from_trace(
                trace,
                classifier,
                inference["confidence_threshold"],
                inference["fallback_label"],
                swing_threshold,
            )
            result = match_events(event_times, positives, tolerance_sec)
            run_reports.append(
                {
                    "tp": len(result["matches"]),
                    "fp": len(result["false_positives"]),
                    "positive_instances": len(positives),
                    "idle_sec": idle_seconds(
                        duration_sec, len(positives), tolerance_sec
                    ),
                    "latencies": [
                        event - gt for event, gt in result["matches"]
                    ],
                    # Kept for inspection: when recall moves with the threshold
                    # the question is always whether events appeared/vanished
                    # or merely shifted in time past the tolerance window, and
                    # only the unmatched timestamps answer it.
                    "missed": result["missed"],
                    "false_positive_times": result["false_positives"],
                }
            )

        summary = aggregate_metrics(run_reports)
        video_reports.append(
            {
                "video": video_name,
                **summary,
                "idle_sec": sum(r["idle_sec"] for r in run_reports),
                "latencies": [lat for r in run_reports for lat in r["latencies"]],
            }
        )

    overall = aggregate_metrics(video_reports)
    overall["threshold"] = float(swing_threshold)
    overall["f1"] = _f1(overall["precision"], overall["recall"])
    overall["videos"] = video_reports
    return overall


def save_trace_cache(path: Path, traces: dict, video_meta: dict, ended_early: list, runs: int) -> None:
    payload = {
        "runs": runs,
        "traces": traces,
        "video_meta": {name: list(meta) for name, meta in video_meta.items()},
        "ended_early": ended_early,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[sweep] wrote trace cache {path}")


def load_trace_cache(path: Path, runs: int):
    """
    Reload a previous run's classification traces.

    Frame sampling is seeded on (video, run_index), so a cache built with the
    same run count describes the same frames — replaying it is equivalent to
    re-running inference, and takes under a second instead of minutes. A
    different run count is a different sample, so it is rejected rather than
    silently reused.
    """
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cached_runs = int(payload.get("runs", 0))
    if cached_runs != runs:
        raise SystemExit(
            f"Trace cache {path} was built with runs={cached_runs}, "
            f"but --runs={runs} was requested. Delete it or match the run count."
        )

    traces = {
        video_name: [
            [(float(timestamp), dict(probabilities)) for timestamp, probabilities in trace]
            for trace in run_traces
        ]
        for video_name, run_traces in payload["traces"].items()
    }
    video_meta = {
        name: (float(duration), list(positives))
        for name, (duration, positives) in payload["video_meta"].items()
    }
    print(f"[sweep] loaded trace cache {path} (no inference needed)")
    return traces, video_meta, list(payload.get("ended_early", []))


def sweep(
    video_dir: Path,
    ground_truth_file: Path,
    thresholds: list[float],
    action_config_path: Path = DEFAULT_ACTION_CONFIG,
    annotation_file: Path = DEFAULT_ANNOTATION_FILE,
    runs: int = DEFAULT_RUNS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    workers: int = DEFAULT_WORKERS,
    trace_cache: Path | None = None,
) -> dict:
    if workers > 1:
        _limit_per_task_threads(workers)

    ground_truth = load_ground_truth(ground_truth_file)
    classifier, _unused_buffer, action_config = create_live_action_components(
        action_config_path
    )
    if classifier is None:
        raise SystemExit("No action classifier available; cannot sweep thresholds.")

    if trace_cache is not None and trace_cache.exists():
        traces, video_meta, ended_early = load_trace_cache(trace_cache, runs)
    else:
        traces, video_meta, ended_early = collect_traces(
            video_dir,
            ground_truth,
            annotation_file,
            PersonEmbedder(),
            (classifier, action_config),
            runs,
            workers,
        )
        if trace_cache is not None:
            save_trace_cache(trace_cache, traces, video_meta, ended_early, runs)
    if not traces:
        raise SystemExit("No videos produced a classification trace.")

    classification_count = sum(
        len(trace) for run_traces in traces.values() for trace in run_traces
    )
    print(
        f"[sweep] {classification_count} classifications recorded; "
        f"replaying {len(thresholds)} thresholds"
    )

    inference = action_config["inference"]
    rows = [
        score_threshold(
            traces, video_meta, classifier, inference, threshold, tolerance_sec
        )
        for threshold in thresholds
    ]
    return {
        "tolerance_sec": tolerance_sec,
        "runs_per_video": runs,
        "confidence_threshold": inference["confidence_threshold"],
        "classifications": classification_count,
        "ended_early": ended_early,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _f1(precision, recall):
    if not precision or not recall:
        return None
    return 2 * precision * recall / (precision + recall)


def _fmt_pct(value):
    return f"{value:6.1%}" if value is not None else "   n/a"


def _fmt(value, width=6, places=2):
    return f"{value:{width}.{places}f}" if value is not None else " " * (width - 3) + "n/a"


def print_report(report: dict, config_threshold=None):
    rows = report["rows"]
    print()
    print("=== Swing threshold sweep: data/hold_out ===")
    print(
        f"tolerance = ±{report['tolerance_sec']:.2f}s   "
        f"runs/video = {report['runs_per_video']}   "
        f"inference.confidence_threshold = {report['confidence_threshold']}   "
        f"classifications = {report['classifications']}"
    )
    print(
        "  NOTE: the effective gate is max(confidence_threshold, swing threshold), "
        "so rows below the former are identical."
    )
    if report.get("ended_early"):
        print(
            "  WARN: runs that stopped before the end of the footage — idle_sec "
            "still counts the full duration, so fp/min is understated for these: "
            + ", ".join(report["ended_early"])
        )
    print()
    header = (
        f"{'thresh':>7}  {'recall':>7}  {'prec':>7}  {'F1':>7}  "
        f"{'fp/min':>7}  {'lat(s)':>7}  {'tp':>5}  {'fp':>5}"
    )
    print(header)
    print("-" * len(header))

    best_f1 = max(
        (row for row in rows if row["f1"] is not None),
        key=lambda row: row["f1"],
        default=None,
    )
    for row in rows:
        marks = []
        if best_f1 is not None and row is best_f1:
            marks.append("<- best F1")
        if config_threshold is not None and abs(row["threshold"] - config_threshold) < 1e-9:
            marks.append("<- config")
        print(
            f"{row['threshold']:7.2f}  {_fmt_pct(row['recall'])}  "
            f"{_fmt_pct(row['precision'])}  {_fmt_pct(row['f1'])}  "
            f"{_fmt(row['fp_per_min_idle'])}  "
            f"{_fmt(row['median_latency_sec'])}  "
            f"{row['tp']:5d}  {row['fp']:5d}"
            + ("  " + " ".join(marks) if marks else "")
        )

    if best_f1 is not None:
        print()
        print(
            f"best F1: threshold={best_f1['threshold']:.2f}  "
            f"recall={_fmt_pct(best_f1['recall']).strip()}  "
            f"precision={_fmt_pct(best_f1['precision']).strip()}  "
            f"fp_per_min_idle={_fmt(best_f1['fp_per_min_idle']).strip()}"
        )
        print(
            "Set it in action_classifier_config.toml under "
            "[inference.class_thresholds] as swing = <threshold>."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the swing decision threshold against data/hold_out/. "
            "Runs inference once and replays the event logic per threshold."
        )
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_FILE)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument("--annotation-file", type=Path, default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--min-threshold", type=float, default=DEFAULT_MIN_THRESHOLD)
    parser.add_argument("--max-threshold", type=float, default=DEFAULT_MAX_THRESHOLD)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Explicit threshold list; overrides --min/--max/--step.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write the JSON report."
    )
    parser.add_argument(
        "--trace-cache",
        type=Path,
        default=None,
        help=(
            "Read classification traces from this file if it exists, otherwise "
            "run inference and write it. Replaying a cache is instant, so "
            "re-sweeping after changing --tolerance-sec or the threshold range "
            "costs nothing. Delete the file after retraining or changing the "
            "pose/tracking path."
        ),
    )
    args = parser.parse_args(argv)

    thresholds = args.thresholds or build_thresholds(
        args.min_threshold, args.max_threshold, args.step
    )
    report = sweep(
        args.video_dir,
        args.ground_truth,
        thresholds,
        action_config_path=args.action_config,
        annotation_file=args.annotation_file,
        runs=args.runs,
        tolerance_sec=args.tolerance_sec,
        workers=args.workers,
        trace_cache=args.trace_cache,
    )

    config_threshold = None
    if args.action_config.exists():
        from action.training import load_action_config

        config_threshold = (
            load_action_config(args.action_config)
            .get("inference", {})
            .get("class_thresholds", {})
            .get(SWING_LABEL)
        )
    print_report(report, config_threshold=config_threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.output}")


if __name__ == "__main__":
    main()
