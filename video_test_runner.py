"""
Run pre-recorded videos through the offline tracker to verify ReID and
action-classification behaviour.

Entry point: golf_cv_video_test
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import cv2

import config
from action.feature_stats import load_feature_stats
from action.live_buffer import ActionSequenceBuffer
from action.preprocessing import (
    frame_feature_vector,
    keypoint_valid_mask,
    preprocess_pose_sequence,
)
from action.runtime import create_live_action_components, update_live_action
from reid.embedder import PersonEmbedder
from reid.matcher import ReIDMatcher
from tracker.person_tracker import PersonTracker
from ui.feature_overlay import draw_feature_panel
from ui.overlay import _COCO_SKELETON
from utils.annotations import (
    discover_videos,
    load_annotation_map,
    load_reference_embeddings,
)
from utils.geometry import box_area
from utils.video_io import SequentialFrameReader

REALISTIC_TARGET_FPS = 15
DEFAULT_NUM_RUNS = 3          # how many times to run each video
ENROLLMENT_FRAMES = 3
REID_AFTER_MISSING_FRAMES = 8
END_AFTER_MISSING_FRAMES = 90
DEFAULT_ANNOTATION_FILE = Path("data/annotations/video_intervals.json")
DEFAULT_ACTION_CONFIG = Path("action_classifier_config.toml")
DEFAULT_OUTPUT_DIR = Path("video_test")


class OfflineVideoRun:
    def __init__(
        self,
        video_path: Path,
        target_fps: int,
        embedder: PersonEmbedder,
        reference_embeddings=None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        write_video: bool = True,
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.embedder = embedder
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.write_video = write_video
        self.tracker = PersonTracker()
        self.matcher = ReIDMatcher(self.embedder)

        self.target_track_id = None
        self.tracked_person = None
        self.enrollment_embeddings = []
        self.enrollment_done = False
        self.missing_frames = 0
        self.last_scores = None
        self.ended_early = False

        if reference_embeddings:
            self.enrollment_embeddings = list(reference_embeddings)
            self.matcher.set_enrolled(self.enrollment_embeddings)
            self.enrollment_done = True
            print(
                f"[enroll refs] {self.video_path.name} {self.target_fps}fps "
                f"references={len(self.enrollment_embeddings)}"
            )

    def process(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[skip] Could not open {self.video_path}")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
        sample_indices = _sample_indices(
            total_frames,
            source_fps,
            self.target_fps,
            seed=_seed_for(self.video_path, self.target_fps),
        )
        writer = None
        output_path = None
        if self.write_video:
            writer_fps, frame_repeats = _writer_timing(self.target_fps)
            output_path = self._output_path()
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(writer_fps),
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                print(f"[skip] Could not create {output_path}")
                return
            print(
                f"[run] {self.video_path.name} -> {output_path.name} "
                f"({self.target_fps}fps sample, {writer_fps}fps encode, "
                f"{len(sample_indices)} samples)"
            )
        else:
            frame_repeats = 0

        reader = SequentialFrameReader(cap)
        for sample_number, frame_index in enumerate(sample_indices):
            frame = reader.read(frame_index)
            if frame is None:
                continue

            people = self.tracker.detect(frame)
            self._update_tracking(frame, people)
            if writer is not None:
                annotated = self._draw(frame.copy(), people, sample_number, frame_index)
                for _ in range(frame_repeats):
                    writer.write(annotated)

            if self.ended_early:
                break

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[done] {output_path}")

    def _update_tracking(self, frame, people):
        self.last_scores = None
        if not self.enrollment_done:
            self._collect_enrollment(frame, people)
            return

        if self.target_track_id is None:
            self._recover_with_reid(frame, people)
            return

        self.tracked_person = _person_by_track_id(people, self.target_track_id)
        if self.tracked_person is not None:
            self.missing_frames = 0
            return

        self.missing_frames += 1
        if self.missing_frames >= END_AFTER_MISSING_FRAMES:
            self.ended_early = True
            return

        if self.missing_frames < REID_AFTER_MISSING_FRAMES:
            return

        self._recover_with_reid(frame, people)

    def _collect_enrollment(self, frame, people):
        person = _largest_person(people)
        if person is None:
            self.tracked_person = None
            return

        self.tracked_person = person
        self.target_track_id = person["track_id"]
        embedding = self.embedder.embed(frame, person["box"])
        if embedding is None:
            return

        self.enrollment_embeddings.append(embedding)
        if len(self.enrollment_embeddings) >= ENROLLMENT_FRAMES:
            self.matcher.set_enrolled(self.enrollment_embeddings)
            self.enrollment_done = True
            print(
                f"[enroll] {self.video_path.name} {self.target_fps}fps "
                f"track_id={self.target_track_id}"
            )

    def _recover_with_reid(self, frame, people):
        if not people:
            return

        boxes = [person["box"] for person in people]
        embeddings = self.embedder.embed_many(frame, boxes)
        scores = {}
        for person, embedding in zip(people, embeddings):
            if embedding is None:
                continue
            _, score = self.matcher.match(embedding)
            scores[person["track_id"]] = score

        self.last_scores = scores
        best_id = self.matcher.select_best_match(scores)
        if best_id is None:
            return

        self.target_track_id = best_id
        self.tracked_person = _person_by_track_id(people, best_id)
        self.missing_frames = 0

    def _draw(self, frame, people, sample_number, source_frame_index):
        status = self._status_text()
        status_color = (0, 255, 0) if self.tracked_person is not None else (0, 180, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (20, 20, 20), -1)
        cv2.putText(
            frame,
            status,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            f"sample={sample_number} source_frame={source_frame_index} fps={self.target_fps}",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
        )

        for person in people:
            track_id = person["track_id"]
            selected = (
                track_id == self.target_track_id and self.tracked_person is not None
            )
            color = (0, 255, 0) if selected else (255, 100, 30)
            x1, y1, x2, y2 = person["box"]
            label = f"ID {track_id}"
            if self.last_scores is not None:
                label += f" reid={self.last_scores.get(track_id, 0.0):.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(96, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            _draw_skeleton(frame, person.get("keypoints"), color)

        return frame

    def _status_text(self):
        if not self.enrollment_done:
            return f"enrolling {len(self.enrollment_embeddings)}/{ENROLLMENT_FRAMES}"
        if self.tracked_person is not None:
            return f"tracking target ID {self.target_track_id}"
        if self.ended_early:
            return "target missing - video ended early"
        return f"target missing {self.missing_frames}/{END_AFTER_MISSING_FRAMES}"

    def _output_path(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{self.video_path.stem}_tracked_skeleton_{self.target_fps}fps.mp4"


class RealisticVideoRun(OfflineVideoRun):
    """
    Samples at ~15 fps with per-interval jitter and overlays the classified
    action label, mimicking real-time live-inference behaviour.

    Run the same video multiple times with different run_index values to get
    varied frame-timing samples and aggregate statistics.
    """

    def __init__(
        self,
        video_path: Path,
        run_index: int,
        embedder,
        reference_embeddings=None,
        action_config_path=DEFAULT_ACTION_CONFIG,
        output_dir: Path = None,
        write_video: bool = True,
        action_components=None,
    ):
        super().__init__(
            video_path,
            target_fps=REALISTIC_TARGET_FPS,
            embedder=embedder,
            reference_embeddings=reference_embeddings,
            output_dir=output_dir,
            write_video=write_video,
        )
        self.run_index = run_index
        if action_components is not None:
            # Reuse an already-loaded classifier/config (expensive checkpoint
            # read) across runs; the buffer still must be per-run state.
            classifier, config_dict = action_components
            self.action_classifier = classifier
            self.action_config = config_dict
            self.action_buffer = (
                None
                if classifier is None
                else ActionSequenceBuffer(
                    sequence_length=classifier.sequence_length,
                    target_fps=config_dict["inference"]["target_fps"],
                    classify_stride_sec=config_dict["inference"]["classify_stride_sec"],
                    max_gap_sec=config_dict["inference"]["max_gap_sec"],
                )
            )
        else:
            self.action_classifier, self.action_buffer, self.action_config = (
                create_live_action_components(action_config_path)
            )
        self.action_prediction = None
        self._confident_swings = 0
        self.swing_events = []  # [{"timestamp": float, "confidence": float}, ...]
        # Only needed when rendering; evaluate_detection and the sweep run with
        # write_video=False and would otherwise pay the file read per run.
        self._feature_stats = load_feature_stats() if write_video else None

    def process(self) -> dict:
        """
        Run the video and return a stats dict:
            frames_total, frames_tracked, track_rate, swing_count, ended_early, swing_events
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[skip] Could not open {self.video_path}")
            return {}

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
        sample_indices = _sample_indices(
            total_frames,
            source_fps,
            REALISTIC_TARGET_FPS,
            seed=_seed_for(self.video_path, self.run_index),
        )
        writer = None
        output_path = None
        if self.write_video:
            writer_fps, frame_repeats = _writer_timing(REALISTIC_TARGET_FPS)
            output_path = self._output_path()
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(writer_fps),
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                print(f"[skip] Could not create {output_path}")
                return {}
            print(
                f"[run {self.run_index}] {self.video_path.name} -> {output_path.name} "
                f"({REALISTIC_TARGET_FPS}fps ±jitter, {len(sample_indices)} samples)"
            )
        else:
            frame_repeats = 0
            print(
                f"[run {self.run_index}] {self.video_path.name} "
                f"({REALISTIC_TARGET_FPS}fps ±jitter, {len(sample_indices)} samples, no video output)"
            )

        frames_tracked = 0
        reader = SequentialFrameReader(cap)
        for sample_number, frame_index in enumerate(sample_indices):
            frame = reader.read(frame_index)
            if frame is None:
                continue

            timestamp = frame_index / source_fps

            people = self.tracker.detect(frame)
            self._update_tracking(frame, people)
            self._update_action(frame, timestamp)
            if writer is not None:
                annotated = self._draw(frame.copy(), people, sample_number, frame_index)
                for _ in range(frame_repeats):
                    writer.write(annotated)

            if self.tracked_person is not None:
                frames_tracked += 1

            if self.ended_early:
                break

        cap.release()
        if writer is not None:
            writer.release()

        frames_total = sample_number + 1 if sample_indices else 0
        track_rate = frames_tracked / frames_total if frames_total else 0.0
        print(
            f"[run {self.run_index}] done  "
            f"track={track_rate:.0%}  swings={self._confident_swings}"
            + ("  [ended early]" if self.ended_early else "")
        )
        return {
            "frames_total": frames_total,
            "frames_tracked": frames_tracked,
            "track_rate": track_rate,
            "swing_count": self._confident_swings,
            "ended_early": self.ended_early,
            "swing_events": self.swing_events,
        }

    def _update_action(self, frame, timestamp):
        prediction = update_live_action(
            classifier=self.action_classifier,
            buffer=self.action_buffer,
            config=self.action_config,
            target_id=self.target_track_id,
            tracked_person=self.tracked_person,
            timestamp=timestamp,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        if prediction is not None:
            prev = self.action_prediction
            self.action_prediction = prediction
            # count leading edge of a confident swing
            if (
                prediction.is_confident
                and prediction.label == "swing"
                and (prev is None or not prev.is_confident or prev.label != "swing")
            ):
                self._confident_swings += 1
                self.swing_events.append(
                    {"timestamp": timestamp, "confidence": prediction.confidence}
                )

    def _draw(self, frame, people, sample_number, source_frame_index):
        frame = super()._draw(frame, people, sample_number, source_frame_index)

        if self.action_prediction is not None:
            label = (
                f"Action: {self.action_prediction.label} "
                f"{self.action_prediction.confidence:.2f}"
            )
            color = (
                (0, 200, 255)
                if self.action_prediction.is_confident
                else (180, 180, 180)
            )
            cv2.rectangle(frame, (0, 74), (frame.shape[1], 104), (20, 20, 20), -1)
            cv2.putText(
                frame,
                label,
                (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
            )

        self._draw_features(frame)
        return frame

    def _draw_features(self, frame):
        """
        Panel of the classifier's current input features.

        Sourced from the buffer's last classified window rather than the
        current frame: classification is throttled to classify_stride_sec, so
        this steps at ~4Hz and shows exactly what the live prediction was made
        from. The dataset debug clips are the per-frame view.
        """
        buffer = self.action_buffer
        sequence = getattr(buffer, "last_sequence", None) if buffer else None
        if sequence is None or self._feature_stats is None:
            return

        features = preprocess_pose_sequence(
            sequence,
            buffer.frame_width or frame.shape[1],
            buffer.frame_height or frame.shape[0],
            confidence_threshold=self.action_config["features"]["confidence_threshold"],
        )
        valid = keypoint_valid_mask(
            sequence[-1], self.action_config["features"]["confidence_threshold"]
        )
        draw_feature_panel(
            frame,
            frame_feature_vector(features[-1], valid),
            stats=self._feature_stats,
            title="features (last classified)",
            no_data=not bool(valid.any()),
        )

    def _output_path(self):
        return self.video_path.with_name(
            f"{self.video_path.stem}_test_run{self.run_index}.mp4"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run offline golf CV tests on pre-recorded videos."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Video file or folder of videos. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help=(
            "Annotation file with saved ReID reference crops. "
            f"Default: {DEFAULT_ANNOTATION_FILE}"
        ),
    )
    parser.add_argument(
        "--action-config",
        type=Path,
        default=DEFAULT_ACTION_CONFIG,
        help=f"Action classifier config. Default: {DEFAULT_ACTION_CONFIG}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of test runs per video (each uses a different jitter seed). Default: {DEFAULT_NUM_RUNS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for test videos. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args(argv)

    input_path = Path(
        args.input or input("Video file or folder: ").strip()
    ).expanduser()
    videos = discover_videos(input_path)
    if not videos:
        raise SystemExit(f"No videos found at {input_path}")

    embedder = PersonEmbedder()
    annotation_by_video = load_annotation_map(args.annotation_file)
    for video in videos:
        reference_embeddings = load_reference_embeddings(
            annotation_by_video.get(video.name),
            embedder,
            args.annotation_file.parent,
        )
        all_stats = []
        for run_index in range(1, args.runs + 1):
            run = RealisticVideoRun(
                video,
                run_index=run_index,
                embedder=embedder,
                reference_embeddings=reference_embeddings,
                action_config_path=args.action_config,
                output_dir=args.output_dir,
            )
            stats = run.process()
            if stats:
                all_stats.append(stats)

        if all_stats:
            avg_track = sum(s["track_rate"] for s in all_stats) / len(all_stats)
            avg_swings = sum(s["swing_count"] for s in all_stats) / len(all_stats)
            early_count = sum(1 for s in all_stats if s["ended_early"])
            print(
                f"[summary] {video.name}  runs={len(all_stats)}  "
                f"avg_track={avg_track:.0%}  avg_swings={avg_swings:.1f}"
                + (f"  ended_early={early_count}" if early_count else "")
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_indices(total_frames, source_fps, target_fps, seed):
    if total_frames <= 0:
        return []
    if source_fps <= target_fps:
        return list(range(total_frames))

    rng = random.Random(seed)
    duration = total_frames / source_fps
    t = 0.0
    indices = []
    previous = -1
    while t < duration:
        idx = min(total_frames - 1, int(round(t * source_fps)))
        if idx > previous:
            indices.append(idx)
            previous = idx
        jitter = rng.uniform(0.65, 1.35)
        t += (1.0 / target_fps) * jitter
    return indices


def _seed_for(video_path: Path, target_fps: int):
    digest = hashlib.sha256(f"{video_path}:{target_fps}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _writer_timing(target_fps):
    if target_fps < 10:
        return 24, max(1, round(24 / target_fps))
    return target_fps, 1


def _draw_skeleton(frame, keypoints, color):
    if keypoints is None:
        return

    thresh = config.POSE_KEYPOINT_CONFIDENCE_THRESHOLD
    for a, b in _COCO_SKELETON:
        if keypoints[a, 2] >= thresh and keypoints[b, 2] >= thresh:
            cv2.line(
                frame,
                (int(keypoints[a, 0]), int(keypoints[a, 1])),
                (int(keypoints[b, 0]), int(keypoints[b, 1])),
                color,
                2,
            )
    for i, (x, y, conf) in enumerate(keypoints):
        if conf < thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        cv2.putText(
            frame,
            str(i),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )


def _person_by_track_id(people, track_id):
    return next((person for person in people if person["track_id"] == track_id), None)


def _largest_person(people):
    return max(people, key=lambda person: box_area(person["box"]), default=None)


if __name__ == "__main__":
    main()
