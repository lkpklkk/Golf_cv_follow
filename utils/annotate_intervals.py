import argparse
import cv2
import json
from pathlib import Path

VIDEO_DIR = Path("data/raw_videos")
OUT_FILE = Path("data/annotations/video_intervals.json")
REFERENCE_FRAME_DIR = Path("data/annotations/embedding_frames")
WINDOW_NAME = "Golf Swing / Walk Annotator"

HOLDOUT_VIDEO_DIR = Path("data/hold_out")
EVENTS_OUT_FILE = HOLDOUT_VIDEO_DIR / "impact_events.json"
EVENTS_WINDOW_NAME = "Golf Impact Event Annotator"

SUPPORTED_EXTS = [".mp4", ".mov", ".avi", ".mkv"]
PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
DEFAULT_PLAYBACK_SPEED_INDEX = 2


def load_existing_annotations(out_file: Path):
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_annotations(out_file: Path, annotations):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)


def normalize_box(x1, y1, x2, y2, width, height):
    left = max(0, min(x1, x2))
    right = min(width, max(x1, x2))
    top = max(0, min(y1, y2))
    bottom = min(height, max(y1, y2))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def save_reference_frame(video_path, frame, frame_idx, sec, crop_box=None):
    video_dir = REFERENCE_FRAME_DIR / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = "crop" if crop_box is not None else "full"
    image_path = video_dir / f"frame_{frame_idx:06d}_{suffix}.jpg"
    if image_path.exists():
        existing_count = len(list(video_dir.glob(f"frame_{frame_idx:06d}_{suffix}_*.jpg")))
        image_path = video_dir / f"frame_{frame_idx:06d}_{suffix}_{existing_count + 1}.jpg"

    image = frame
    metadata = {
        "frame": frame_idx,
        "sec": round(sec, 3),
        "image": image_path.as_posix(),
    }
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        image = frame[y1:y2, x1:x2]
        metadata["crop_box"] = [x1, y1, x2, y2]
        metadata["kind"] = "crop"
    else:
        metadata["kind"] = "full_frame"

    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Could not save reference frame to {image_path}")
    return metadata


def get_video_files(video_dir: Path):
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(video_dir.glob(f"*{ext}"))
    return sorted(files)


def choose_video_queue(videos, annotations):
    if not annotations:
        return videos

    last_video = annotations[-1].get("video")
    if not last_video:
        return videos

    answer = input(f"Continue after last saved video ({last_video})? [Y/n]: ").strip().lower()
    if answer not in ("", "y", "yes"):
        return videos

    for i, video_path in enumerate(videos):
        if video_path.name == last_video:
            remaining = videos[i + 1 :]
            print(f"Continuing with {len(remaining)} video(s) after {last_video}.")
            return remaining

    print(f"Last saved video {last_video} was not found in {VIDEO_DIR}; using full list.")
    return videos


def seconds_to_frame(sec, fps):
    return int(sec * fps)


def frame_to_seconds(frame_idx, fps):
    if fps <= 0:
        return 0.0
    return frame_idx / fps


def get_current_frame_idx(cap):
    return max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)


def set_frame(cap, frame_idx, total_frames):
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def read_frame_at(cap, frame_idx, total_frames):
    frame_idx = max(0, min(frame_idx, total_frames - 1))
    set_frame(cap, frame_idx, total_frames)
    ret, frame = cap.read()
    if not ret:
        return None, frame_idx
    return frame, frame_idx


def latest_labeled_end_frame(intervals, fps):
    if not intervals:
        return 0
    return max(
        interval.get("end_frame", seconds_to_frame(interval.get("end_sec", 0.0), fps))
        for interval in intervals
    )


def playback_delay_ms(fps, speed):
    if fps <= 0:
        fps = 30
    return max(1, int(round(1000.0 / (fps * speed))))


class CropTool:
    def __init__(self):
        self.active = False
        self.dragging = False
        self.start = None
        self.end = None

    def begin(self):
        self.active = True
        self.dragging = False
        self.start = None
        self.end = None

    def cancel(self):
        self.active = False
        self.dragging = False
        self.start = None
        self.end = None

    def mouse_callback(self, event, x, y, flags, user_data):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)

    def box(self, frame_shape):
        if self.start is None or self.end is None:
            return None
        h, w = frame_shape[:2]
        return normalize_box(self.start[0], self.start[1], self.end[0], self.end[1], w, h)


def draw_overlay(
    frame,
    video_name,
    current_frame,
    total_frames,
    current_sec,
    intervals,
    active_label,
    active_start_frame,
    active_start_sec,
    playback_speed,
    reference_frame_count,
    crop_tool,
    paused,
):
    display = frame.copy()

    status = "PAUSED" if paused else "PLAYING"

    lines = [
        f"Video: {video_name}",
        f"Frame: {current_frame}/{max(0, total_frames - 1)} | Time: {current_sec:.2f}s | {status} | {playback_speed:g}x",
        f"Intervals: {len(intervals)} | Target refs: {reference_frame_count}",
        "SPACE play/pause | 1/2 swing | 3/4 walk | 5/6 other",
        "E crop target ref | Enter save crop | Esc cancel crop | [/] speed | ,/. frame | A/D jump | U/N/K/Q",
    ]

    if crop_tool.active:
        lines.append("CROP MODE: drag a tight box around the target person")

    if active_label is not None:
        lines.append(
            f"ACTIVE: {active_label} started at frame {active_start_frame}, {active_start_sec:.2f}s"
        )

    y = 30
    for line in lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    recent = intervals[-5:]
    y += 10
    cv2.putText(
        display,
        "Recent intervals:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    y += 28

    for interval in recent:
        text = (
            f"{interval['label']}: f{interval['start_frame']} "
            f"({interval['start_sec']:.2f}s) -> f{interval['end_frame']} "
            f"({interval['end_sec']:.2f}s)"
        )
        cv2.putText(
            display,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24

    crop_box = crop_tool.box(frame.shape)
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return display


def annotate_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return None, [], "error"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_to_seconds(total_frames, fps)

    intervals = []
    reference_frames = []

    paused = False
    playback_speed_index = DEFAULT_PLAYBACK_SPEED_INDEX
    crop_tool = CropTool()
    active_label = None
    active_start_frame = None
    active_start_sec = None

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, crop_tool.mouse_callback)

    print(f"\nAnnotating: {video_path.name}")
    print(f"FPS: {fps:.2f}, duration: {duration_sec:.2f}s")
    print("Controls:")
    print("SPACE = play/pause")
    print("1 = start swing")
    print("2 = end swing")
    print("3 = start walk")
    print("4 = end walk")
    print("5 = start other")
    print("6 = end other")
    print("E = crop current frame as target reference")
    print("Enter = save target crop")
    print("Esc = cancel target crop")
    print("[ = slower playback")
    print("] = faster playback")
    print(", = previous frame")
    print(". = next frame")
    print("A = back 5 sec, capped by latest labeled end")
    print("D = forward 5 sec")
    print("U = delete previous marking")
    print("N = save and next")
    print("K = skip")
    print("Q = save and quit")

    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
            last_frame = frame
        else:
            if last_frame is None:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame
            frame = last_frame

        current_frame = get_current_frame_idx(cap)
        current_sec = frame_to_seconds(current_frame, fps)

        display = draw_overlay(
            frame=frame,
            video_name=video_path.name,
            current_frame=current_frame,
            total_frames=total_frames,
            current_sec=current_sec,
            intervals=intervals,
            active_label=active_label,
            active_start_frame=active_start_frame,
            active_start_sec=active_start_sec,
            playback_speed=PLAYBACK_SPEEDS[playback_speed_index],
            reference_frame_count=len(reference_frames),
            crop_tool=crop_tool,
            paused=paused,
        )

        cv2.imshow(WINDOW_NAME, display)

        delay_ms = playback_delay_ms(fps, PLAYBACK_SPEEDS[playback_speed_index])
        if crop_tool.active:
            wait_ms = 20
        elif paused:
            wait_ms = 0
        else:
            wait_ms = delay_ms

        key = cv2.waitKey(wait_ms) & 0xFF

        # No key pressed during playback
        if key == 255:
            continue

        # Esc: cancel target crop
        if key == 27 and crop_tool.active:
            crop_tool.cancel()
            paused = True
            print("Cancelled target crop.")

        # Enter: save active target crop
        elif key in (10, 13) and crop_tool.active:
            crop_box = crop_tool.box(frame.shape)
            if crop_box is None:
                print("Draw a target crop before saving.")
            else:
                saved = save_reference_frame(
                    video_path,
                    frame,
                    current_frame,
                    current_sec,
                    crop_box=crop_box,
                )
                reference_frames.append(saved)
                crop_tool.cancel()
                paused = True
                print(
                    f"Saved target crop frame {current_frame} "
                    f"({current_sec:.2f}s) -> {saved['image']}"
                )

        # SPACE: play/pause
        elif key == ord(" "):
            paused = not paused
            if not paused:
                crop_tool.cancel()

        # [: slower playback
        elif key == ord("["):
            playback_speed_index = max(0, playback_speed_index - 1)
            print(f"Playback speed: {PLAYBACK_SPEEDS[playback_speed_index]:g}x")

        # ]: faster playback
        elif key == ord("]"):
            playback_speed_index = min(len(PLAYBACK_SPEEDS) - 1, playback_speed_index + 1)
            print(f"Playback speed: {PLAYBACK_SPEEDS[playback_speed_index]:g}x")

        # e: crop current frame as a target reference for later Re-ID/keypoint extraction
        elif key in (ord("e"), ord("E")):
            paused = True
            crop_tool.begin()
            print("Crop mode: drag a tight box around the target, then press Enter.")

        # 1: start swing
        elif key == ord("1"):
            if active_label is not None:
                print(f"Already marking {active_label}. End it first.")
            else:
                active_label = "swing"
                active_start_frame = current_frame
                active_start_sec = current_sec
                paused = True
                print(f"Started swing at frame {active_start_frame}, {active_start_sec:.2f}s")

        # 2: end swing
        elif key == ord("2"):
            if active_label != "swing":
                print("No active swing interval. Press 1 first.")
            else:
                end_frame = current_frame
                end_sec = current_sec
                if end_frame <= active_start_frame:
                    print("End time must be after start time.")
                else:
                    intervals.append(
                        {
                            "label": "swing",
                            "start_frame": active_start_frame,
                            "end_frame": end_frame,
                            "start_sec": round(active_start_sec, 3),
                            "end_sec": round(end_sec, 3),
                        }
                    )
                    print(
                        f"Saved swing: frame {active_start_frame} ({active_start_sec:.2f}s) "
                        f"-> frame {end_frame} ({end_sec:.2f}s)"
                    )
                    active_label = None
                    active_start_frame = None
                    active_start_sec = None
                    paused = True

        # 3: start walk
        elif key == ord("3"):
            if active_label is not None:
                print(f"Already marking {active_label}. End it first.")
            else:
                active_label = "walk"
                active_start_frame = current_frame
                active_start_sec = current_sec
                paused = True
                print(f"Started walk at frame {active_start_frame}, {active_start_sec:.2f}s")

        # 4: end walk
        elif key == ord("4"):
            if active_label != "walk":
                print("No active walk interval. Press 3 first.")
            else:
                end_frame = current_frame
                end_sec = current_sec
                if end_frame <= active_start_frame:
                    print("End time must be after start time.")
                else:
                    intervals.append(
                        {
                            "label": "walk",
                            "start_frame": active_start_frame,
                            "end_frame": end_frame,
                            "start_sec": round(active_start_sec, 3),
                            "end_sec": round(end_sec, 3),
                        }
                    )
                    print(
                        f"Saved walk: frame {active_start_frame} ({active_start_sec:.2f}s) "
                        f"-> frame {end_frame} ({end_sec:.2f}s)"
                    )
                    active_label = None
                    active_start_frame = None
                    active_start_sec = None
                    paused = True

        # 5: start other
        elif key == ord("5"):
            if active_label is not None:
                print(f"Already marking {active_label}. End it first.")
            else:
                active_label = "other"
                active_start_frame = current_frame
                active_start_sec = current_sec
                paused = True
                print(f"Started other at frame {active_start_frame}, {active_start_sec:.2f}s")

        # 6: end other
        elif key == ord("6"):
            if active_label != "other":
                print("No active other interval. Press 5 first.")
            else:
                end_frame = current_frame
                end_sec = current_sec
                if end_frame <= active_start_frame:
                    print("End time must be after start time.")
                else:
                    intervals.append(
                        {
                            "label": "other",
                            "start_frame": active_start_frame,
                            "end_frame": end_frame,
                            "start_sec": round(active_start_sec, 3),
                            "end_sec": round(end_sec, 3),
                        }
                    )
                    print(
                        f"Saved other: frame {active_start_frame} ({active_start_sec:.2f}s) "
                        f"-> frame {end_frame} ({end_sec:.2f}s)"
                    )
                    active_label = None
                    active_start_frame = None
                    active_start_sec = None
                    paused = True

        # u: undo previous completed interval
        elif key == ord("u"):
            if active_label is not None:
                print(f"Cancelled active {active_label} interval.")
                active_label = None
                active_start_frame = None
                active_start_sec = None
            elif intervals:
                removed = intervals.pop()
                print(f"Removed previous interval: {removed}")
            else:
                print("Nothing to undo.")

        # ,: previous frame
        elif key == ord(","):
            frame, target_frame = read_frame_at(
                cap,
                current_frame - 1,
                total_frames,
            )
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # .: next frame
        elif key == ord("."):
            frame, target_frame = read_frame_at(
                cap,
                current_frame + 1,
                total_frames,
            )
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # a: back 5 seconds
        elif key == ord("a"):
            current_frame = get_current_frame_idx(cap)

            # Important behavior:
            # Prevent going back before the latest completed labeled interval,
            # unless you undo/delete that previous marking first.
            min_back_frame = latest_labeled_end_frame(intervals, fps)

            target_frame = max(current_frame - seconds_to_frame(5.0, fps), min_back_frame)
            frame, target_frame = read_frame_at(cap, target_frame, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved back to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )


        # d: forward 5 seconds
        elif key == ord("d"):
            current_frame = get_current_frame_idx(cap)

            target_frame = min(current_frame + seconds_to_frame(5.0, fps), total_frames - 1)
            frame, target_frame = read_frame_at(cap, target_frame, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved forward to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # n: save and next
        elif key == ord("n"):
            print("Save and next.")
            cap.release()
            return intervals, reference_frames, "next"

        # k: skip current video
        elif key == ord("k"):
            print("Skipped video.")
            cap.release()
            return [], [], "skip"

        # q: save and quit whole program
        elif key == ord("q"):
            print("Save and quit.")
            cap.release()
            return intervals, reference_frames, "quit"

        # Left arrow / right arrow support varies by system.
        # Some OpenCV builds do not return reliable arrow key codes.
        # Use A/D if arrows do not work.

    cap.release()
    return intervals, reference_frames, "next"


# ---------------------------------------------------------------------------
# Impact-event mode
#
# Holdout footage is never trained on, so it never goes through window
# generation and never needs intervals. A single impact frame per swing is
# enough, and impact is visually unambiguous in a way that "when does a swing
# start" is not. Existing interval annotations are untouched by this mode.
# ---------------------------------------------------------------------------


def draw_events_overlay(
    frame,
    video_name,
    current_frame,
    total_frames,
    current_sec,
    impacts,
    practice_swings,
    playback_speed,
    paused,
):
    display = frame.copy()

    status = "PAUSED" if paused else "PLAYING"
    lines = [
        f"Video: {video_name}",
        f"Frame: {current_frame}/{max(0, total_frames - 1)} | Time: {current_sec:.2f}s | {status} | {playback_speed:g}x",
        f"Impacts: {len(impacts)} | Practice swings: {len(practice_swings)}",
        "SPACE play/pause | I impact | P practice swing | U undo",
        "[/] speed | ,/. frame | A/D jump 5s | N next | K skip | Q save+quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 28

    recent = sorted((*((s, "impact") for s in impacts), *((s, "practice") for s in practice_swings)))[-5:]
    y += 10
    cv2.putText(
        display,
        "Recent marks:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    y += 28
    for sec, kind in recent:
        cv2.putText(
            display,
            f"{kind}: {sec:.2f}s",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return display


def add_event_mark(marks, other_marks, sec, kind):
    """Append sec to marks unless this exact instant is already marked."""
    rounded = round(sec, 3)
    if rounded in marks:
        print(f"Frame already marked as {kind} ({rounded:.2f}s).")
        return False
    if rounded in other_marks:
        print(f"Frame already marked with the other event type ({rounded:.2f}s).")
        return False
    marks.append(rounded)
    print(f"Marked {kind} at {rounded:.2f}s")
    return True


def annotate_video_events(video_path: Path, existing_entry=None):
    """
    Mark impact instants in a video.

    Returns (impacts_sec, practice_swings_sec, fps, status).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return [], [], 0.0, "error"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_to_seconds(total_frames, fps)

    existing_entry = existing_entry or {}
    impacts = [round(float(s), 3) for s in existing_entry.get("impacts_sec", [])]
    practice_swings = [
        round(float(s), 3) for s in existing_entry.get("practice_swings_sec", [])
    ]
    # Marks are appended in the order they were made so U undoes the last one.
    mark_order = [(s, "impact") for s in impacts] + [
        (s, "practice") for s in practice_swings
    ]

    paused = False
    playback_speed_index = DEFAULT_PLAYBACK_SPEED_INDEX

    cv2.namedWindow(EVENTS_WINDOW_NAME)

    print(f"\nLabeling impacts: {video_path.name}")
    print(f"FPS: {fps:.2f}, duration: {duration_sec:.2f}s")
    if mark_order:
        print(f"Loaded {len(impacts)} impact(s), {len(practice_swings)} practice swing(s).")
    print("Controls:")
    print("SPACE = play/pause")
    print("I = mark current frame as impact")
    print("P = mark current frame as practice swing")
    print("U = delete last mark")
    print("[ / ] = slower / faster playback")
    print(", / . = previous / next frame")
    print("A / D = back / forward 5 sec")
    print("N = save and next")
    print("K = skip (discard marks for this video)")
    print("Q = save and quit")

    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break
            last_frame = frame
        else:
            if last_frame is None:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame
            frame = last_frame

        current_frame = get_current_frame_idx(cap)
        current_sec = frame_to_seconds(current_frame, fps)

        display = draw_events_overlay(
            frame=frame,
            video_name=video_path.name,
            current_frame=current_frame,
            total_frames=total_frames,
            current_sec=current_sec,
            impacts=impacts,
            practice_swings=practice_swings,
            playback_speed=PLAYBACK_SPEEDS[playback_speed_index],
            paused=paused,
        )
        cv2.imshow(EVENTS_WINDOW_NAME, display)

        wait_ms = (
            0
            if paused
            else playback_delay_ms(fps, PLAYBACK_SPEEDS[playback_speed_index])
        )
        key = cv2.waitKey(wait_ms) & 0xFF

        # No key pressed during playback
        if key == 255:
            continue

        # SPACE: play/pause
        if key == ord(" "):
            paused = not paused

        # i: mark impact
        elif key in (ord("i"), ord("I")):
            paused = True
            if add_event_mark(impacts, practice_swings, current_sec, "impact"):
                mark_order.append((round(current_sec, 3), "impact"))

        # p: mark practice swing
        elif key in (ord("p"), ord("P")):
            paused = True
            if add_event_mark(practice_swings, impacts, current_sec, "practice"):
                mark_order.append((round(current_sec, 3), "practice"))

        # u: delete last mark
        elif key in (ord("u"), ord("U")):
            if not mark_order:
                print("Nothing to undo.")
            else:
                sec, kind = mark_order.pop()
                target = impacts if kind == "impact" else practice_swings
                if sec in target:
                    target.remove(sec)
                print(f"Removed {kind} mark at {sec:.2f}s")

        # [: slower playback
        elif key == ord("["):
            playback_speed_index = max(0, playback_speed_index - 1)
            print(f"Playback speed: {PLAYBACK_SPEEDS[playback_speed_index]:g}x")

        # ]: faster playback
        elif key == ord("]"):
            playback_speed_index = min(
                len(PLAYBACK_SPEEDS) - 1, playback_speed_index + 1
            )
            print(f"Playback speed: {PLAYBACK_SPEEDS[playback_speed_index]:g}x")

        # ,: previous frame
        elif key == ord(","):
            frame, target_frame = read_frame_at(cap, current_frame - 1, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # .: next frame
        elif key == ord("."):
            frame, target_frame = read_frame_at(cap, current_frame + 1, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # a: back 5 seconds.
        # Unlike interval mode there is no floor here: marks are instants, so
        # seeking back past one cannot leave a half-open annotation.
        elif key in (ord("a"), ord("A")):
            target_frame = max(0, current_frame - seconds_to_frame(5.0, fps))
            frame, target_frame = read_frame_at(cap, target_frame, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved back to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # d: forward 5 seconds
        elif key in (ord("d"), ord("D")):
            target_frame = min(
                current_frame + seconds_to_frame(5.0, fps), total_frames - 1
            )
            frame, target_frame = read_frame_at(cap, target_frame, total_frames)
            if frame is not None:
                last_frame = frame
                paused = True
                print(
                    f"Moved forward to frame {target_frame}, "
                    f"{frame_to_seconds(target_frame, fps):.2f}s"
                )

        # n: save and next
        elif key in (ord("n"), ord("N")):
            print("Save and next.")
            cap.release()
            return sorted(impacts), sorted(practice_swings), fps, "next"

        # k: skip current video
        elif key in (ord("k"), ord("K")):
            print("Skipped video.")
            cap.release()
            return [], [], fps, "skip"

        # q: save and quit whole program
        elif key in (ord("q"), ord("Q")):
            print("Save and quit.")
            cap.release()
            return sorted(impacts), sorted(practice_swings), fps, "quit"

    cap.release()
    return sorted(impacts), sorted(practice_swings), fps, "next"


def merge_or_replace_events(events, video_name, fps, impacts, practice_swings):
    events = [e for e in events if e.get("video") != video_name]
    events.append(
        {
            "video": video_name,
            "fps": round(float(fps), 3),
            "impacts_sec": sorted(round(float(s), 3) for s in impacts),
            "practice_swings_sec": sorted(round(float(s), 3) for s in practice_swings),
        }
    )
    return sorted(events, key=lambda e: e["video"])


def events_main(video_dir: Path, out_file: Path):
    videos = get_video_files(video_dir)
    if not videos:
        print(f"No videos found in {video_dir}")
        return

    events = load_existing_annotations(out_file)
    events_by_video = {e.get("video"): e for e in events if e.get("video")}

    print(f"Found {len(videos)} videos in {video_dir}.")
    print(f"Existing event annotations: {len(events_by_video)}")

    for video_path in videos:
        existing = events_by_video.get(video_path.name)
        if existing is not None:
            answer = (
                input(f"{video_path.name} already labeled. Relabel? [y/N]: ")
                .strip()
                .lower()
            )
            if answer not in ("y", "yes"):
                continue

        impacts, practice_swings, fps, status = annotate_video_events(
            video_path, existing_entry=existing
        )
        if status == "error":
            continue
        if status == "skip":
            continue

        events = merge_or_replace_events(
            events, video_path.name, fps, impacts, practice_swings
        )
        events_by_video = {e.get("video"): e for e in events if e.get("video")}
        save_annotations(out_file, events)
        print(
            f"Saved {len(impacts)} impact(s) and {len(practice_swings)} practice "
            f"swing(s) for {video_path.name} to {out_file}"
        )

        if status == "quit":
            break

    cv2.destroyAllWindows()
    print("Done.")


def merge_or_replace_annotation(
    annotations,
    video_name,
    new_intervals,
    reference_frames,
    skipped=False,
):
    annotations = [a for a in annotations if a["video"] != video_name]

    annotations.append(
        {
            "video": video_name,
            "skipped": skipped,
            "intervals": new_intervals,
            "reference_frames": reference_frames,
        }
    )

    return annotations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Annotate golf footage: action intervals, or impact events."
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help=(
            "Impact-event mode: mark a single frame per swing instead of "
            f"intervals. Defaults to {HOLDOUT_VIDEO_DIR} -> {EVENTS_OUT_FILE}."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory of videos to annotate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output annotation JSON file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.events:
        events_main(
            video_dir=args.video_dir or HOLDOUT_VIDEO_DIR,
            out_file=args.out or EVENTS_OUT_FILE,
        )
        return

    video_dir = args.video_dir or VIDEO_DIR
    out_file = args.out or OUT_FILE

    videos = get_video_files(video_dir)

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    annotations = load_existing_annotations(out_file)
    already_annotated = {a["video"] for a in annotations}
    videos = choose_video_queue(videos, annotations)

    print(f"Found {len(videos)} videos.")
    print(f"Existing annotations: {len(already_annotated)}")

    for video_path in videos:
        # Skip already annotated videos by default.
        # Comment this out if you want to re-annotate everything.
        if video_path.name in already_annotated:
            print(f"Already annotated, skipping: {video_path.name}")
            continue

        intervals, reference_frames, status = annotate_video(video_path)

        if status == "error":
            continue

        skipped = status == "skip"

        annotations = merge_or_replace_annotation(
            annotations=annotations,
            video_name=video_path.name,
            new_intervals=intervals,
            reference_frames=reference_frames,
            skipped=skipped,
        )

        save_annotations(out_file, annotations)
        print(f"Saved annotations to {out_file}")

        if status == "quit":
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
