import cv2
import config
from control.steering import get_movement_command


_COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def _draw_pose_keypoints(frame, keypoints, color):
    if keypoints is None or not config.DRAW_POSE_KEYPOINTS:
        return

    conf_thresh = config.POSE_KEYPOINT_CONFIDENCE_THRESHOLD

    for a, b in _COCO_SKELETON:
        if keypoints[a, 2] < conf_thresh or keypoints[b, 2] < conf_thresh:
            continue
        ax, ay = int(keypoints[a, 0]), int(keypoints[a, 1])
        bx, by = int(keypoints[b, 0]), int(keypoints[b, 1])
        cv2.line(frame, (ax, ay), (bx, by), color, 2)

    for i, (x, y, conf) in enumerate(keypoints):
        if conf < conf_thresh:
            continue
        center = (int(x), int(y))
        cv2.circle(frame, center, 4, (255, 255, 255), -1)
        cv2.circle(frame, center, 3, color, -1)
        cv2.putText(
            frame,
            str(i),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )


def draw(
    frame,
    people,
    markers,
    tracked_person,
    selected_track_id,
    gesture=None,
    fps=None,
    reid_matches=None,
    enrolling_id=None,
    enroll_status=None,
):
    """
    Render all overlays onto the frame in-place and return it.

    Args:
        frame:             BGR numpy array
        people:            list of person dicts from PersonTracker
        markers:           list of marker dicts from ArucoTracker
        tracked_person:    the currently selected/matched person dict, or None
        selected_track_id: int or None
        gesture:           gesture label string or None
        fps:               float or None
        reid_matches:      dict {track_id: similarity_score} when re-id is active,
                           None when not yet enrolled (falls back to click-selection colour)
        enrolling_id:      track_id currently being enrolled (draw yellow), or None
        enroll_status:     short status string to show top-left, or None
    """
    frame_h, frame_w = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)

    # --- People bounding boxes ---
    for person in people:
        x1, y1, x2, y2 = person["box"]
        cx, cy = person["center"]
        track_id = person["track_id"]
        confidence = person["confidence"]

        # Priority: yellow = currently enrolling, then reid colour, then click-select
        if track_id == enrolling_id:
            color = (0, 255, 255)  # yellow
            label = f"ENROLLING ID {track_id}"
        elif reid_matches is not None:
            score = reid_matches.get(track_id, 0.0)
            is_match = score >= config.REID_SIMILARITY_THRESHOLD
            color = (0, 255, 0) if is_match else (255, 0, 0)
            tag = "TARGET" if is_match else "Person"
            label = f"{tag} {track_id}  sim={score:.2f}"
        else:
            # Re-ID not yet enrolled — fall back to click selection
            is_selected = track_id == selected_track_id
            color = (0, 255, 0) if is_selected else (255, 0, 0)
            label = (
                f"TRACKING ID {track_id} {confidence:.2f}"
                if is_selected
                else f"Person ID {track_id} {confidence:.2f}"
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        _draw_pose_keypoints(frame, person.get("keypoints"), color)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # --- ArUco markers ---
    for marker in markers:
        pts = marker["corners"]
        marker_id = marker["id"]
        cx, cy = marker["center"]
        assoc_id = marker.get("track_id")

        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"ArUco {marker_id}"
        if assoc_id is not None:
            label += f" -> P{assoc_id}"

        cv2.putText(
            frame,
            label,
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # --- Frame center crosshair ---
    cv2.circle(frame, frame_center, 6, (255, 255, 255), -1)
    cv2.line(
        frame,
        (frame_center[0] - 20, frame_center[1]),
        (frame_center[0] + 20, frame_center[1]),
        (255, 255, 255),
        1,
    )
    cv2.line(
        frame,
        (frame_center[0], frame_center[1] - 20),
        (frame_center[0], frame_center[1] + 20),
        (255, 255, 255),
        1,
    )

    # --- Tracking HUD ---
    if tracked_person is not None:
        target_center = tracked_person["center"]
        offset_x = target_center[0] - frame_center[0]
        offset_y = target_center[1] - frame_center[1]
        command = get_movement_command(offset_x)

        cv2.line(frame, frame_center, target_center, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Selected ID: {selected_track_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Offset X: {offset_x}, Offset Y: {offset_y}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Command: {command}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Click a detected person to select target",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # --- Gesture label (shown once gesture module is implemented) ---
    if gesture:
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

    # --- FPS (top-right) ---
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame,
            fps_text,
            (frame_w - tw - 15, th + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # --- Re-ID / enrollment status banner ---
    if enroll_status is not None:
        is_enrolling = enroll_status.startswith("ENROLLING")
        banner_color = (0, 255, 255) if is_enrolling else (0, 200, 80)
        cv2.putText(
            frame,
            enroll_status,
            (20, frame_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            banner_color,
            2,
        )

    # --- Bottom bar ---
    cv2.putText(
        frame,
        "Q: quit | C: clear | Click=select | E=enroll 360 | S=single frame",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return frame
