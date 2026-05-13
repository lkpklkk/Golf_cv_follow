import cv2
from control.steering import get_movement_command


def draw(frame, people, markers, tracked_person, selected_track_id, gesture=None):
    """
    Render all overlays onto the frame in-place and return it.

    Args:
        frame:             BGR numpy array
        people:            list of person dicts from PersonTracker
        markers:           list of marker dicts from ArucoTracker (may include track_id key)
        tracked_person:    the currently selected person dict, or None
        selected_track_id: int or None
        gesture:           gesture label string or None (unused until implemented)
    """
    frame_h, frame_w = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)

    # --- People bounding boxes ---
    for person in people:
        x1, y1, x2, y2 = person["box"]
        cx, cy = person["center"]
        track_id = person["track_id"]
        confidence = person["confidence"]
        is_selected = track_id == selected_track_id

        color = (0, 255, 0) if is_selected else (255, 0, 0)
        label = (
            f"TRACKING ID {track_id} {confidence:.2f}"
            if is_selected
            else f"Person ID {track_id} {confidence:.2f}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
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

    # --- Bottom bar ---
    cv2.putText(
        frame,
        "Q: quit | C: clear target | Click person to track",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return frame
