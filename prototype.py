import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

CAMERA_INDEX = 0

MODEL_NAME = "yolo26s.pt"
TRACKER_CONFIG = "bytetrack.yaml"  # Try "botsort.yaml" later if needed

CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CENTER_DEAD_ZONE = 80

CLICK_POINT = None
selected_track_id = None

# -----------------------------
# Load YOLO model
# -----------------------------

model = YOLO(MODEL_NAME)

# -----------------------------
# Open webcam
# -----------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing CAMERA_INDEX to 1.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# -----------------------------
# ArUco setup
# -----------------------------

aruco_available = hasattr(cv2, "aruco")

if aruco_available:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
else:
    print("ArUco is not available. Make sure opencv-contrib-python is installed.")


# -----------------------------
# Helper functions
# -----------------------------


def mouse_callback(event, x, y, flags, param):
    global CLICK_POINT

    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINT = (x, y)


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def detect_aruco_markers(frame):
    if not aruco_available:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    markers = []

    if ids is not None:
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            pts = marker_corners[0].astype(int)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            markers.append(
                {
                    "id": int(marker_id),
                    "corners": pts,
                    "center": (cx, cy),
                }
            )

    return markers


def get_movement_command(offset_x):
    if abs(offset_x) < CENTER_DEAD_ZONE:
        return "CENTERED"
    elif offset_x > CENTER_DEAD_ZONE:
        return "TURN RIGHT"
    else:
        return "TURN LEFT"


# -----------------------------
# Main window
# -----------------------------

WINDOW_NAME = "YOLO26 Human + ArUco Tracker"

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("Starting tracker...")
print("Controls:")
print("  Click person: select target")
print("  C: clear selected target")
print("  Q: quit")

# -----------------------------
# Main loop
# -----------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    # Mirror front camera view
    frame = cv2.flip(frame, 1)

    # -----------------------------
    # YOLO human detection + tracking
    # -----------------------------

    results = model.track(
        source=frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[0],  # COCO class 0 = person
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
        device="cpu",
    )

    people = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])

            person_box = (x1, y1, x2, y2)

            people.append(
                {
                    "track_id": track_id,
                    "box": person_box,
                    "center": box_center(person_box),
                    "confidence": confidence,
                }
            )

    # -----------------------------
    # Mouse selection
    # -----------------------------

    if CLICK_POINT is not None:
        for person in people:
            if point_inside_box(CLICK_POINT, person["box"]):
                selected_track_id = person["track_id"]
                print(f"Selected person track ID: {selected_track_id}")
                break

        CLICK_POINT = None

    # -----------------------------
    # Find selected person
    # -----------------------------

    tracked_person = None

    for person in people:
        if person["track_id"] == selected_track_id:
            tracked_person = person
            break

    # -----------------------------
    # Draw human detections
    # -----------------------------

    for person in people:
        x1, y1, x2, y2 = person["box"]
        cx, cy = person["center"]
        track_id = person["track_id"]
        confidence = person["confidence"]

        is_selected = track_id == selected_track_id

        if is_selected:
            color = (0, 255, 0)
            label = f"TRACKING ID {track_id} {confidence:.2f}"
        else:
            color = (255, 0, 0)
            label = f"Person ID {track_id} {confidence:.2f}"

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

    # -----------------------------
    # ArUco marker detection
    # -----------------------------

    markers = detect_aruco_markers(frame)

    for marker in markers:
        pts = marker["corners"]
        marker_id = marker["id"]
        cx, cy = marker["center"]

        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        cv2.putText(
            frame,
            f"ArUco ID: {marker_id}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # -----------------------------
    # Frame center + movement command
    # -----------------------------

    frame_h, frame_w = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)

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

    # -----------------------------
    # UI instructions
    # -----------------------------

    cv2.putText(
        frame,
        "Q: quit | C: clear target | Click person to track",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        selected_track_id = None
        print("Selection cleared.")

# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()
