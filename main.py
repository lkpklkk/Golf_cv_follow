import cv2
import time
import config
from camera_selector import pick_camera
from tracker.person_tracker import PersonTracker
from tracker.aruco_tracker import ArucoTracker
from gesture.gesture_recognizer import GestureRecognizer
from control.steering import get_movement_command
from control.cart_controller import CartController
from ui.overlay import draw
from utils.geometry import point_inside_box
from reid.embedder import PersonEmbedder
from reid.enroller import Enroller
from reid.matcher import ReIDMatcher
from reid.orientation import OrientationEstimator

# -----------------------------
# State
# -----------------------------
CLICK_POINT = None
selected_track_id = None
selected_aruco_id = None  # ArUco ID of the selected person (for re-id)
enroll_track_id = None  # track_id of the person queued for enrollment


def mouse_callback(event, x, y, flags, param):
    global CLICK_POINT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINT = (x, y)


# -----------------------------
# Init
# -----------------------------
person_tracker = PersonTracker()
aruco_tracker = ArucoTracker()
gesture_recognizer = GestureRecognizer()
cart = CartController()

embedder = PersonEmbedder()
enroller = Enroller(embedder)
matcher = ReIDMatcher()
orientation_estimator = OrientationEstimator()

camera_index = pick_camera(default=config.CAMERA_INDEX)
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

cv2.namedWindow(config.WINDOW_NAME)
cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

print("Starting tracker...")
print("Controls:")
print("  Click person: select target for enrollment")
print("  E: start 360° enrollment (have player slowly turn around)")
print("  S: single-frame enrollment (quick snapshot)")
print("  C: clear selection + reset re-id")
print("  Q: quit")

_prev_time = time.time()
_fps = 0.0

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    # --- FPS ---
    now = time.time()
    _fps = 1.0 / (now - _prev_time) if (now - _prev_time) > 0 else 0.0
    _prev_time = now

    # --- Detections ---
    people = person_tracker.detect(frame)
    markers = aruco_tracker.detect(frame)

    # Associate markers to people and update the re-id mapping
    markers = aruco_tracker.associate(markers, people)

    # Gesture recognition (returns None until model is trained)
    gesture = gesture_recognizer.recognize(frame)

    # --- Mouse selection ---
    if CLICK_POINT is not None:
        for person in people:
            if point_inside_box(CLICK_POINT, person["box"]):
                selected_track_id = person["track_id"]
                enroll_track_id = person["track_id"]
                # Remember ArUco ID if one is associated with this person
                for marker in markers:
                    if marker.get("track_id") == selected_track_id:
                        selected_aruco_id = marker["id"]
                        print(f"Selected person has ArUco ID: {selected_aruco_id}")
                        break
                print(
                    f"Selected person track ID: {selected_track_id} — "
                    "press E (360°) or S (single) to enroll"
                )
                break
        CLICK_POINT = None

    # --- ArUco re-identification ---
    # If the selected person disappeared from YOLO's list, try to recover
    # them via their ArUco marker.
    track_ids_in_frame = {p["track_id"] for p in people}
    if (
        selected_track_id is not None
        and selected_track_id not in track_ids_in_frame
        and selected_aruco_id is not None
    ):
        recovered = aruco_tracker.resolve_track_id(selected_aruco_id, people)
        if recovered is not None:
            print(
                f"Re-identified target via ArUco {selected_aruco_id}: new track_id={recovered}"
            )
            selected_track_id = recovered

    # --- Find tracked person ---
    tracked_person = next(
        (p for p in people if p["track_id"] == selected_track_id), None
    )

    # --- Re-ID: enrollment + matching ---
    reid_matches = None
    enroll_status = None

    if enroller.state == Enroller.ENROLLING:
        # Feed crops of the enrollment target into the enroller
        enroll_target = next(
            (p for p in people if p["track_id"] == enroll_track_id), None
        )
        if enroll_target is not None:
            done = enroller.update(
                frame,
                enroll_target["box"],
                keypoints=enroll_target.get("keypoints"),
            )
            if done:
                matcher.set_enrolled(enroller.enrolled_embeddings)
        collected, total = enroller.progress
        unit = (
            "views"
            if (orientation_estimator.available and enroller._mode == "360")
            else "frames"
        )
        enroll_status = f"ENROLLING {collected}/{total} {unit} — keep player in view"
        if enroller._mode == "360":
            enroll_status += f" | {enroller.angle_status}"

    elif enroller.is_enrolled:
        enroll_status = "RE-ID ACTIVE"
        # Embed every detected person in one batched pass and score them.
        embeddings_by_id = {}
        boxes = [person["box"] for person in people]
        embeddings = embedder.embed_many(frame, boxes)
        for person, emb in zip(people, embeddings):
            if emb is not None:
                embeddings_by_id[person["track_id"]] = emb

        reid_matches = matcher.match_all(embeddings_by_id)

        stable_id = matcher.select_stable_target(reid_matches, selected_track_id)
        if stable_id is not None:
            selected_track_id = stable_id
            tracked_person = next(
                (p for p in people if p["track_id"] == stable_id), None
            )

    # --- Steering ---
    if tracked_person is not None:
        offset_x = tracked_person["center"][0] - (frame.shape[1] // 2)
        command = get_movement_command(offset_x)
        cart.send(command)

    # --- Render ---
    frame = draw(
        frame,
        people,
        markers,
        tracked_person,
        selected_track_id,
        gesture,
        fps=_fps,
        reid_matches=reid_matches,
        enrolling_id=enroll_track_id if enroller.state == Enroller.ENROLLING else None,
        enroll_status=enroll_status,
    )
    cv2.imshow(config.WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("e") and enroll_track_id is not None:
        enroller.reset()
        enroller.start_360(orientation_estimator)
    if key == ord("s") and enroll_track_id is not None:
        enroller.reset()
        enroller.start_single()
    if key == ord("c"):
        selected_track_id = None
        selected_aruco_id = None
        enroll_track_id = None
        enroller.reset()
        matcher.set_enrolled(None)
        print("Selection and re-id cleared.")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
gesture_recognizer.close()
orientation_estimator.close()
cart.close()
