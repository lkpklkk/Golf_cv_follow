import cv2
import time
import config
from camera_selector import pick_camera
from pipeline.pipeline import TrackerPipeline
from ui.overlay import draw
from utils.geometry import point_inside_box

# -----------------------------
# State
# -----------------------------
CLICK_POINT = None


def mouse_callback(event, x, y, flags, param):
    global CLICK_POINT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINT = (x, y)


# -----------------------------
# Init
# -----------------------------
pipeline = TrackerPipeline()

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

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    now = time.time()
    fps = 1.0 / (now - _prev_time) if (now - _prev_time) > 0 else 0.0
    _prev_time = now

    session = pipeline.tick(frame, now)

    # --- Mouse selection ---
    if CLICK_POINT is not None:
        for person in session.people:
            if point_inside_box(CLICK_POINT, person["box"]):
                pipeline.select_target(person["track_id"])
                print(
                    f"Selected person track ID: {person['track_id']} — "
                    "press E (360°) or S (single) to enroll"
                )
                break
        CLICK_POINT = None

    # --- Render ---
    enrolling_id = (
        pipeline.target_track_id if session.enroll_status is not None else None
    )
    frame = draw(
        frame,
        session.people,
        session.tracked_person,
        session.target_track_id,
        session.action_prediction,
        fps=fps,
        reid_matches=session.reid_matches,
        enrolling_id=enrolling_id,
        enroll_status=session.enroll_status,
    )
    cv2.imshow(config.WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("e"):
        pipeline.start_360_enroll()
    if key == ord("s"):
        pipeline.start_single_enroll()
    if key == ord("c"):
        pipeline.clear()
        print("Selection and re-id cleared.")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
pipeline.close()
