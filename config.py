# -----------------------------
# Camera
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# -----------------------------
# YOLO model
# -----------------------------
MODEL_NAME = "yolo26s.pt"
TRACKER_CONFIG = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

# -----------------------------
# Steering
# -----------------------------
CENTER_DEAD_ZONE = 80  # pixels — offset within this range = CENTERED

# -----------------------------
# ArUco
# -----------------------------
# Max pixel distance from a marker center to a person box center
# to consider them "associated"
ARUCO_ASSOCIATION_MAX_DIST = 200

# -----------------------------
# UI
# -----------------------------
WINDOW_NAME = "Golf Cart Tracker"
