# -----------------------------
# Camera
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Camera probing can make OpenCV/AVFoundation print noisy warnings when indexes
# are out of range. Increase this if you expect more than two cameras.
CAMERA_SCAN_MAX_INDEX = 2

# -----------------------------
# YOLO model
# -----------------------------
MODEL_NAME = "yolo26s-pose.pt"  # Try "yolo26s.pt" if pose model is too slow
TRACKER_CONFIG = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
# "auto" prefers CUDA, then Apple MPS, then CPU. You can also set "cuda:0",
# "mps", or "cpu" explicitly if a backend is unstable on your machine.
INFERENCE_DEVICE = "auto"
# Half precision is only enabled on CUDA; MPS/CPU stay float32 for stability.
USE_HALF_ON_CUDA = True

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
# Re-ID
# -----------------------------
# Torchreid model used by PersonEmbedder. OSNet-AIN is a person-ReID model
# designed for better cross-domain generalization than ImageNet classifiers.
REID_MODEL_NAME = "osnet_ain_x1_0"
# Public Torchreid checkpoint trained on MSMT17 for cross-domain person Re-ID.
REID_MODEL_WEIGHTS = "weights/osnet_ain_x1_0_msmt17.pth"
# Cosine similarity threshold — tune up if getting false matches on the course
REID_SIMILARITY_THRESHOLD = 0.55
# Reject a Re-ID winner when it is not clearly ahead of the runner-up.
REID_MIN_SCORE_MARGIN = 0.08
# Require the same new Re-ID winner for this many frames before switching.
REID_SWITCH_CONFIRM_FRAMES = 5
# 360 enrollment captures a view only after the same view is stable for this
# many frames.
REID_VIEW_STABLE_FRAMES = 5
# Box movement limits for "relatively still" during view capture.
REID_VIEW_STILL_MAX_CENTER_SHIFT = 0.08  # fraction of box width/height
REID_VIEW_STILL_MAX_SIZE_CHANGE = 0.12  # fraction of box width/height

# -----------------------------
# UI
# -----------------------------
WINDOW_NAME = "Golf Cart Tracker"
DRAW_POSE_KEYPOINTS = True
POSE_KEYPOINT_CONFIDENCE_THRESHOLD = 0.30
