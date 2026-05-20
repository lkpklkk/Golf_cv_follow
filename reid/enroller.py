import numpy as np
import config

_ANGLE_LABELS = ("front", "right", "back", "left")


class Enroller:
    """
    State machine that collects person crops and builds a gallery embedding.

    States
    ------
    IDLE       — waiting for user to trigger enrollment
    ENROLLING  — actively collecting crops from the target person
    DONE       — enrollment finished; enrolled_embedding is ready

    Modes
    -----
    single  — capture one frame immediately.
    360     — orientation-aware: collect one embedding for each cardinal view:
              front, right, back, and left.

    Usage
    -----
        enroller.start_single()         # grab one frame
        enroller.start_360(orientation) # sector-based automatic capture
        done = enroller.update(frame, box)
        embedding = enroller.enrolled_embedding  # ready once DONE
    """

    IDLE = "idle"
    ENROLLING = "enrolling"
    DONE = "done"

    def __init__(self, embedder):
        self._embedder = embedder
        self._orientation = None  # set by start_360()
        self._mode = "single"
        self.state = self.IDLE
        self._embeddings: list[np.ndarray] = []
        self._angle_embeddings: dict[str, np.ndarray] = {}
        self._angle_crops: dict[str, np.ndarray] = {}
        self._candidate_view = None
        self._candidate_frames = 0
        self._last_box = None
        self.enrolled_embeddings: list[np.ndarray] = []
        self.enrolled_gallery_items: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_single(self):
        """Enroll from a single frame."""
        self._reset_buffers()
        self._mode = "single"
        self.state = self.ENROLLING
        print("[ReID] Single-frame enrollment started — keep person in frame.")

    def start_360(self, orientation_estimator):
        """
        Sector-based 360° enrollment.

        Captures four labeled views: front, right, back, and left. Enrollment
        completes once every view has been observed.

        Orientation is estimated from YOLO keypoints (no extra model).

        Args:
            orientation_estimator: OrientationEstimator instance from main.
        """
        self._reset_buffers()
        self._mode = "360"
        self._orientation = orientation_estimator
        self.state = self.ENROLLING
        print(
            "[ReID] 360° enrollment started — capture front, right, back, left. "
            "Have the player slowly turn around."
        )

    def update(self, frame, box, keypoints=None):
        """
        Feed a frame + person box (and optional YOLO keypoints) during enrollment.
        Returns True the moment enrollment completes, False otherwise.
        Silently ignores calls when not in ENROLLING state.
        """
        if self.state != self.ENROLLING:
            return False

        if self._mode == "single":
            return self._update_single(frame, box)
        else:
            return self._update_360(frame, box, keypoints)

    def reset(self):
        """Return to IDLE and clear any collected data."""
        self._reset_buffers()
        self.enrolled_embeddings = []
        self.enrolled_gallery_items = []
        self._orientation = None
        self.state = self.IDLE
        print("[ReID] Enrollment reset.")

    @property
    def progress(self):
        """
        Returns (covered, total) for progress display.
        In 360° mode: sectors covered vs total sectors.
        In single mode: frames collected vs 1.
        """
        if self._mode == "360":
            return len(self._angle_embeddings), len(_ANGLE_LABELS)
        return len(self._embeddings), 1

    @property
    def captured_angles(self):
        """Names of captured views in display order."""
        return [name for name in _ANGLE_LABELS if name in self._angle_embeddings]

    @property
    def missing_angles(self):
        """Names of missing views in display order."""
        return [name for name in _ANGLE_LABELS if name not in self._angle_embeddings]

    @property
    def angle_status(self):
        captured = ", ".join(self.captured_angles) or "none"
        missing = ", ".join(self.missing_angles) or "none"
        hold = ""
        if self._candidate_view is not None:
            hold = (
                f" | hold {self._candidate_view} "
                f"{self._candidate_frames}/{config.REID_VIEW_STABLE_FRAMES}"
            )
        return f"captured: {captured} | missing: {missing}{hold}"

    @property
    def is_enrolled(self):
        return self.state == self.DONE and len(self.enrolled_embeddings) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_single(self, frame, box):
        embedding = self._embedder.embed(frame, box)
        if embedding is not None:
            self._embeddings.append(embedding)
            self.enrolled_gallery_items = [("single", self._crop(frame, box))]
            self._finalize()
            return True
        return False

    def _update_360(self, frame, box, keypoints=None):
        view = None
        if self._orientation is not None:
            view = self._orientation.estimate_view(keypoints, box)
        print(
            f"[ReID] Estimated view: {view}"
            if view is not None
            else "[ReID] View unavailable; waiting for a clear front/left/right/back view."
        )
        if view is not None:
            if self._view_is_stable(view, box):
                self._candidate_frames += 1
            else:
                self._candidate_view = view
                self._candidate_frames = 1
            self._last_box = box

            if (
                self._candidate_frames >= config.REID_VIEW_STABLE_FRAMES
                and view not in self._angle_embeddings
            ):
                embedding = self._embedder.embed(frame, box)
                if embedding is not None:
                    self._angle_embeddings[view] = embedding
                    self._angle_crops[view] = self._crop(frame, box)
                    self._embeddings.append(embedding)
                    print(f"[ReID] Captured {view} view.")

            if len(self._angle_embeddings) == len(_ANGLE_LABELS):
                self._finalize()
                return True

        else:
            self._candidate_view = None
            self._candidate_frames = 0
            self._last_box = box

        return False

    def _finalize(self):
        # Keep individual sector embeddings for max-score gallery matching.
        # Each entry is already L2-normalised by the embedder.
        self.enrolled_embeddings: list[np.ndarray] = list(self._embeddings)
        if self._mode == "360":
            self.enrolled_gallery_items = [
                (name, self._angle_crops[name])
                for name in _ANGLE_LABELS
                if name in self._angle_crops
            ]
        self.state = self.DONE
        print(f"[ReID] Enrollment complete — {len(self._embeddings)} gallery entries.")

    def _reset_buffers(self):
        self._embeddings = []
        self._angle_embeddings = {}
        self._angle_crops = {}
        self._candidate_view = None
        self._candidate_frames = 0
        self._last_box = None

    def _view_is_stable(self, view, box):
        if view != self._candidate_view or self._last_box is None:
            return False

        x1, y1, x2, y2 = box
        lx1, ly1, lx2, ly2 = self._last_box

        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        lw = max(float(lx2 - lx1), 1.0)
        lh = max(float(ly2 - ly1), 1.0)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        lcx = (lx1 + lx2) * 0.5
        lcy = (ly1 + ly2) * 0.5

        center_still = (
            abs(cx - lcx) <= config.REID_VIEW_STILL_MAX_CENTER_SHIFT * w
            and abs(cy - lcy) <= config.REID_VIEW_STILL_MAX_CENTER_SHIFT * h
        )
        size_still = (
            abs(w - lw) <= config.REID_VIEW_STILL_MAX_SIZE_CHANGE * w
            and abs(h - lh) <= config.REID_VIEW_STILL_MAX_SIZE_CHANGE * h
        )
        return center_still and size_still

    def _crop(self, frame, box):
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy()
