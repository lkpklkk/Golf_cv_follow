from action.runtime import create_live_action_components, update_live_action
from control.cart_controller import CartController
from control.steering import get_movement_command
from pipeline.session import TrackingSession
from reid.embedder import PersonEmbedder
from reid.enroller import Enroller
from reid.matcher import ReIDMatcher
from reid.orientation import OrientationEstimator
from tracker.person_tracker import PersonTracker


class TrackerPipeline:
    """
    Per-frame detect → ReID → action → command pipeline.

    Owns all CV components and tracking state.  Frontends (GUI, headless,
    video test runner) call tick() each frame and receive a TrackingSession;
    they call the helper methods below to drive enrollment and selection.
    """

    def __init__(self):
        self.person_tracker = PersonTracker()
        self.cart = CartController()
        self.action_classifier, self.action_buffer, self.action_config = (
            create_live_action_components()
        )
        self.embedder = PersonEmbedder()
        self.enroller = Enroller(self.embedder)
        self.matcher = ReIDMatcher(self.embedder)
        self.orientation_estimator = OrientationEstimator()

        self._target_track_id: int | None = None
        self._tracked_person: dict | None = None
        self._reid_matches: dict | None = None
        self._action_prediction = None
        self._action_target_id: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def tick(self, frame, timestamp: float) -> TrackingSession:
        """Run one frame through the full pipeline. Returns a TrackingSession."""
        people = self.person_tracker.detect(frame)
        self._tracked_person = next(
            (p for p in people if p["track_id"] == self._target_track_id), None
        )

        enroll_status, gallery_updated = self._step_enrollment(frame, people)

        if self.enroller.state != Enroller.ENROLLING and self.matcher.is_ready:
            new_target, new_tracked, new_scores = self.matcher.update(
                frame, people, self._target_track_id, self._tracked_person
            )
            self._target_track_id = new_target
            self._tracked_person = new_tracked
            if new_scores is not None:
                self._reid_matches = new_scores

        self._step_action(frame, timestamp)
        command = self._step_command(frame)

        return TrackingSession(
            people=people,
            target_track_id=self._target_track_id,
            tracked_person=self._tracked_person,
            reid_matches=self._reid_matches,
            action_prediction=self._action_prediction,
            command=command,
            enroll_status=enroll_status,
            gallery_updated=gallery_updated,
        )

    def select_target(self, track_id: int) -> None:
        self._target_track_id = track_id
        self.matcher.reset_tracking_state()

    def clear(self) -> None:
        self._target_track_id = None
        self._tracked_person = None
        self._reid_matches = None
        self._action_prediction = None
        self._action_target_id = None
        if self.action_buffer is not None:
            self.action_buffer.reset()
        self.enroller.reset()
        self.matcher.set_enrolled(None)

    def start_360_enroll(self) -> None:
        if self._target_track_id is None:
            return
        self.enroller.reset()
        self.enroller.start_360(self.orientation_estimator)

    def start_single_enroll(self) -> None:
        if self._target_track_id is None:
            return
        self.enroller.reset()
        self.enroller.start_single()

    def enroll_manual_crop(self, frame, box) -> bool:
        """Embed a manually selected bounding box and add it to the gallery."""
        emb = self.embedder.embed(frame, box)
        return self.matcher.add_embedding(emb)

    def close(self) -> None:
        self.orientation_estimator.close()
        self.cart.close()

    # ------------------------------------------------------------------
    # Internal steps — called once per tick() in order
    # ------------------------------------------------------------------

    def _step_enrollment(self, frame, people) -> tuple[str | None, bool]:
        """Drive the enrollment state machine. Returns (status_text, gallery_updated)."""
        if self.enroller.state != Enroller.ENROLLING:
            return None, False

        gallery_updated = False
        target = next(
            (p for p in people if p["track_id"] == self._target_track_id), None
        )
        if target is not None:
            done = self.enroller.update(
                frame,
                target["box"],
                keypoints=target.get("keypoints"),
            )
            if done:
                self.matcher.set_enrolled(self.enroller.enrolled_embeddings)
                gallery_updated = True

        collected, total = self.enroller.progress
        unit = "views" if self.enroller._mode == "360" else "frames"
        status = f"ENROLLING {collected}/{total} {unit}"
        if self.enroller._mode == "360":
            status += f" | {self.enroller.angle_status}"
        return status, gallery_updated

    def _step_action(self, frame, timestamp: float) -> None:
        if self._target_track_id != self._action_target_id:
            self._action_prediction = None
            self._action_target_id = self._target_track_id

        prediction = update_live_action(
            classifier=self.action_classifier,
            buffer=self.action_buffer,
            config=self.action_config,
            target_id=self._target_track_id,
            tracked_person=self._tracked_person,
            timestamp=timestamp,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        if prediction is not None:
            self._action_prediction = prediction

    def _step_command(self, frame) -> str | None:
        if self._tracked_person is None:
            return None
        offset_x = self._tracked_person["center"][0] - (frame.shape[1] // 2)
        command = get_movement_command(offset_x)
        self.cart.send(command)
        return command
