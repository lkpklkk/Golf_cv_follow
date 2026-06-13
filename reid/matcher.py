import numpy as np
import config


class ReIDMatcher:
    """
    Full Re-ID subsystem: gallery management, per-frame identity verification,
    and target recovery / switching.

    The gallery is populated via set_enrolled() (or extended with add_embedding()).
    After enrollment, call update() once per frame to maintain target tracking.

    Gallery uses MAX-score across all entries so that a frontal query matches the
    frontal sector even when other sectors score low (important for 360° enrollment).
    Score range: 0.0 (completely different) → 1.0 (identical).
    """

    def __init__(self, embedder):
        self._embedder = embedder
        self._gallery: list[np.ndarray] = []
        self._verify_misses: int = 0
        self._call_index: int = 0

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------

    def set_enrolled(self, embeddings: list[np.ndarray] | np.ndarray | None) -> None:
        """Set (or clear) the gallery. Accepts a list (360° mode) or a single embedding."""
        if embeddings is None:
            self._gallery = []
        elif isinstance(embeddings, list):
            self._gallery = [e for e in embeddings if e is not None]
        else:
            self._gallery = [embeddings]
        self.reset_tracking_state()

    def add_embedding(self, embedding: np.ndarray) -> bool:
        if embedding is None:
            return False
        self._gallery.append(embedding)
        self.reset_tracking_state()
        return True

    @property
    def is_ready(self) -> bool:
        return len(self._gallery) > 0

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        frame,
        people: list[dict],
        target_track_id: int | None,
        tracked_person: dict | None,
    ) -> tuple[int | None, dict | None, dict | None]:
        """
        Run one Re-ID frame.

        Embeds all visible people (throttled to every REID_VERIFY_SELECTED_EVERY_FRAMES
        frames when the target is visible), then verifies the target's identity,
        checks for a better-matching candidate, and attempts recovery when the
        target is out of frame.

        Returns:
            (target_track_id, tracked_person, scores)
            scores is None when the embedding pass was skipped (throttled) —
            the caller should retain its last known scores for the debug overlay.
        """
        self._call_index += 1

        if not people:
            return target_track_id, None, None

        # Skip embedding on non-verify frames when the target is already tracked.
        on_verify_frame = (
            self._call_index % config.REID_VERIFY_SELECTED_EVERY_FRAMES == 0
        )
        if tracked_person is not None and not on_verify_frame:
            return target_track_id, tracked_person, None

        # Embed every visible person and score against the gallery.
        boxes = [p["box"] for p in people]
        embeddings = self._embedder.embed_many(frame, boxes)
        scores: dict[int, float] = {}
        for person, emb in zip(people, embeddings):
            if emb is not None:
                _, score = self.match(emb)
                scores[person["track_id"]] = score

        if not scores:
            return target_track_id, tracked_person, None

        if tracked_person is not None and target_track_id is not None:
            return self._verify_and_switch(
                scores, people, target_track_id, tracked_person
            )

        return self._recover(scores, people)

    def reset_tracking_state(self) -> None:
        """Reset per-target counters. Call when a new target is selected or cleared."""
        self._verify_misses = 0
        self._call_index = 0

    # ------------------------------------------------------------------
    # Low-level matching (used by enrollment helpers and update())
    # ------------------------------------------------------------------

    def match(self, embedding: np.ndarray) -> tuple[bool, float]:
        """Compare a single embedding against the gallery. Returns (is_match, score)."""
        if not self._gallery or embedding is None:
            return False, 0.0
        best = max(float(np.dot(ref, embedding)) for ref in self._gallery)
        best = max(0.0, min(1.0, best))
        return best >= config.REID_SIMILARITY_THRESHOLD, best

    @staticmethod
    def select_best_match(scores: dict) -> int | None:
        """
        Return the highest-scoring key if it clears the similarity threshold
        and margin over the runner-up, else None.

        Works with any key type (track_id, detection index, etc.).
        """
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < config.REID_SIMILARITY_THRESHOLD:
            return None
        if best_score - second_score < config.REID_MIN_SCORE_MARGIN:
            return None
        return best_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_and_switch(
        self,
        scores: dict,
        people: list[dict],
        target_track_id: int,
        tracked_person: dict,
    ) -> tuple[int | None, dict | None, dict]:
        """Verify the tracked target and switch immediately if a better match appears."""
        target_score = scores.get(target_track_id, 0.0)

        if target_score < config.REID_VERIFY_SELECTED_MIN_SCORE:
            self._verify_misses += 1
            if self._verify_misses >= config.REID_VERIFY_SELECTED_MAX_MISSES:
                self._verify_misses = 0
                return None, None, scores
        else:
            self._verify_misses = 0

        better = {
            tid: s
            for tid, s in scores.items()
            if tid != target_track_id
            and s >= target_score + config.REID_SWITCH_BETTER_MARGIN
        }
        if better:
            best_id = max(better, key=better.__getitem__)
            self._verify_misses = 0
            new_tracked = next((p for p in people if p["track_id"] == best_id), None)
            return best_id, new_tracked, scores

        return target_track_id, tracked_person, scores

    def _recover(
        self,
        scores: dict,
        people: list[dict],
    ) -> tuple[int | None, dict | None, dict]:
        """Try to recover a lost target from the visible candidates."""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < config.REID_SIMILARITY_THRESHOLD:
            return None, None, scores
        if best_score - second_score < config.REID_MIN_SCORE_MARGIN:
            return None, None, scores

        self._verify_misses = 0
        new_tracked = next((p for p in people if p["track_id"] == best_id), None)
        return best_id, new_tracked, scores
