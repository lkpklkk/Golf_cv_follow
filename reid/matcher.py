import numpy as np
import config


class ReIDMatcher:
    """
    Compares query embeddings against a gallery of enrolled embeddings
    using cosine similarity (dot product of L2-normalised vectors).

    In 360° mode the gallery contains one embedding per orientation sector.
    Matching uses the MAX score across all gallery entries so that a frontal
    query matches the frontal sector embedding at high confidence even when
    other sectors (e.g. back) would score low.

    Score range: 0.0 (completely different) → 1.0 (identical).
    A score >= REID_SIMILARITY_THRESHOLD is considered a match.
    """

    def __init__(self):
        self._gallery: list[np.ndarray] = []
        self._pending_track_id = None
        self._pending_frames = 0

    def set_enrolled(self, embeddings: list[np.ndarray] | np.ndarray | None):
        """
        Set (or clear) the gallery from the enroller.
        Accepts a list of embeddings (360° mode) or a single embedding
        (single-frame mode) for backward compatibility.
        """
        if embeddings is None:
            self._gallery = []
        elif isinstance(embeddings, list):
            self._gallery = [e for e in embeddings if e is not None]
        else:
            self._gallery = [embeddings]
        self.reset_temporal_state()

    @property
    def is_ready(self):
        return len(self._gallery) > 0

    @property
    def gallery_size(self):
        return len(self._gallery)

    def add_embedding(self, embedding: np.ndarray):
        if embedding is None:
            return False
        self._gallery.append(embedding)
        self.reset_temporal_state()
        return True

    def match(self, embedding: np.ndarray) -> tuple[bool, float]:
        """
        Compare a query embedding against every gallery entry.
        Returns (is_match, best_score) using max-score across all entries.
        """
        if not self._gallery or embedding is None:
            return False, 0.0

        best = max(float(np.dot(ref, embedding)) for ref in self._gallery)
        best = max(0.0, min(1.0, best))  # clamp numerical noise
        return best >= config.REID_SIMILARITY_THRESHOLD, best

    def match_all(self, embeddings_by_id: dict) -> dict:
        """
        Match multiple people at once.

        Args:
            embeddings_by_id: {track_id: embedding_vector}

        Returns:
            {track_id: similarity_score}  — score for every supplied track_id
        """
        results = {}
        for track_id, emb in embeddings_by_id.items():
            _, score = self.match(emb)
            results[track_id] = score
        return results

    def reset_temporal_state(self):
        self._pending_track_id = None
        self._pending_frames = 0

    def select_stable_target(self, scores_by_id: dict, current_track_id=None):
        """
        Pick a target using thresholding, score-margin rejection, and temporal
        smoothing.

        A new target is accepted only when:
          - it is above REID_SIMILARITY_THRESHOLD,
          - it beats the runner-up by REID_MIN_SCORE_MARGIN,
          - and the same candidate wins for REID_SWITCH_CONFIRM_FRAMES frames.

        The current target is kept immediately when it is still the winner.
        """
        if not scores_by_id:
            self.reset_temporal_state()
            return current_track_id

        ranked = sorted(scores_by_id.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < config.REID_SIMILARITY_THRESHOLD:
            self.reset_temporal_state()
            return current_track_id

        if (best_score - second_score) < config.REID_MIN_SCORE_MARGIN:
            self.reset_temporal_state()
            return current_track_id

        if best_id == current_track_id:
            self.reset_temporal_state()
            return best_id

        if best_id == self._pending_track_id:
            self._pending_frames += 1
        else:
            self._pending_track_id = best_id
            self._pending_frames = 1

        if self._pending_frames >= config.REID_SWITCH_CONFIRM_FRAMES:
            self.reset_temporal_state()
            return best_id

        return current_track_id
