import cv2
import numpy as np
import config
from utils.geometry import distance


class ArucoTracker:
    """
    Detects ArUco markers and uses them to re-identify tracked persons.

    Problem being solved:
        On a busy golf course YOLO can lose track of a person (occlusion,
        crowding, camera shake) and reassign a new track_id.  If the target
        player carries a unique ArUco marker we can recover the correct
        track_id even after YOLO forgets them.

    How it works:
        1. Each frame, detect all ArUco markers.
        2. Associate each marker with the nearest person bounding box
           (within ARUCO_ASSOCIATION_MAX_DIST pixels).
        3. Maintain a persistent mapping:  aruco_id  →  track_id
           This is updated every time a marker and a person are seen together.
        4. resolve_track_id(aruco_id, people) looks up the mapping and finds
           the current person that matches — even if their track_id changed.
    """

    def __init__(self):
        self.available = hasattr(cv2, "aruco")

        if self.available:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        else:
            print("ArUco not available. Install opencv-contrib-python.")

        # aruco_id (int) → last known track_id (int)
        self._aruco_to_track: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame):
        """
        Detect ArUco markers in a frame.
        Returns a list of dicts: {id, corners, center}
        Does NOT update the re-id mapping — call associate() for that.
        """
        if not self.available:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

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

    def associate(self, markers, people):
        """
        Match each detected marker to the nearest person box and update
        the aruco_id → track_id mapping.

        Call this every frame after both detect() and PersonTracker.detect().

        Returns the same markers list, each entry augmented with a
        "track_id" key (int) or None if no person was close enough.
        """
        for marker in markers:
            nearest_track_id = self._nearest_person(marker["center"], people)
            marker["track_id"] = nearest_track_id

            if nearest_track_id is not None:
                self._aruco_to_track[marker["id"]] = nearest_track_id

        return markers

    def resolve_track_id(self, aruco_id, people):
        """
        Given an aruco_id, return the track_id of the person currently
        carrying that marker, or None if they are not in frame.

        Use this to recover a lost target:
            If selected_track_id is gone from `people`, call this with
            the target's known aruco_id to find their new track_id.
        """
        last_known = self._aruco_to_track.get(aruco_id)
        if last_known is None:
            return None

        # Check if that track_id still exists in the current frame
        for person in people:
            if person["track_id"] == last_known:
                return last_known

        # Track was lost — scan for any person now associated with this marker
        for person in people:
            for tid, aid in self._aruco_to_track.items():
                if aid == aruco_id and person["track_id"] == tid:
                    return tid

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_person(self, marker_center, people):
        """Return the track_id of the person whose center is closest to
        marker_center, within ARUCO_ASSOCIATION_MAX_DIST. Returns None
        if no person qualifies."""
        best_id = None
        best_dist = config.ARUCO_ASSOCIATION_MAX_DIST

        for person in people:
            d = distance(marker_center, person["center"])
            if d < best_dist:
                best_dist = d
                best_id = person["track_id"]

        return best_id
