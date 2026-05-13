# COCO 17-keypoint indices used for orientation
_NOSE = 0
_LEFT_EAR = 3
_RIGHT_EAR = 4

_CONF_THRESH = 0.3


class OrientationEstimator:
    """
    Estimates coarse body view directly from YOLO pose keypoints.
    No separate model is needed — keypoints come free from the pose head.

    Method
    ------
    - front: nose plus both ears visible
    - left/right: only that ear is visible
    - back: nose not visible
    """

    available = True  # always — uses YOLO keypoints, not a separate model

    def estimate_view(self, keypoints, box=None):
        """
        Args:
            keypoints: numpy array (17, 3) [x_px, y_px, conf], or None
            box:       unused; accepted for API symmetry

        Returns:
            "front", "left", "right", "back", or None if keypoints are absent.
        """
        if keypoints is None:
            return None

        nose_c = keypoints[_NOSE, 2]
        l_ear_c = keypoints[_LEFT_EAR, 2]
        r_ear_c = keypoints[_RIGHT_EAR, 2]

        nose_visible = nose_c > _CONF_THRESH
        left_ear_visible = l_ear_c > _CONF_THRESH
        right_ear_visible = r_ear_c > _CONF_THRESH

        if nose_visible and left_ear_visible and right_ear_visible:
            return "front"

        if left_ear_visible and not right_ear_visible:
            return "left"

        if right_ear_visible and not left_ear_visible:
            return "right"

        if not nose_visible:
            return "back"

        return None

    def estimate(self, keypoints, box):
        """
        Backward-compatible wrapper. New enrollment logic uses estimate_view().
        """
        return self.estimate_view(keypoints, box)

    def close(self):
        pass  # nothing to release
