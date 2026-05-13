class GestureRecognizer:
    """
    Recognizes control gestures from YOLO pose keypoints.

    NOT YET IMPLEMENTED.

    Plan:
        - Use the 17 COCO keypoints already produced by the pose model
          (no extra model needed).
        - Train a custom classifier on keypoint sequences for each gesture.
        - Each gesture class maps to a cart command (e.g. FOLLOW, STOP,
          TURN LEFT, TURN RIGHT, SPEED UP, SLOW DOWN).
        - recognize() will accept the keypoints array from a person dict
          and return a gesture label string or None.

    To implement:
        1. Collect keypoint sequences per gesture (e.g. raised arm, wave).
        2. Train a lightweight classifier (e.g. sklearn MLP or small ONNX)
           on the keypoint vectors.
        3. Load the model in __init__ and run inference in recognize().
    """

    def __init__(self):
        # TODO: load trained model weights here
        pass

    def recognize(self, frame, person_box=None):
        """
        Args:
            frame:       Full BGR frame.
            person_box:  Optional (x1, y1, x2, y2) to crop to one person.
                         If None, uses the full frame.

        Returns:
            A gesture label string (e.g. "FOLLOW", "STOP") or None.
        """
        # TODO: implement once custom model is trained
        return None

    def close(self):
        # TODO: release any model resources
        pass
