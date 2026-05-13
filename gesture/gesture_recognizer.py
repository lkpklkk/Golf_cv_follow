class GestureRecognizer:
    """
    Recognizes control gestures from a person bounding-box crop.

    NOT YET IMPLEMENTED.

    Plan:
        - Train a custom classifier on the 32 MediaPipe pose landmarks.
        - Each gesture class maps to a cart command (e.g. FOLLOW, STOP,
          TURN LEFT, TURN RIGHT, SPEED UP, SLOW DOWN).
        - recognize() will accept a cropped person frame (or full frame +
          box) and return a gesture label string or None.

    To implement:
        1. Collect pose landmark sequences for each gesture.
        2. Train a lightweight classifier (e.g. sklearn MLP or small ONNX
           model) on the 32-pose set.
        3. Load the model here and run inference in recognize().
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
