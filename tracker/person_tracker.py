from ultralytics import YOLO
import config
from utils.geometry import box_center


class PersonTracker:
    def __init__(self):
        self.model = YOLO(config.MODEL_NAME)

    def detect(self, frame):
        """
        Run YOLO tracking on a frame.
        Returns a list of dicts:
            {track_id, box (x1,y1,x2,y2), center (cx,cy), confidence}
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[0],  # COCO class 0 = person
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
            device="cpu",
        )

        people = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                person_box = (x1, y1, x2, y2)

                people.append(
                    {
                        "track_id": track_id,
                        "box": person_box,
                        "center": box_center(person_box),
                        "confidence": confidence,
                    }
                )

        return people
