from ultralytics import YOLO
import numpy as np
import config
from utils.device import select_torch_device, use_half_precision
from utils.geometry import box_center


class PersonTracker:
    def __init__(self):
        self.device = select_torch_device(config.INFERENCE_DEVICE)
        self.half = use_half_precision(self.device, config.USE_HALF_ON_CUDA)
        self.model = YOLO(config.MODEL_NAME)
        print(f"[YOLO] Using device: {self.device}")

    def detect(self, frame):
        """
        Run YOLO pose tracking on a frame.
        Returns a list of dicts:
            {track_id, box (x1,y1,x2,y2), center (cx,cy), confidence,
             keypoints}  — keypoints is a (17, 3) numpy array [x, y, conf]
             or None if the model has no pose head.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[0],  # COCO class 0 = person
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
            device=self.device,
            half=self.half,
        )

        people = []

        if results and results[0].boxes is not None:
            kpts_data = results[0].keypoints  # None for non-pose models

            for i, box in enumerate(results[0].boxes):
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                person_box = (x1, y1, x2, y2)

                # Extract (17, 3) keypoints [x_px, y_px, conf] if available
                keypoints = None
                if kpts_data is not None and i < len(kpts_data.xy):
                    xy = kpts_data.xy[i].cpu().numpy()  # (17, 2)
                    if kpts_data.conf is not None:
                        conf = kpts_data.conf[i].cpu().numpy()  # (17,)
                    else:
                        conf = np.ones(17, dtype=np.float32)
                    keypoints = np.concatenate([xy, conf[:, None]], axis=1).astype(
                        np.float32
                    )  # (17, 3)

                people.append(
                    {
                        "track_id": track_id,
                        "box": person_box,
                        "center": box_center(person_box),
                        "confidence": confidence,
                        "keypoints": keypoints,
                    }
                )

        return people
