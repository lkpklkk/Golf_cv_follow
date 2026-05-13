import importlib.util
import os

import cv2
import numpy as np
import config
from utils.device import select_torch_device

try:
    import torch
    import torchvision.transforms as T

    _REID_BACKEND_AVAILABLE = True
except ImportError:
    _REID_BACKEND_AVAILABLE = False
    print("[ReID] torch/torchvision not found. Re-ID embedding will be disabled.")


def _load_osnet_ain_factory(model_name):
    spec = importlib.util.find_spec("torchreid")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("torchreid is not installed")

    model_path = os.path.join(
        spec.submodule_search_locations[0], "reid", "models", "osnet_ain.py"
    )
    module_spec = importlib.util.spec_from_file_location("_torchreid_osnet_ain", model_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load Torchreid OSNet-AIN module from {model_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, model_name)


def _load_checkpoint_weights(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model", state))

    model_state = model.state_dict()
    matched = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key in model_state and model_state[key].size() == value.size():
            matched[key] = value

    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"[ReID] Loaded {len(matched)} checkpoint tensors from {checkpoint_path}")


class PersonEmbedder:
    """
    Extracts person Re-ID embeddings from person crops using OSNet-AIN.

    Output: L2-normalized numpy vector, or None on failure.

    OSNet-AIN is used because it is trained for person re-identification and
    generalizes better across camera/lighting domains than ImageNet features.
    """

    # Standard ImageNet normalization
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    # Re-ID convention: tall narrow crops
    _INPUT_SIZE = (256, 128)  # (H, W)

    def __init__(self):
        self.available = _REID_BACKEND_AVAILABLE

        if not self.available:
            return

        self.device = select_torch_device(config.INFERENCE_DEVICE)
        try:
            model_factory = _load_osnet_ain_factory(config.REID_MODEL_NAME)
            self._model = model_factory(
                num_classes=1,
                loss="softmax",
                pretrained=not bool(config.REID_MODEL_WEIGHTS),
            )

            if config.REID_MODEL_WEIGHTS:
                _load_checkpoint_weights(self._model, config.REID_MODEL_WEIGHTS)
        except Exception as exc:
            self.available = False
            print(f"[ReID] Could not initialize {config.REID_MODEL_NAME}: {exc}")
            return

        self._model.eval().to(self.device)
        print(f"[ReID] Using {config.REID_MODEL_NAME} on device: {self.device}")

        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self._INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=self._MEAN, std=self._STD),
            ]
        )

    def embed(self, frame, box):
        """
        Crop the person out of `frame` using `box` (x1,y1,x2,y2) and
        return an L2-normalized embedding vector (numpy float32 array).
        Returns None if the crop is invalid or OSNet-AIN is unavailable.
        """
        if not self.available:
            return None

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self._transform(crop_rgb).unsqueeze(0).to(self.device)  # (1, C, H, W)

        with torch.no_grad():
            feat = self._model(tensor).squeeze(0).cpu().numpy()  # (1280,)

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat.astype(np.float32)

    def embed_many(self, frame, boxes):
        """
        Embed multiple person boxes in one model pass.

        Args:
            frame: Full BGR frame.
            boxes: Iterable of (x1,y1,x2,y2) boxes.

        Returns:
            List aligned with `boxes`; entries are numpy float32 embeddings or
            None when a crop is invalid.
        """
        boxes = list(boxes)
        if not self.available:
            return [None for _ in boxes]

        h, w = frame.shape[:2]
        tensors = []
        valid_indices = []
        outputs = [None for _ in boxes]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(crop_rgb))
            valid_indices.append(i)

        if not tensors:
            return outputs

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self._model(batch).cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = np.divide(feats, norms, out=np.zeros_like(feats), where=norms > 0)

        for i, feat in zip(valid_indices, feats):
            outputs[i] = feat.astype(np.float32)

        return outputs
