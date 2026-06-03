from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from action.preprocessing import FEATURE_NAMES, preprocess_pose_sequence
from utils.device import select_torch_device


@dataclass(frozen=True)
class ActionPrediction:
    label: str
    label_id: int
    confidence: float
    raw_label: str
    raw_label_id: int
    is_confident: bool
    probabilities: dict[str, float]


class TemporalPoseCNN(nn.Module):
    def __init__(
        self,
        sequence_length,
        keypoint_count,
        feature_count,
        num_classes,
        channels,
        kernel_size,
        pool_size,
        dropout,
        hidden_dim,
    ):
        super().__init__()
        input_channels = keypoint_count * feature_count
        blocks = []
        current_channels = input_channels
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = output_channels
        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            flattened_dim = int(self.features(dummy).numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected model input (B, T, 17, F), got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return self.classifier(self.features(x))


class ActionSequenceClassifier:
    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        model_config,
        label_to_id,
        feature_config=None,
        decision_config=None,
        device="auto",
    ):
        self.model_config = dict(model_config)
        self.label_to_id = {str(k): int(v) for k, v in label_to_id.items()}
        self.id_to_label = {value: key for key, value in self.label_to_id.items()}
        self.feature_config = dict(feature_config or {})
        self.feature_config.setdefault("confidence_threshold", 0.25)
        self.feature_config.setdefault("feature_names", list(FEATURE_NAMES))
        if tuple(self.feature_config["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                "Action feature_names must match the supported preprocessing schema: "
                f"{list(FEATURE_NAMES)}"
            )
        self.decision_config = dict(decision_config or {})
        self.decision_config.setdefault("confidence_threshold", 0.60)
        self.decision_config.setdefault("fallback_label", "other")
        self.decision_config.setdefault("class_thresholds", {})
        self.decision_config["class_thresholds"] = {
            str(label): float(threshold)
            for label, threshold in self.decision_config["class_thresholds"].items()
        }
        self.device = select_torch_device(device)

        self.sequence_length = int(self.model_config["sequence_length"])
        self.keypoint_count = int(self.model_config.get("keypoint_count", 17))
        self.feature_count = len(self.feature_config["feature_names"])
        self.model = TemporalPoseCNN(
            sequence_length=self.sequence_length,
            keypoint_count=self.keypoint_count,
            feature_count=self.feature_count,
            num_classes=len(self.label_to_id),
            channels=list(self.model_config["channels"]),
            kernel_size=int(self.model_config.get("kernel_size", 3)),
            pool_size=int(self.model_config.get("pool_size", 2)),
            dropout=float(self.model_config.get("dropout", 0.25)),
            hidden_dim=int(self.model_config.get("hidden_dim", 128)),
        ).to(self.device)

    def fit(
        self,
        train_loader,
        validation_loader,
        epochs,
        learning_rate,
        weight_decay,
        patience,
        evaluate_fn,
    ):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_metrics = None
        best_macro_f1 = -1.0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                total_samples += len(labels)

            train_loss = total_loss / max(total_samples, 1)
            validation_metrics = evaluate_fn(self, validation_loader)
            validation_metrics["epoch"] = epoch
            validation_metrics["train_loss"] = train_loss
            history.append(validation_metrics)
            macro_f1 = float(validation_metrics["macro_f1"])
            print(
                f"[action train] epoch={epoch} train_loss={train_loss:.4f} "
                f"val_accuracy={validation_metrics['accuracy']:.4f} "
                f"val_macro_f1={macro_f1:.4f}"
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_metrics = dict(validation_metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(patience):
                    print(f"[action train] early stopping after epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_validation": best_metrics, "history": history}

    def predict_proba(self, keypoints_sequence, frame_width, frame_height):
        sequence = np.asarray(keypoints_sequence, dtype=np.float32)
        expected_shape = (self.sequence_length, self.keypoint_count, 3)
        if sequence.shape != expected_shape:
            raise ValueError(f"Expected keypoints shape {expected_shape}, got {sequence.shape}")

        features = preprocess_pose_sequence(
            sequence,
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_threshold=self.feature_config["confidence_threshold"],
        )
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        return {
            self.id_to_label[index]: float(probabilities[index])
            for index in sorted(self.id_to_label)
        }

    def classify(
        self,
        keypoints_sequence,
        frame_width,
        frame_height,
        confidence_threshold=None,
        fallback_label=None,
        class_thresholds=None,
    ):
        probabilities = self.predict_proba(keypoints_sequence, frame_width, frame_height)
        return self.classify_probabilities(
            probabilities,
            confidence_threshold=confidence_threshold,
            fallback_label=fallback_label,
            class_thresholds=class_thresholds,
        )

    def classify_probabilities(
        self,
        probabilities,
        confidence_threshold=None,
        fallback_label=None,
        class_thresholds=None,
    ):
        confidence_threshold = (
            self.decision_config["confidence_threshold"]
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        fallback_label = fallback_label or self.decision_config["fallback_label"]
        thresholds = dict(self.decision_config.get("class_thresholds", {}))
        if class_thresholds:
            thresholds.update(
                {str(label): float(value) for label, value in class_thresholds.items()}
            )

        raw_label = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[raw_label])
        required_threshold = max(
            float(confidence_threshold),
            float(thresholds.get(raw_label, 0.0)),
        )
        is_confident = confidence >= required_threshold
        label = raw_label if is_confident else fallback_label
        return ActionPrediction(
            label=label,
            label_id=self.label_to_id[label],
            confidence=confidence,
            raw_label=raw_label,
            raw_label_id=self.label_to_id[raw_label],
            is_confident=is_confident,
            probabilities=probabilities,
        )

    def save(self, path, extra=None):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "decision_config": self.decision_config,
            "label_to_id": self.label_to_id,
            "extra": extra or {},
        }
        torch.save(checkpoint, output_path)

    @classmethod
    def load(cls, path, device="auto"):
        checkpoint_path = Path(path)
        target_device = select_torch_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        classifier = cls(
            model_config=checkpoint["model_config"],
            label_to_id=checkpoint["label_to_id"],
            feature_config=checkpoint["feature_config"],
            decision_config=checkpoint.get("decision_config"),
            device=target_device,
        )
        classifier.model.load_state_dict(checkpoint["model_state_dict"])
        classifier.model.eval()
        return classifier

    def checkpoint_summary(self):
        return {
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "decision_config": self.decision_config,
            "label_to_id": self.label_to_id,
        }

    @staticmethod
    def prediction_to_dict(prediction):
        return asdict(prediction)
