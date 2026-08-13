from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from action.classifier import ActionSequenceClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export action classifier checkpoint to ONNX and optional CoreML."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to action_sequence_classifier checkpoint (.pt)",
    )
    parser.add_argument(
        "--onnx-out",
        default="weights/action_sequence_classifier.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--coreml-out",
        default=None,
        help="Optional output CoreML path (.mlpackage or .mlmodel)",
    )
    parser.add_argument(
        "--metadata-out",
        default="weights/action_sequence_classifier.metadata.json",
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Export batch size for sample input",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_onnx(
    classifier: ActionSequenceClassifier,
    output_path: Path,
    opset: int,
    batch_size: int,
) -> None:
    ensure_parent(output_path)
    sequence_length = classifier.sequence_length
    keypoint_count = classifier.keypoint_count
    feature_count = classifier.feature_count

    dummy_input = torch.randn(
        batch_size,
        sequence_length,
        keypoint_count,
        feature_count,
        dtype=torch.float32,
    )

    classifier.model.eval()
    with torch.no_grad():
        torch.onnx.export(
            classifier.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pose_features"],
            output_names=["logits"],
            dynamic_axes={
                "pose_features": {0: "batch"},
                "logits": {0: "batch"},
            },
        )


def export_coreml(
    classifier: ActionSequenceClassifier,
    output_path: Path,
    batch_size: int,
) -> None:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is required for --coreml-out. Install with `pip install coremltools`."
        ) from exc

    ensure_parent(output_path)
    sequence_length = classifier.sequence_length
    keypoint_count = classifier.keypoint_count
    feature_count = classifier.feature_count

    dummy_input = torch.randn(
        batch_size,
        sequence_length,
        keypoint_count,
        feature_count,
        dtype=torch.float32,
    )

    classifier.model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(classifier.model, dummy_input)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="pose_features",
                shape=dummy_input.shape,
                dtype=dummy_input.numpy().dtype,
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(str(output_path))


def write_metadata(classifier: ActionSequenceClassifier, output_path: Path) -> None:
    ensure_parent(output_path)
    metadata = {
        "model_config": classifier.model_config,
        "feature_config": classifier.feature_config,
        "decision_config": classifier.decision_config,
        "label_to_id": classifier.label_to_id,
        "id_to_label": classifier.id_to_label,
        "input_spec": {
            "name": "pose_features",
            "shape": [
                "batch",
                classifier.sequence_length,
                classifier.keypoint_count,
                classifier.feature_count,
            ],
            "dtype": "float32",
        },
        "output_spec": {
            "name": "logits",
            "shape": ["batch", len(classifier.label_to_id)],
            "dtype": "float32",
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    classifier = ActionSequenceClassifier.load(checkpoint_path, device="cpu")

    onnx_path = Path(args.onnx_out)
    export_onnx(classifier, onnx_path, args.opset, args.batch_size)
    print(f"Exported ONNX: {onnx_path}")

    if args.coreml_out:
        coreml_path = Path(args.coreml_out)
        export_coreml(classifier, coreml_path, args.batch_size)
        print(f"Exported CoreML: {coreml_path}")

    metadata_path = Path(args.metadata_out)
    write_metadata(classifier, metadata_path)
    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
