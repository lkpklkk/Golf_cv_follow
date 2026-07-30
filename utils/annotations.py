from __future__ import annotations

import json
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def discover_videos(path: Path) -> list[Path]:
    """Return a sorted list of video files at or under path."""
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    return []


def holdout_video_names(holdout_dir: Path | None) -> set[str]:
    """
    Return the filenames of every video in the holdout directory.

    Holdout footage must never reach the training set; silently training on it
    would invalidate every measurement made against it.
    """
    if holdout_dir is None:
        return set()
    return {p.name for p in discover_videos(Path(holdout_dir).expanduser())}


def load_annotations(annotation_file: Path) -> list[dict]:
    """Load a JSON annotation file and return the list of entries."""
    with annotation_file.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    if not isinstance(annotations, list):
        raise ValueError(f"Annotation file must contain a list: {annotation_file}")
    return annotations


def load_annotation_map(annotation_file: Path) -> dict:
    """Return {video_filename: annotation_entry} from an annotation JSON file."""
    path = annotation_file.expanduser()
    if not path.exists():
        print(
            f"[annotations] no annotation file found at {path}; using live enrollment"
        )
        return {}
    try:
        annotations = load_annotations(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[annotations] could not load {path}: {exc}; using live enrollment")
        return {}
    return {entry.get("video"): entry for entry in annotations if entry.get("video")}


def load_reference_embeddings(
    annotation: dict | None, embedder, annotation_dir: Path
) -> list:
    """
    Embed saved reference crops listed in an annotation entry.

    Returns a (possibly empty) list of L2-normalised embedding vectors.
    Falls back to an empty list when no usable crops are found.
    """
    if not annotation or embedder is None:
        return []

    reference_frames = annotation.get("reference_frames") or []
    embeddings = []
    for reference in reference_frames:
        image_path = resolve_reference_image_path(
            reference.get("image"), annotation_dir
        )
        if image_path is None:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[enroll refs] could not read reference crop: {image_path}")
            continue

        height, width = frame.shape[:2]
        embedding = embedder.embed(frame, (0, 0, width, height))
        if embedding is None:
            print(f"[enroll refs] could not embed reference crop: {image_path}")
            continue

        embeddings.append(embedding)

    if reference_frames and not embeddings:
        print(
            f"[enroll refs] no usable reference crops for {annotation.get('video')}; "
            "using live enrollment"
        )
    return embeddings


def resolve_reference_image_path(image, annotation_dir: Path):
    """Resolve a potentially relative reference image path against annotation_dir."""
    if not image:
        return None

    image_path = Path(image).expanduser()
    if image_path.is_absolute():
        return image_path if image_path.exists() else None

    if image_path.exists():
        return image_path

    candidate = annotation_dir / image_path
    return candidate if candidate.exists() else image_path
