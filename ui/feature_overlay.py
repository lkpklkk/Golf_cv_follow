"""
Debug panel showing the classifier's 19 input features as numbers plus bars.

Drawn down the right edge of a frame: one row per feature, the numeric value on
the label line and an animated bar beneath it. Kept separate from ui/overlay.py
because both the dataset debug clips and the offline test runner draw it, and
neither owns the other's overlay.
"""

from __future__ import annotations

import cv2
import numpy as np

from action.preprocessing import FEATURE_NAMES

# Full names overflow any sane panel width at readable font sizes.
SHORT_NAMES = {
    "left_wrist_shoulder_distance": "L_wrist_dist",
    "right_wrist_shoulder_distance": "R_wrist_dist",
    "left_wrist_relative_speed": "L_wrist_speed",
    "right_wrist_relative_speed": "R_wrist_speed",
    "shoulder_rotation_delta": "shoulder_rot_d",
    "hip_rotation_delta": "hip_rot_d",
}

PANEL_WIDTH_RATIO = 0.26
PANEL_ALPHA = 0.55
BAR_COLOR = (120, 220, 120)
BAR_NEGATIVE_COLOR = (120, 180, 255)
OVERRANGE_COLOR = (80, 80, 255)
TEXT_COLOR = (235, 235, 235)
VALUE_COLOR = (170, 230, 255)
NO_DATA_COLOR = (110, 110, 110)
TRACK_COLOR = (70, 70, 70)


def short_name(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def draw_feature_panel(
    frame,
    values,
    stats=None,
    title=None,
    no_data=False,
):
    """
    Render the feature panel onto `frame` in-place and return it.

    Args:
        frame:   BGR numpy array.
        values:  (19,) sequence of feature values for the current frame.
        stats:   dict from action.feature_stats.load_feature_stats(), or None.
                 Without it the numbers still render and the bars are omitted —
                 a missing stats file should not cost the whole panel.
        title:   optional heading string.
        no_data: draw the rows dimmed, for frames with no usable pose.

    Returns:
        The same frame object.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} feature values, got {len(values)}"
        )

    height, width = frame.shape[:2]
    feature_stats = (stats or {}).get("features", {})
    layout = _layout(width, height, bool(title))

    _draw_backdrop(frame, layout)

    y = layout["top"] + layout["row_height"]
    if title:
        cv2.putText(
            frame,
            _fit_text(title, layout),
            (layout["left"] + layout["pad"], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            layout["font_scale"],
            TEXT_COLOR,
            layout["thickness"],
            cv2.LINE_AA,
        )
        y += layout["row_height"]

    for name, value in zip(FEATURE_NAMES, values):
        entry = feature_stats.get(name)
        _draw_row(frame, layout, y, name, float(value), entry, no_data)
        y += layout["row_height"] + layout["bar_height"] + layout["row_gap"]

    return frame


def _layout(width, height, has_title):
    """Size everything off the frame so the panel reads at 720p and at 4K."""
    rows = len(FEATURE_NAMES) + (1 if has_title else 0)
    panel_width = int(min(max(width * PANEL_WIDTH_RATIO, 180), 520))
    pad = max(6, panel_width // 40)

    # Fit rows to the available height, but cap the pitch: on a 4K frame an
    # uncapped row would be ~108px tall and leave the text swimming in it.
    usable = height - 2 * pad
    per_row = min(max(usable / max(rows, 1), 9.0), 46.0)
    row_height = int(per_row * 0.50)
    bar_height = max(3, int(per_row * 0.20))
    bar_width = panel_width - 2 * pad

    # Constrained by both axes. Sizing on row height alone overflows a narrow
    # panel — a portrait 1080x1920 frame gets tall rows and a 280px panel, and
    # the label runs straight into the value ("L_wrist_dist+0.888").
    font_scale = max(0.28, min(1.2, row_height / 26.0, _width_limited_scale(bar_width)))
    thickness = 1 if font_scale < 0.5 else 2

    # Measured, not guessed: putText anchors on the baseline, so a bar placed a
    # fixed fraction below it slices through the descenders of g, p and y.
    (_, _), descent = cv2.getTextSize(
        "gpy", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    # Asymmetric on purpose: the bar hugs the label it belongs to and the slack
    # goes between rows. Even spacing makes each bar read as the next row's.
    label_gap = descent + 2
    row_gap = max(label_gap + 2, int(per_row * 0.22))
    left = width - panel_width

    return {
        "left": left,
        "top": pad,
        "width": panel_width,
        "pad": pad,
        "row_height": row_height,
        "bar_height": bar_height,
        "label_gap": label_gap,
        "row_gap": row_gap,
        "font_scale": font_scale,
        "thickness": thickness,
        "bar_left": left + pad,
        "bar_width": bar_width,
        "bottom": height - pad,
    }


def _fit_text(text, layout):
    """Trim a caption to the panel width; captions are free-form, unlike labels."""
    for length in range(len(text), 0, -1):
        candidate = text if length == len(text) else text[: length - 1] + "."
        (text_width, _), _ = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            layout["font_scale"],
            layout["thickness"],
        )
        if text_width <= layout["bar_width"]:
            return candidate
    return ""


def _width_limited_scale(bar_width):
    """Largest font at which the widest label and its value fit side by side."""
    widest = max((short_name(name) for name in FEATURE_NAMES), key=len)
    sample = f"{widest}  -0.000"
    (text_width, _), _ = cv2.getTextSize(sample, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    if text_width <= 0:
        return 1.2
    return bar_width / float(text_width)


def _draw_backdrop(frame, layout):
    height = frame.shape[0]
    region = frame[0:height, layout["left"] :]
    if region.size == 0:
        return
    shade = np.zeros_like(region)
    cv2.addWeighted(shade, PANEL_ALPHA, region, 1.0 - PANEL_ALPHA, 0.0, region)


def _draw_row(frame, layout, y, name, value, entry, no_data):
    if y > layout["bottom"]:
        return

    label_color = NO_DATA_COLOR if no_data else TEXT_COLOR
    cv2.putText(
        frame,
        short_name(name),
        (layout["bar_left"], y),
        cv2.FONT_HERSHEY_SIMPLEX,
        layout["font_scale"],
        label_color,
        layout["thickness"],
        cv2.LINE_AA,
    )

    text = "--" if no_data else f"{value:+.3f}"
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, layout["font_scale"], layout["thickness"]
    )
    cv2.putText(
        frame,
        text,
        (layout["bar_left"] + layout["bar_width"] - text_width, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        layout["font_scale"],
        NO_DATA_COLOR if no_data else VALUE_COLOR,
        layout["thickness"],
        cv2.LINE_AA,
    )

    if entry is None:
        return

    _draw_bar(frame, layout, y + layout["label_gap"], value, entry, no_data)


def _draw_bar(frame, layout, top, value, entry, no_data):
    left = layout["bar_left"]
    width = layout["bar_width"]
    height = layout["bar_height"]
    bottom = top + height
    if bottom > layout["bottom"] or width <= 0:
        return

    cv2.rectangle(frame, (left, top), (left + width, bottom), TRACK_COLOR, -1)
    if no_data:
        return

    scale = float(entry.get("scale", 0.0)) or 1.0
    signed = bool(entry.get("signed", False))
    ratio = value / scale
    # Values past the measured percentile are pinned to the end of the bar and
    # recoloured, so a saturated bar is visibly saturated rather than looking
    # like a legitimate maximum.
    over_range = abs(ratio) > 1.0
    ratio = float(np.clip(ratio, -1.0, 1.0))
    color = OVERRANGE_COLOR if over_range else (
        BAR_NEGATIVE_COLOR if value < 0 else BAR_COLOR
    )

    if signed:
        center = left + width // 2
        span = int(abs(ratio) * (width / 2))
        if span > 0:
            start = center - span if value < 0 else center
            cv2.rectangle(frame, (start, top), (start + span, bottom), color, -1)
        cv2.line(frame, (center, top), (center, bottom), (200, 200, 200), 1)
    else:
        span = int(abs(ratio) * width)
        if span > 0:
            cv2.rectangle(frame, (left, top), (left + span, bottom), color, -1)
