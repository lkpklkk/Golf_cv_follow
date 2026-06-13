import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import config
from camera_selector import _probe_cameras
from pipeline.pipeline import TrackerPipeline
from pipeline.session import TrackingSession
from reid.enroller import Enroller
from ui.overlay import _COCO_SKELETON
from utils.geometry import point_inside_box

LIVE_W = 840
LIVE_H = 472
CROP_W = 440
CROP_H = 248
SKEL_W = 280
SKEL_H = 360


class GolfTrackerGui:
    def __init__(self):
        self.pipeline = TrackerPipeline()
        self.session = TrackingSession()

        self.cap = None
        self.camera_index = config.CAMERA_INDEX
        self.running = False

        self.frame = None
        self.raw_frame = None
        self.fps = 0.0
        self._last_time = cv2.getTickCount()
        self.live_scale_x = 1.0
        self.live_scale_y = 1.0
        self.crop_scale_x = 1.0
        self.crop_scale_y = 1.0

        self.frozen_frame = None
        self.crop_start = None
        self.crop_end = None
        self.gallery_tiles = []
        self._gallery_placeholder_visible = True
        self._gallery_texture_counter = 0

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Golf Cart CV Tracker", width=1580, height=900)

        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                LIVE_W,
                LIVE_H,
                self._blank_texture(LIVE_W, LIVE_H),
                tag="live_texture",
            )
            dpg.add_dynamic_texture(
                CROP_W,
                CROP_H,
                self._blank_texture(CROP_W, CROP_H),
                tag="crop_texture",
            )
            dpg.add_dynamic_texture(
                SKEL_W,
                SKEL_H,
                self._blank_texture(SKEL_W, SKEL_H),
                tag="skeleton_texture",
            )
            dpg.add_texture_registry(tag="gallery_texture_registry")

        with dpg.window(tag="main_window", no_title_bar=True, no_resize=True):
            with dpg.group(horizontal=True):
                self._build_live_panel()
                self._build_gallery_panel()
                self._build_skeleton_panel()

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        self._open_camera(self.camera_index)
        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()
        self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.pipeline.close()
        dpg.destroy_context()

    def _build_live_panel(self):
        with dpg.child_window(width=880, height=860, border=True):
            dpg.add_text("Live Feed")
            dpg.add_text("No user selected", tag="status_text", color=(240, 220, 80))
            dpg.add_image("live_texture", tag="live_image")
            with dpg.item_handler_registry(tag="live_handlers"):
                dpg.add_item_clicked_handler(callback=self._on_live_clicked)
            dpg.bind_item_handler_registry("live_image", "live_handlers")

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start", callback=lambda *args: self._set_running(True)
                )
                dpg.add_button(
                    label="Stop", callback=lambda *args: self._set_running(False)
                )
                dpg.add_button(label="360 Enroll", callback=self._start_360)
                dpg.add_button(label="Single Enroll", callback=self._start_single)
                dpg.add_button(label="Clear", callback=self._clear_selection)

            with dpg.group(horizontal=True):
                dpg.add_text("Camera")
                cameras = _probe_cameras()
                labels = [f"{idx}: {label}" for idx, label in cameras] or [
                    f"{config.CAMERA_INDEX}: Default"
                ]
                dpg.add_combo(
                    labels,
                    default_value=labels[0],
                    width=220,
                    callback=self._on_camera_selected,
                    user_data=[idx for idx, _ in cameras] or [config.CAMERA_INDEX],
                )
                dpg.add_text("", tag="fps_text")
            dpg.add_text("", tag="enroll_text", wrap=850)

    def _build_gallery_panel(self):
        with dpg.child_window(width=460, height=860, border=True):
            dpg.add_text("Re-ID Gallery")
            dpg.add_text("Manual Crop")
            dpg.add_image("crop_texture", tag="crop_image")
            with dpg.item_handler_registry(tag="crop_handlers"):
                dpg.add_item_clicked_handler(callback=self._on_crop_clicked)
            dpg.bind_item_handler_registry("crop_image", "crop_handlers")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Freeze Frame", callback=self._freeze_frame)
                dpg.add_button(label="Save Crop", callback=self._save_manual_crop)
                dpg.add_button(label="Reset Crop", callback=self._reset_crop)
            dpg.add_text(
                "Click two corners on the frozen frame, then Save Crop.", wrap=430
            )
            dpg.add_text("Gallery: 0 embeddings", tag="gallery_count")
            dpg.add_separator()
            with dpg.child_window(tag="gallery_tiles", height=430, border=False):
                dpg.add_text("Enrollment and manual gallery crops will appear here.")

    def _build_skeleton_panel(self):
        with dpg.child_window(width=300, height=860, border=True):
            dpg.add_text("Skeleton")
            dpg.add_text("Shown when selected target is in frame", wrap=270)
            dpg.add_text("Action: waiting", tag="action_text", wrap=270)
            dpg.add_text("", tag="reid_score_text", wrap=270)
            dpg.add_image("skeleton_texture")

    def _tick(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self._set_status("Camera frame unavailable", (255, 120, 80))
            return

        frame = cv2.flip(frame, 1)
        self.raw_frame = frame.copy()
        self.live_scale_x = frame.shape[1] / LIVE_W
        self.live_scale_y = frame.shape[0] / LIVE_H

        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        self.session = self.pipeline.tick(frame, timestamp)

        if self.session.gallery_updated:
            self._refresh_enrollment_gallery_tiles()
            dpg.set_value(
                "gallery_count",
                f"Gallery: {self.pipeline.matcher.gallery_size} embeddings",
            )

        display = self._draw_live_overlay(frame.copy())
        self.frame = display
        self._update_live_texture(display)
        self._update_skeleton_texture()
        self._update_status_text()

    def _draw_live_overlay(self, frame):
        status, status_color = self._selection_status()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (25, 25, 25), -1)
        cv2.putText(
            frame,
            status,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            status_color,
            2,
        )

        for person in self.session.people:
            x1, y1, x2, y2 = person["box"]
            track_id = person["track_id"]
            selected = track_id == self.session.target_track_id
            color = (0, 255, 0) if selected else (255, 80, 40)
            label = f"ID {track_id}"
            if self.session.reid_matches is not None:
                label += f" sim={self.session.reid_matches.get(track_id, 0.0):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 58)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            self._draw_pose(frame, person.get("keypoints"), color)

        if self.session.enroll_status:
            cv2.putText(
                frame,
                self.session.enroll_status,
                (16, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

        if self.session.action_prediction is not None:
            cv2.putText(
                frame,
                (
                    f"Action: {self.session.action_prediction.label} "
                    f"{self.session.action_prediction.confidence:.2f}"
                ),
                (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )

        return frame

    def _draw_pose(self, frame, keypoints, color):
        if keypoints is None:
            return
        thresh = config.POSE_KEYPOINT_CONFIDENCE_THRESHOLD
        for a, b in _COCO_SKELETON:
            if keypoints[a, 2] >= thresh and keypoints[b, 2] >= thresh:
                cv2.line(
                    frame,
                    (int(keypoints[a, 0]), int(keypoints[a, 1])),
                    (int(keypoints[b, 0]), int(keypoints[b, 1])),
                    color,
                    2,
                )
        for x, y, conf in keypoints:
            if conf >= thresh:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    def _update_live_texture(self, frame):
        self._update_texture("live_texture", frame, LIVE_W, LIVE_H)
        now = cv2.getTickCount()
        dt = (now - self._last_time) / cv2.getTickFrequency()
        self._last_time = now
        self.fps = 1.0 / dt if dt > 0 else 0.0
        dpg.set_value("fps_text", f"FPS {self.fps:.1f}")
        dpg.set_value("enroll_text", self.session.enroll_status or "")

        if self.pipeline.action_classifier is None:
            dpg.set_value("action_text", "Action: not loaded")
        elif self.session.action_prediction is None:
            dpg.set_value("action_text", "Action: waiting")
        else:
            dpg.set_value(
                "action_text",
                (
                    f"Action: {self.session.action_prediction.label} "
                    f"({self.session.action_prediction.confidence:.2f})"
                ),
            )

        score = (
            self.session.reid_matches.get(self.session.target_track_id)
            if self.session.reid_matches is not None
            and self.session.target_track_id is not None
            else None
        )
        mode = "verify" if self.session.tracked_person is not None else "recover"
        if score is not None:
            dpg.set_value("reid_score_text", f"ReID ({mode}): {score:.3f}")
        elif self.session.target_track_id is None or not self.pipeline.matcher.is_ready:
            dpg.set_value("reid_score_text", "")
        else:
            dpg.set_value("reid_score_text", f"ReID ({mode}): \u2014")

    def _update_skeleton_texture(self):
        canvas = np.full((SKEL_H, SKEL_W, 3), 18, dtype=np.uint8)
        tracked = self.session.tracked_person
        if tracked is not None:
            keypoints = tracked.get("keypoints")
            box = tracked["box"]
            if keypoints is not None:
                canvas = self._render_skeleton_only(keypoints, box)
        self._set_texture_pixels("skeleton_texture", canvas)

    def _render_skeleton_only(self, keypoints, box):
        canvas = np.full((SKEL_H, SKEL_W, 3), 18, dtype=np.uint8)
        x1, y1, x2, y2 = box
        bw = max(float(x2 - x1), 1.0)
        bh = max(float(y2 - y1), 1.0)
        scale = min((SKEL_W - 50) / bw, (SKEL_H - 50) / bh)
        ox = (SKEL_W - bw * scale) * 0.5
        oy = (SKEL_H - bh * scale) * 0.5
        pts = keypoints.copy()
        pts[:, 0] = (pts[:, 0] - x1) * scale + ox
        pts[:, 1] = (pts[:, 1] - y1) * scale + oy
        self._draw_pose(canvas, pts, (0, 220, 255))
        return canvas

    def _update_crop_texture(self):
        if self.frozen_frame is None:
            self._set_texture_pixels(
                "crop_texture",
                np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8),
            )
            return
        frame = self.frozen_frame.copy()
        if self.crop_start is not None and self.crop_end is not None:
            x1, y1, x2, y2 = self._crop_box_frame_coords()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
        self._update_texture("crop_texture", frame, CROP_W, CROP_H)

    def _on_live_clicked(self, *args):
        pos = self._item_mouse_pos("live_image")
        if pos is None:
            return
        x = int(pos[0] * self.live_scale_x)
        y = int(pos[1] * self.live_scale_y)
        for person in self.session.people:
            if point_inside_box((x, y), person["box"]):
                self.pipeline.select_target(person["track_id"])
                return

    def _on_crop_clicked(self, *args):
        if self.frozen_frame is None:
            return
        pos = self._item_mouse_pos("crop_image")
        if pos is None:
            return
        if self.crop_start is None or (
            self.crop_start is not None and self.crop_end is not None
        ):
            self.crop_start = pos
            self.crop_end = None
        else:
            self.crop_end = pos
        self._update_crop_texture()

    def _freeze_frame(self, *args):
        if self.raw_frame is None:
            return
        self.frozen_frame = self.raw_frame.copy()
        self.crop_scale_x = self.frozen_frame.shape[1] / CROP_W
        self.crop_scale_y = self.frozen_frame.shape[0] / CROP_H
        self.crop_start = None
        self.crop_end = None
        self._update_crop_texture()

    def _save_manual_crop(self, *args):
        if (
            self.frozen_frame is None
            or self.crop_start is None
            or self.crop_end is None
        ):
            return
        box = self._crop_box_frame_coords()
        if self.pipeline.enroll_manual_crop(self.frozen_frame, box):
            x1, y1, x2, y2 = box
            self._add_gallery_crop_tile(
                self.frozen_frame[y1:y2, x1:x2],
                title=f"Manual crop {self.pipeline.matcher.gallery_size}",
                detail="added to gallery",
            )
            dpg.set_value(
                "gallery_count",
                f"Gallery: {self.pipeline.matcher.gallery_size} embeddings",
            )
            self.pipeline.enroller.state = Enroller.DONE

    def _reset_crop(self, *args):
        self.crop_start = None
        self.crop_end = None
        self._update_crop_texture()

    def _refresh_enrollment_gallery_tiles(self):
        self._clear_gallery_tiles()
        for title, crop in self.pipeline.enroller.enrolled_gallery_items:
            self._add_gallery_crop_tile(
                crop,
                title=f"Enrollment {title}",
                detail="initial gallery",
            )

    def _add_gallery_crop_tile(self, crop, title, detail):
        if crop.size == 0:
            return
        if self._gallery_placeholder_visible:
            dpg.delete_item("gallery_tiles", children_only=True)
            self._gallery_placeholder_visible = False
        tag = f"gallery_tex_{self._gallery_texture_counter}"
        self._gallery_texture_counter += 1
        thumb = cv2.resize(crop, (96, 160))
        rgba = self._texture_data(thumb)
        dpg.add_dynamic_texture(
            96,
            160,
            self._blank_texture(96, 160),
            tag=tag,
            parent="gallery_texture_registry",
        )
        dpg.set_value(tag, rgba)
        with dpg.group(parent="gallery_tiles"):
            dpg.add_image(tag)
            dpg.add_text(title)
            dpg.add_text(detail)
        self.gallery_tiles.append(tag)

    def _crop_box_frame_coords(self):
        x1 = int(min(self.crop_start[0], self.crop_end[0]) * self.crop_scale_x)
        y1 = int(min(self.crop_start[1], self.crop_end[1]) * self.crop_scale_y)
        x2 = int(max(self.crop_start[0], self.crop_end[0]) * self.crop_scale_x)
        y2 = int(max(self.crop_start[1], self.crop_end[1]) * self.crop_scale_y)
        h, w = self.frozen_frame.shape[:2]
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    def _start_360(self, *args):
        self.pipeline.start_360_enroll()

    def _start_single(self, *args):
        self.pipeline.start_single_enroll()

    def _clear_selection(self, *args):
        self.pipeline.clear()
        self.session = TrackingSession()
        self._clear_gallery_tiles()
        dpg.set_value("gallery_count", "Gallery: 0 embeddings")

    def _clear_gallery_tiles(self):
        self.gallery_tiles = []
        self._gallery_placeholder_visible = True
        dpg.delete_item("gallery_tiles", children_only=True)
        dpg.add_text(
            "Enrollment and manual gallery crops will appear here.",
            parent="gallery_tiles",
        )

    def _on_camera_selected(self, sender, value, user_data):
        try:
            selected = int(value.split(":", 1)[0])
        except (ValueError, IndexError):
            selected = user_data[0]
        self._open_camera(selected)

    def _open_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        self.camera_index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.running = self.cap.isOpened()

    def _set_running(self, running):
        self.running = running and self.cap is not None and self.cap.isOpened()

    def _selection_status(self):
        if self.session.target_track_id is None:
            if self.pipeline.matcher.is_ready:
                return "Selected and out of frame", (0, 80, 255)
            return "No user selected", (0, 255, 255)
        if self.session.tracked_person is not None:
            return "Selected and in frame", (0, 255, 0)
        return "Selected and out of frame", (0, 80, 255)

    def _update_status_text(self):
        status, color = self._selection_status()
        dpg.set_value("status_text", status)
        dpg.configure_item("status_text", color=self._bgr_to_rgba(color))

    def _set_status(self, text, color):
        dpg.set_value("status_text", text)
        dpg.configure_item("status_text", color=self._bgr_to_rgba(color))

    def _item_mouse_pos(self, item):
        item_min = dpg.get_item_rect_min(item)
        mouse = dpg.get_mouse_pos(local=False)
        x = mouse[0] - item_min[0]
        y = mouse[1] - item_min[1]
        width, height = dpg.get_item_rect_size(item)
        if x < 0 or y < 0 or x > width or y > height:
            return None
        return (x, y)

    def _update_texture(self, tag, frame, width, height):
        resized = cv2.resize(frame, (width, height))
        self._set_texture_pixels(tag, resized)

    def _set_texture_pixels(self, tag, bgr):
        dpg.set_value(tag, self._texture_data(bgr))

    def _texture_data(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
        return np.dstack([rgb, alpha]).ravel()

    def _blank_texture(self, width, height):
        return np.zeros((height, width, 4), dtype=np.float32).ravel()

    def _bgr_to_rgba(self, color):
        return (color[2], color[1], color[0], 255)


def main():
    app = GolfTrackerGui()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
