import json
import tempfile
import unittest
from pathlib import Path

from action.generate_dataset import assert_no_holdout_videos
from utils.annotate_intervals import (
    add_event_mark,
    load_existing_annotations,
    merge_or_replace_events,
    save_annotations,
)
from utils.annotations import holdout_video_names


class HoldoutGuardTests(unittest.TestCase):
    def test_holdout_names_lists_only_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            holdout = Path(tmp)
            (holdout / "IMG_1.mp4").touch()
            (holdout / "IMG_2.MOV").touch()
            (holdout / "impact_events.json").touch()

            self.assertEqual(
                holdout_video_names(holdout), {"IMG_1.mp4", "IMG_2.MOV"}
            )

    def test_missing_holdout_dir_is_empty(self):
        self.assertEqual(holdout_video_names(Path("does/not/exist")), set())
        self.assertEqual(holdout_video_names(None), set())

    def test_leaked_holdout_video_aborts_generation(self):
        with self.assertRaises(SystemExit) as ctx:
            assert_no_holdout_videos(
                ["IMG_1.mp4", "IMG_9.mp4"], {"IMG_9.mp4"}, "generated samples"
            )
        self.assertIn("IMG_9.mp4", str(ctx.exception))

    def test_clean_video_list_passes(self):
        assert_no_holdout_videos(["IMG_1.mp4", None], {"IMG_9.mp4"}, "source")

    def test_no_holdout_configured_is_inert(self):
        assert_no_holdout_videos(["IMG_1.mp4"], set(), "source")


class ImpactEventTests(unittest.TestCase):
    def test_events_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "impact_events.json"
            events = merge_or_replace_events(
                [], "IMG_1.mp4", 59.94, [38.911, 12.4444], [22.1]
            )
            save_annotations(out_file, events)

            loaded = load_existing_annotations(out_file)

            self.assertEqual(loaded, events)
            self.assertEqual(
                loaded[0],
                {
                    "video": "IMG_1.mp4",
                    "fps": 59.94,
                    "impacts_sec": [12.444, 38.911],
                    "practice_swings_sec": [22.1],
                },
            )
            # Relabeling the same marks must not change the file.
            relabeled = merge_or_replace_events(
                loaded,
                "IMG_1.mp4",
                loaded[0]["fps"],
                loaded[0]["impacts_sec"],
                loaded[0]["practice_swings_sec"],
            )
            self.assertEqual(relabeled, events)

    def test_relabeling_replaces_only_that_video(self):
        events = merge_or_replace_events([], "IMG_1.mp4", 30.0, [1.0], [])
        events = merge_or_replace_events(events, "IMG_2.mp4", 30.0, [2.0], [])
        events = merge_or_replace_events(events, "IMG_1.mp4", 30.0, [1.5], [3.0])

        by_video = {e["video"]: e for e in events}
        self.assertEqual(len(events), 2)
        self.assertEqual(by_video["IMG_1.mp4"]["impacts_sec"], [1.5])
        self.assertEqual(by_video["IMG_1.mp4"]["practice_swings_sec"], [3.0])
        self.assertEqual(by_video["IMG_2.mp4"]["impacts_sec"], [2.0])

    def test_duplicate_mark_on_same_frame_is_rejected(self):
        impacts = []
        practice = []

        self.assertTrue(add_event_mark(impacts, practice, 12.4441, "impact"))
        self.assertFalse(add_event_mark(impacts, practice, 12.4442, "impact"))
        self.assertFalse(add_event_mark(practice, impacts, 12.4442, "practice"))
        self.assertTrue(add_event_mark(practice, impacts, 12.5, "practice"))

        self.assertEqual(impacts, [12.444])
        self.assertEqual(practice, [12.5])


if __name__ == "__main__":
    unittest.main()
