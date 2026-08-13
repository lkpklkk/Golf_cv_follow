import unittest

from action.evaluate_detection import (
    aggregate_metrics,
    idle_seconds,
    load_ground_truth,
    match_events,
)


class MatchEventsTests(unittest.TestCase):
    def test_exact_and_close_matches_within_tolerance(self):
        result = match_events([12.4, 40.0], [12.44, 38.91], tolerance_sec=1.0)
        self.assertEqual(result["matches"], [(12.4, 12.44)])
        self.assertEqual(result["false_positives"], [40.0])
        self.assertEqual(result["missed"], [38.91])

    def test_outside_tolerance_is_unmatched(self):
        result = match_events([10.0], [12.5], tolerance_sec=1.0)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["false_positives"], [10.0])
        self.assertEqual(result["missed"], [12.5])

    def test_greedy_nearest_first_prefers_closest_pair(self):
        # event at 10.0 is within tolerance of both positives; nearest-first
        # must claim 10.3 (diff 0.3) over 9.5 (diff 0.5), leaving 9.5 missed.
        result = match_events([10.0], [9.5, 10.3], tolerance_sec=1.0)
        self.assertEqual(result["matches"], [(10.0, 10.3)])
        self.assertEqual(result["missed"], [9.5])

    def test_no_double_matching_when_events_cluster(self):
        result = match_events([10.0, 10.2], [10.1], tolerance_sec=1.0)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(len(result["false_positives"]), 1)

    def test_empty_inputs(self):
        result = match_events([], [], tolerance_sec=1.0)
        self.assertEqual(result, {"matches": [], "false_positives": [], "missed": []})


class IdleSecondsTests(unittest.TestCase):
    def test_subtracts_tolerance_band_per_positive(self):
        self.assertAlmostEqual(idle_seconds(100.0, 2, 1.0), 96.0)

    def test_never_negative(self):
        self.assertEqual(idle_seconds(1.0, 10, 1.0), 0.0)


class AggregateMetricsTests(unittest.TestCase):
    def test_pools_counts_and_computes_rates(self):
        reports = [
            {
                "tp": 2,
                "fp": 1,
                "positive_instances": 2,
                "idle_sec": 60.0,
                "latencies": [0.2, 0.4],
            },
            {
                "tp": 1,
                "fp": 1,
                "positive_instances": 2,
                "idle_sec": 60.0,
                "latencies": [0.6],
            },
        ]
        summary = aggregate_metrics(reports)
        self.assertEqual(summary["tp"], 3)
        self.assertEqual(summary["fp"], 2)
        self.assertEqual(summary["positive_instances"], 4)
        self.assertAlmostEqual(summary["recall"], 0.75)
        self.assertAlmostEqual(summary["precision"], 0.6)
        self.assertAlmostEqual(summary["fp_per_min_idle"], 2 / 2.0)
        self.assertAlmostEqual(summary["median_latency_sec"], 0.4)

    def test_output_is_re_aggregatable_across_videos(self):
        # evaluate_holdout() pools evaluate_video()'s per-video output through
        # aggregate_metrics() a second time, so that output must carry the
        # same tp/fp/positive_instances/idle_sec/latencies keys it consumed.
        per_video = aggregate_metrics(
            [{"tp": 1, "fp": 0, "positive_instances": 1, "idle_sec": 30.0, "latencies": [0.1]}]
        )
        video_report = {
            "tp": per_video["tp"],
            "fp": per_video["fp"],
            "positive_instances": per_video["positive_instances"],
            "idle_sec": 30.0,
            "latencies": [0.1],
            **per_video,
        }
        overall = aggregate_metrics([video_report, video_report])
        self.assertEqual(overall["tp"], 2)
        self.assertAlmostEqual(overall["fp_per_min_idle"], 0.0)

    def test_handles_zero_denominators(self):
        summary = aggregate_metrics(
            [{"tp": 0, "fp": 0, "positive_instances": 0, "idle_sec": 0.0, "latencies": []}]
        )
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["precision"])
        self.assertIsNone(summary["fp_per_min_idle"])
        self.assertIsNone(summary["median_latency_sec"])


class LoadGroundTruthTests(unittest.TestCase):
    def test_holdout_ground_truth_unions_impacts_and_practice_swings(self):
        ground_truth = load_ground_truth("data/hold_out/impact_events.json")
        entry = ground_truth["596_raw - 01 - 01.mp4"]
        self.assertEqual(entry, [1.856, 17.522])


if __name__ == "__main__":
    unittest.main()
