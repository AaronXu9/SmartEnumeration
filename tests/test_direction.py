"""Tests for the RTCNN_Score lower-is-better direction fix.

RTCNN_Score is a binding-energy proxy: more-negative = better. After the
Phase-3 fix, all sort and hit-threshold logic must treat lower as better.
"""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import sdf_utils
import run_srg_batch


def _record(rid: str, rtcnn: float) -> str:
    """Build a minimal SDF record with the two tags we care about."""
    return (
        f"{rid}\n"
        "  dummy mol block\n"
        "\n"
        "\n"
        "> <full_synthon_id>\n"
        f"{rid}\n"
        "\n"
        "> <RTCNN_Score>\n"
        f"{rtcnn}\n"
        "\n"
    )


class TestMergeSortedAscending(unittest.TestCase):
    """merge_sorted_by_rtcnn must put lowest score first (best under
    lower=better)."""

    def test_ascending_order(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_a = td_path / "a.sdf"
            in_b = td_path / "b.sdf"
            out = td_path / "merged.sdf"

            # Two input files, scores interleaved across them.
            sdf_utils.write_sdf(
                [_record("s1", 20.0), _record("s2", -10.0)], in_a)
            sdf_utils.write_sdf(
                [_record("s3", 0.0), _record("s4", -30.0)], in_b)

            n = sdf_utils.merge_sorted_by_rtcnn([in_a, in_b], out)
            self.assertEqual(n, 4)

            scores = [s for _, s in sdf_utils.iter_rtcnn(out)]
            self.assertEqual(scores, [-30.0, -10.0, 0.0, 20.0])

    def test_unscored_records_go_to_tail(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_a = td_path / "a.sdf"
            out = td_path / "merged.sdf"

            sdf_utils.write_sdf(
                [_record("s1", 20.0),
                 # no RTCNN tag
                 "s2\n  dummy\n\n\n> <full_synthon_id>\ns2\n\n",
                 _record("s3", -30.0)],
                in_a)
            sdf_utils.merge_sorted_by_rtcnn([in_a], out)

            scored = [s for _, s in sdf_utils.iter_rtcnn(out)]
            # Scored records come first, in ascending order.
            self.assertEqual(scored, [-30.0, 20.0])


class TestEvaluateProbeLowerIsBetter(unittest.TestCase):
    """evaluate_probe must count hits with `score <= threshold` and report
    the minimum (best) score as probe_best."""

    def _args(self, **overrides):
        defaults = dict(
            stop_criterion="expected-hits",
            hit_threshold=-25.0,
            min_expected_hits=10.0,
            top_threshold=-30.0,
            percentile=5.0,
            pct_threshold=-28.0,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_expected_hits_pass(self):
        # Scores: one hit at -30 (below -25 threshold).
        scores = [-30.0, -10.0, 0.0, 20.0]
        args = self._args()
        d = run_srg_batch.evaluate_probe(scores, n_total=100, n_probe=4, args=args)
        self.assertEqual(d["probe_hits_at_threshold"], 1)
        self.assertEqual(d["probe_best"], -30.0)
        self.assertAlmostEqual(d["expected_hits_total"], 25.0)
        self.assertEqual(d["decision"], "pass")

    def test_expected_hits_abort(self):
        # No score <= -25.
        scores = [-20.0, -10.0, 0.0, 20.0]
        args = self._args()
        d = run_srg_batch.evaluate_probe(scores, n_total=100, n_probe=4, args=args)
        self.assertEqual(d["probe_hits_at_threshold"], 0)
        self.assertEqual(d["decision"], "abort")

    def test_top_score_pass(self):
        # min(scores) = -35 <= -30 threshold → pass.
        scores = [-35.0, -10.0, 0.0]
        args = self._args(stop_criterion="top-score")
        d = run_srg_batch.evaluate_probe(scores, n_total=100, n_probe=3, args=args)
        self.assertEqual(d["probe_best"], -35.0)
        self.assertEqual(d["decision"], "pass")

    def test_top_score_abort(self):
        # min(scores) = -20 > -30 threshold → abort.
        scores = [-20.0, -10.0, 0.0]
        args = self._args(stop_criterion="top-score")
        d = run_srg_batch.evaluate_probe(scores, n_total=100, n_probe=3, args=args)
        self.assertEqual(d["decision"], "abort")

    def test_percentile_pass(self):
        # P5 of 100 scores spanning [-50, 49] is near -47.5. <= -28 → pass.
        scores = list(range(-50, 50))
        args = self._args(stop_criterion="percentile")
        d = run_srg_batch.evaluate_probe(scores, n_total=1000, n_probe=100, args=args)
        self.assertEqual(d["decision"], "pass")

    def test_percentile_abort(self):
        # P5 of all-positive scores > -28 → abort.
        scores = list(range(10, 110))
        args = self._args(stop_criterion="percentile")
        d = run_srg_batch.evaluate_probe(scores, n_total=1000, n_probe=100, args=args)
        self.assertEqual(d["decision"], "abort")


if __name__ == "__main__":
    unittest.main()
