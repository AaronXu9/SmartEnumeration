"""Tests for the budget-sharing allocator (allocate_budget) that replaces
target-K in Phase 3.

Structure: a passing-MEL is a namespace with (row, remainder, expected_hits).
The allocator shares a commit budget B across passing MELs weighted by
expected_hits^alpha, respecting per-MEL caps (remainder) and a floor.
"""
from __future__ import annotations

import types
import unittest

import run_srg_batch


def _pr(row: int, remainder: int, expected_hits: float):
    """Minimal ProbeResult-like object for the pure allocator."""
    return types.SimpleNamespace(
        row=row, remainder=remainder, expected_hits=expected_hits)


class TestAllocatorBasic(unittest.TestCase):

    def test_alpha_one_proportional(self):
        # MEL 5 has 10x the expected_hits of MEL 2 → gets ~10x the commit.
        passing = [_pr(2, 10_000, 100.0), _pr(5, 10_000, 1000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=5500, alpha=1.0, min_commit=0)
        # Ratio ≈ 1:10 → ~500 : 5000. Within ±2% rounding tolerance.
        self.assertAlmostEqual(alloc[2] / 500, 1.0, delta=0.02)
        self.assertAlmostEqual(alloc[5] / 5000, 1.0, delta=0.02)
        self.assertEqual(sum(alloc.values()), 5500)

    def test_alpha_zero_uniform(self):
        # alpha=0 → all MELs get equal share regardless of expected_hits.
        passing = [_pr(2, 10_000, 1.0),
                   _pr(5, 10_000, 5000.0),
                   _pr(6, 10_000, 1_000_000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=3000, alpha=0.0, min_commit=0)
        # Rounded to int; exact equality not guaranteed but should be within 1.
        vals = sorted(alloc.values())
        self.assertLessEqual(vals[-1] - vals[0], 1)

    def test_cap_triggers_and_spills(self):
        # MEL 2 has small remainder (50) but huge expected_hits.
        # Raw allocation would exceed 50; cap at 50; spill to MEL 5.
        passing = [_pr(2, 50, 1000.0), _pr(5, 10_000, 1000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=2000, alpha=1.0, min_commit=0)
        self.assertEqual(alloc[2], 50)
        self.assertEqual(alloc[5], 1950)

    def test_floor_lifts_small_allocs(self):
        # MEL 2's raw allocation is much smaller than min_commit.
        passing = [_pr(2, 10_000, 1.0), _pr(5, 10_000, 10_000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=10_000, alpha=1.0, min_commit=500)
        self.assertEqual(alloc[2], 500)          # floored up
        self.assertGreater(alloc[5], 9000)

    def test_floor_clipped_by_remainder(self):
        # Floor=500 but remainder=100 → final commit_n=100, not 500.
        passing = [_pr(2, 100, 1.0), _pr(5, 10_000, 10_000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=10_000, alpha=1.0, min_commit=500)
        self.assertEqual(alloc[2], 100)

    def test_budget_conservation_no_cap_no_floor(self):
        passing = [_pr(i, 100_000, float(i)) for i in range(1, 6)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=10_000, alpha=1.0, min_commit=0)
        self.assertEqual(sum(alloc.values()), 10_000)

    def test_budget_zero_with_floor(self):
        # Budget=0 but min_commit=500 → each passing MEL gets floor.
        passing = [_pr(2, 10_000, 100.0), _pr(5, 10_000, 1000.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=0, alpha=1.0, min_commit=500)
        self.assertEqual(alloc[2], 500)
        self.assertEqual(alloc[5], 500)

    def test_empty_passing_returns_empty_dict(self):
        alloc = run_srg_batch.allocate_budget(
            [], budget=10_000, alpha=1.0, min_commit=500)
        self.assertEqual(alloc, {})

    def test_single_passing_mel_gets_everything(self):
        passing = [_pr(2, 100_000, 500.0)]
        alloc = run_srg_batch.allocate_budget(
            passing, budget=42_000, alpha=1.0, min_commit=500)
        self.assertEqual(alloc[2], 42_000)


class TestAllocatorWorkedExample(unittest.TestCase):
    """The 5-passing-MEL example from the Phase-3 plan. Sanity check that
    the allocator gives rich MELs (high expected_hits) the lion's share."""

    def test_plan_worked_example(self):
        # From results_local_macos/adaptive/batch_manifest.tsv,
        # passing rows under the old direction (placeholder until re-probe).
        passing = [
            _pr(2,  4493,   79.9),
            _pr(5, 31456, 5138.8),
            _pr(6, 28215, 4597.1),
            _pr(8, 11916,   59.9),
            _pr(9, 25886, 8876.4),
        ]
        total_remainder = sum(p.remainder for p in passing)
        budget = int(0.5 * total_remainder)  # commit_budget_frac=0.5

        alloc = run_srg_batch.allocate_budget(
            passing, budget=budget, alpha=1.0, min_commit=500)

        # Rich MELs (5, 6, 9) must get much larger commits than sparse (2, 8).
        self.assertGreater(alloc[5], alloc[2] * 10)
        self.assertGreater(alloc[9], alloc[8] * 10)
        # Sparse MELs floored to 500.
        self.assertEqual(alloc[2], 500)
        self.assertEqual(alloc[8], 500)
        # No allocation exceeds its remainder.
        for p in passing:
            self.assertLessEqual(alloc[p.row], p.remainder)


if __name__ == "__main__":
    unittest.main()
