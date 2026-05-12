"""Tests for UCBAllocator and ThompsonSamplingAllocator."""
from __future__ import annotations

import types
import unittest

from al_policies import DictHistory, EmptyHistory, get
from al_policies.bandit import (
    ThompsonSamplingAllocator,
    UCBAllocator,
    _cap_spill,
)


def _pr(row, remainder, expected_hits=0.0):
    return types.SimpleNamespace(row=row, remainder=remainder,
                                 expected_hits=expected_hits)


class TestCapSpillHelper(unittest.TestCase):
    """The cap-and-spill helper is the load-bearing piece shared by all
    weight-based policies. Lock down its contract."""

    def test_empty(self):
        self.assertEqual(_cap_spill({}, {}, budget=100, min_commit=0), {})

    def test_uniform_weights(self):
        weights = {2: 1.0, 5: 1.0}
        remainders = {2: 1000, 5: 1000}
        out = _cap_spill(weights, remainders, budget=200, min_commit=0)
        self.assertEqual(out[2], 100)
        self.assertEqual(out[5], 100)

    def test_cap_and_spill(self):
        # MEL 2 has tiny remainder; spill should go to MEL 5.
        weights = {2: 1.0, 5: 1.0}
        remainders = {2: 50, 5: 10_000}
        out = _cap_spill(weights, remainders, budget=2000, min_commit=0)
        self.assertEqual(out[2], 50)
        self.assertEqual(out[5], 1950)

    def test_floor_clipped_by_remainder(self):
        weights = {2: 1.0, 5: 1.0}
        remainders = {2: 10, 5: 1000}
        out = _cap_spill(weights, remainders, budget=20, min_commit=50)
        # MEL 2's floor is clipped to its remainder (10).
        self.assertEqual(out[2], 10)
        self.assertEqual(out[5], 50)


class TestUCB(unittest.TestCase):

    def test_empty_passing(self):
        pol = UCBAllocator()
        self.assertEqual(pol.allocate([], budget=100, history=EmptyHistory(),
                                       alpha=1.0, min_commit=0), {})

    def test_uniform_with_no_history(self):
        # No observations → posteriors are the prior → all MELs equal weight.
        pol = UCBAllocator()
        passing = [_pr(2, 5000), _pr(5, 5000)]
        out = pol.allocate(passing, budget=200, history=EmptyHistory(),
                           alpha=1.0, min_commit=0)
        # Symmetric → equal allocation.
        self.assertEqual(out[2], out[5])

    def test_better_mel_gets_more(self):
        # MEL 2's observed scores are much better than MEL 5's.
        # Expected: MEL 2 gets the larger share.
        pol = UCBAllocator()
        passing = [_pr(2, 5000), _pr(5, 5000)]
        hist = DictHistory({2: [-30.0, -32.0, -28.0, -29.0],
                            5: [+8.0, +10.0, +12.0, +9.0]})
        out = pol.allocate(passing, budget=1000, history=hist,
                           alpha=1.0, min_commit=0)
        self.assertGreater(out[2], out[5])

    def test_registered(self):
        self.assertIsInstance(get("ucb"), UCBAllocator)


class TestThompsonSampling(unittest.TestCase):

    def test_empty_passing(self):
        pol = ThompsonSamplingAllocator(seed=0)
        self.assertEqual(pol.allocate([], budget=100, history=EmptyHistory(),
                                       alpha=1.0, min_commit=0), {})

    def test_deterministic_with_fixed_seed(self):
        # Same seed → same output across two calls. Tests that the seed is
        # latched at construction (per the bandit module's design).
        passing = [_pr(2, 5000), _pr(5, 5000)]
        hist = DictHistory({2: [-30.0], 5: [+10.0]})
        a = ThompsonSamplingAllocator(seed=42).allocate(
            passing, budget=200, history=hist, alpha=1.0, min_commit=0)
        b = ThompsonSamplingAllocator(seed=42).allocate(
            passing, budget=200, history=hist, alpha=1.0, min_commit=0)
        self.assertEqual(a, b)

    def test_budget_conservation_approximate(self):
        # Sum of allocations should be within ±|passing|*min_commit of budget.
        pol = ThompsonSamplingAllocator(seed=0)
        passing = [_pr(r, 5000) for r in (2, 3, 5, 6, 7)]
        hist = DictHistory({r: [float(-(r + 1))] for r in (2, 3, 5, 6, 7)})
        out = pol.allocate(passing, budget=2000, history=hist,
                           alpha=1.0, min_commit=10)
        self.assertGreaterEqual(sum(out.values()), 2000 - 10)
        self.assertLessEqual(sum(out.values()), 2000 + 5 * 10)

    def test_registered(self):
        self.assertIsInstance(get("ts"), ThompsonSamplingAllocator)


if __name__ == "__main__":
    unittest.main()
