"""Tests for the BaselineDynamicAllocator policy wrapper.

The baseline policy MUST produce bit-identical numerics to the legacy
`run_srg_batch.allocate_budget()` for any input — it's a thin wrapper
that exists only to fit the new AllocationPolicy Protocol. These tests
encode that contract.
"""
from __future__ import annotations

import types
import unittest

import run_srg_batch
from al_policies import EmptyHistory, get
from al_policies.baseline import BaselineDynamicAllocator


def _pr(row, remainder, expected_hits):
    return types.SimpleNamespace(row=row, remainder=remainder,
                                 expected_hits=expected_hits)


class TestBaselinePolicyParity(unittest.TestCase):
    """For every input the legacy tests cover, the policy must match."""

    def _check_parity(self, passing, budget, alpha, min_commit):
        legacy = run_srg_batch.allocate_budget(
            passing, budget=budget, alpha=alpha, min_commit=min_commit)
        pol = BaselineDynamicAllocator()
        new = pol.allocate(passing, budget=budget, history=EmptyHistory(),
                           alpha=alpha, min_commit=min_commit)
        self.assertEqual(legacy, new)

    def test_alpha_one_proportional(self):
        self._check_parity(
            [_pr(2, 10_000, 100.0), _pr(5, 10_000, 1000.0)],
            budget=5500, alpha=1.0, min_commit=0)

    def test_alpha_zero_uniform(self):
        self._check_parity(
            [_pr(2, 10_000, 1.0), _pr(5, 10_000, 5000.0),
             _pr(6, 10_000, 1_000_000.0)],
            budget=3000, alpha=0.0, min_commit=0)

    def test_cap_and_spill(self):
        self._check_parity(
            [_pr(2, 50, 1000.0), _pr(5, 10_000, 1000.0)],
            budget=2000, alpha=1.0, min_commit=0)

    def test_floor_clipped_by_remainder(self):
        self._check_parity(
            [_pr(2, 10, 1.0), _pr(5, 10_000, 1000.0)],
            budget=500, alpha=1.0, min_commit=50)

    def test_empty_passing(self):
        pol = BaselineDynamicAllocator()
        out = pol.allocate([], budget=100, history=EmptyHistory(),
                           alpha=1.0, min_commit=10)
        self.assertEqual(out, {})

    def test_registered_under_name_baseline(self):
        self.assertIsInstance(get("baseline"), BaselineDynamicAllocator)


class TestBaselineIgnoresHistory(unittest.TestCase):
    """The baseline policy must IGNORE the history arg — by design it relies
    only on the probe-summary `expected_hits`. Passing in junk history
    should not change the output."""

    def test_history_does_not_affect_output(self):
        passing = [_pr(2, 5000, 50.0), _pr(5, 5000, 5.0)]
        pol = BaselineDynamicAllocator()
        from al_policies import DictHistory
        a = pol.allocate(passing, budget=1000, history=EmptyHistory(),
                         alpha=1.0, min_commit=0)
        b = pol.allocate(
            passing, budget=1000,
            history=DictHistory({2: [-30, -32, -28], 5: [+12, +15]}),
            alpha=1.0, min_commit=0)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
