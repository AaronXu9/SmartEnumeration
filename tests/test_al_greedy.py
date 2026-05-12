"""Tests for the EpsilonGreedyAllocator."""
from __future__ import annotations

import types
import unittest

from al_policies import DictHistory, EmptyHistory, get
from al_policies.greedy import EpsilonGreedyAllocator


def _pr(row, remainder, expected_hits=0.0):
    return types.SimpleNamespace(row=row, remainder=remainder,
                                 expected_hits=expected_hits)


class TestGreedyBasic(unittest.TestCase):

    def test_empty_passing(self):
        pol = EpsilonGreedyAllocator()
        self.assertEqual(
            pol.allocate([], budget=100, history=EmptyHistory(),
                         alpha=1.0, min_commit=0),
            {},
        )

    def test_single_mel_gets_all(self):
        pol = EpsilonGreedyAllocator(epsilon=0.1)
        passing = [_pr(2, 5000)]
        out = pol.allocate(passing, budget=500, history=EmptyHistory(),
                           alpha=1.0, min_commit=0)
        # No other MELs to spread ε to — the lone MEL gets everything.
        self.assertEqual(out, {2: 500})

    def test_best_mel_gets_majority(self):
        # MEL 2 has the best observed top-K mean → it's the exploit target.
        # ε=0.1, budget=1000 → exploit=900, explore=100 split across MEL 5 only.
        pol = EpsilonGreedyAllocator(epsilon=0.1, top_k=3)
        passing = [_pr(2, 5000), _pr(5, 5000)]
        hist = DictHistory({2: [-30.0, -32.0, -28.0], 5: [+10.0, +12.0]})
        out = pol.allocate(passing, budget=1000, history=hist,
                           alpha=1.0, min_commit=0)
        self.assertEqual(out[2], 900)
        self.assertEqual(out[5], 100)

    def test_exploit_capped_spills_to_explore(self):
        # MEL 2 is best but has a small remainder (50). exploit=450 caps at
        # 50; spill of 400 goes to MEL 5's explore share.
        pol = EpsilonGreedyAllocator(epsilon=0.1)
        passing = [_pr(2, 50), _pr(5, 5000)]
        hist = DictHistory({2: [-30.0, -32.0], 5: [+10.0]})
        out = pol.allocate(passing, budget=500, history=hist,
                           alpha=1.0, min_commit=0)
        self.assertEqual(out[2], 50)
        self.assertEqual(out[5], 450)
        self.assertEqual(sum(out.values()), 500)

    def test_floor_lifts(self):
        # MEL with explore-share=0 still gets the floor (clipped by remainder).
        pol = EpsilonGreedyAllocator(epsilon=0.0)  # pure greedy, ε=0
        passing = [_pr(2, 5000), _pr(5, 5000)]
        hist = DictHistory({2: [-30.0], 5: [+10.0]})
        out = pol.allocate(passing, budget=1000, history=hist,
                           alpha=1.0, min_commit=10)
        self.assertEqual(out[2], 1000)
        self.assertEqual(out[5], 10)         # lifted from 0 → 10

    def test_invalid_epsilon_rejected(self):
        with self.assertRaises(ValueError):
            EpsilonGreedyAllocator(epsilon=1.0)
        with self.assertRaises(ValueError):
            EpsilonGreedyAllocator(epsilon=-0.1)

    def test_registered(self):
        self.assertIsInstance(get("greedy"), EpsilonGreedyAllocator)


if __name__ == "__main__":
    unittest.main()
