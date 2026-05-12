"""Tests for the MLRegressionAllocator (skipped if scikit-learn missing)."""
from __future__ import annotations

import types
import unittest

try:
    import sklearn  # noqa: F401
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _pr(row, remainder, expected_hits=0.0):
    return types.SimpleNamespace(row=row, remainder=remainder,
                                 expected_hits=expected_hits)


@unittest.skipUnless(_SKLEARN, "scikit-learn not installed — install in the "
                                "OpenVsynthes mamba env to enable the ML policy")
class TestMLRegression(unittest.TestCase):

    def test_falls_back_uniform_with_no_history(self):
        # No labeled examples → policy returns uniform-by-remainder weights,
        # so the allocation is split proportionally to remainder.
        from al_policies import EmptyHistory
        from al_policies.ml import MLRegressionAllocator
        pol = MLRegressionAllocator()
        passing = [_pr(2, 5000), _pr(5, 5000)]
        out = pol.allocate(passing, budget=200, history=EmptyHistory(),
                           alpha=1.0, min_commit=0)
        self.assertEqual(out[2], out[5])

    def test_train_then_allocate(self):
        # With enough labeled examples the regressor fits and produces
        # non-trivial weights. We don't assert which MEL wins — only that
        # the allocator runs and returns a valid dict summing to ~budget.
        from al_policies import DictHistory
        from al_policies.ml import MLRegressionAllocator
        pol = MLRegressionAllocator(n_estimators=5)  # small for test speed
        # 5 MELs with varying hit rates in their probe scores.
        passing = [_pr(r, 5000) for r in (2, 3, 5, 6, 7)]
        hist = DictHistory({
            2: [-30.0, -32.0, -28.0, -29.0, -27.0],   # mostly hits
            3: [-15.0, -10.0, -8.0, -5.0, +2.0],      # mid
            5: [+5.0, +10.0, +12.0, +8.0, +9.0],       # no hits
            6: [-26.0, -28.0, -25.5, -24.0, -23.0],   # boundary
            7: [+15.0, +20.0, +18.0, +14.0, +12.0],   # bad
        })
        out = pol.allocate(passing, budget=1000, history=hist,
                           alpha=1.0, min_commit=0)
        self.assertEqual(set(out.keys()), {2, 3, 5, 6, 7})
        self.assertGreaterEqual(sum(out.values()), 950)
        self.assertLessEqual(sum(out.values()), 1050)

    def test_registered(self):
        from al_policies import get
        from al_policies.ml import MLRegressionAllocator
        self.assertIsInstance(get("ml"), MLRegressionAllocator)


if __name__ == "__main__":
    unittest.main()
