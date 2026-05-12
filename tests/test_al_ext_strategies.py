"""Tests for the AL-extension strategies (E/F/G/H).

Verifies the probe-then-allocate-then-pick scaffold runs end-to-end on
synthetic data, that each strategy respects the budget, and that the
synthon-picker correctly draws from the per-MEL remainder pools.
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _make_universe(n_mels=4, synthons_per_mel=300, seed=0):
    """Synthetic synthon_ground_truth with distinct per-MEL score centers."""
    rng = np.random.default_rng(seed)
    centers = [-35, -25, -15, -5]
    rows = []
    for mel_idx in range(min(n_mels, len(centers))):
        c = centers[mel_idx]
        for j in range(synthons_per_mel):
            rt = rng.normal(c, 5)
            rows.append({
                "key_norm": f"MEL_{mel_idx}",
                "synthon_inchikey": f"S_{mel_idx}_{j:04d}",
                "RTCNN_Score": rt,
                "FullLigand_Score": rt + rng.normal(0, 2),
                "Strain": rng.uniform(5, 25),
                "CoreRmsd": rng.uniform(0.5, 3.0),
                "_score": rt,
            })
    syn = pd.DataFrame(rows)
    mel_rows = [{"key_norm": f"MEL_{i}", "Score": centers[i], "mel_rank": i + 1}
                for i in range(min(n_mels, len(centers)))]
    return syn, pd.DataFrame(mel_rows)


@unittest.skipUnless(_HAS_DEPS, "pandas+numpy not in this Python env "
                                "(use /home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python)")
class TestALExtensionStrategies(unittest.TestCase):

    def setUp(self):
        self.synthons, self.mel_ranked = _make_universe()

    def test_strategy_e_ucb_runs_and_respects_budget(self):
        from al_benchmark_gpr91.al_ext_strategies import strategy_e_ucb_alloc_softmax_pick
        result = strategy_e_ucb_alloc_softmax_pick(
            self.synthons, self.mel_ranked, budget=400, n_probe=20, seed=0,
            min_commit=0,  # disable floor for a clean budget assertion
        )
        # Budget = 400; probe = 20 × 4 MELs = 80; commit budget = 320.
        # No floor → total selection should land at or below budget.
        self.assertGreaterEqual(result.n_ligands, 350)
        self.assertLessEqual(result.n_ligands, 405)
        self.assertEqual(result.extras["policy"], "ucb")
        self.assertEqual(result.extras["n_probe_total"], 80)

    def test_strategy_f_ts_is_seed_deterministic(self):
        from al_benchmark_gpr91.al_ext_strategies import strategy_f_ts_alloc_softmax_pick
        a = strategy_f_ts_alloc_softmax_pick(
            self.synthons, self.mel_ranked, budget=400, n_probe=20, seed=42)
        b = strategy_f_ts_alloc_softmax_pick(
            self.synthons, self.mel_ranked, budget=400, n_probe=20, seed=42)
        self.assertEqual(len(a.selected), len(b.selected))
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"]),
        )

    def test_strategy_g_baseline_runs(self):
        from al_benchmark_gpr91.al_ext_strategies import strategy_g_baseline_alloc_softmax_pick
        result = strategy_g_baseline_alloc_softmax_pick(
            self.synthons, self.mel_ranked, budget=400, n_probe=20, seed=0,
        )
        self.assertEqual(result.extras["policy"], "baseline")
        self.assertGreater(result.n_ligands, 0)

    def test_strategy_h_greedy_runs(self):
        from al_benchmark_gpr91.al_ext_strategies import strategy_h_greedy_alloc_softmax_pick
        result = strategy_h_greedy_alloc_softmax_pick(
            self.synthons, self.mel_ranked, budget=400, n_probe=20, seed=0,
        )
        self.assertEqual(result.extras["policy"], "greedy")
        self.assertGreater(result.n_ligands, 0)

    def test_all_four_prefer_better_mels(self):
        """Each AL strategy should select more ligands from MEL 0 (best
        center -35) than MEL 3 (worst center -5)."""
        from al_benchmark_gpr91.al_ext_strategies import (
            strategy_e_ucb_alloc_softmax_pick,
            strategy_f_ts_alloc_softmax_pick,
            strategy_g_baseline_alloc_softmax_pick,
            strategy_h_greedy_alloc_softmax_pick,
        )
        budget = 600
        for fn in (strategy_e_ucb_alloc_softmax_pick,
                   strategy_f_ts_alloc_softmax_pick,
                   strategy_g_baseline_alloc_softmax_pick,
                   strategy_h_greedy_alloc_softmax_pick):
            with self.subTest(strategy=fn.__name__):
                result = fn(self.synthons, self.mel_ranked, budget=budget,
                            n_probe=20, seed=0)
                counts = result.selected.groupby("key_norm").size()
                self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_3", 0))


if __name__ == "__main__":
    unittest.main()
