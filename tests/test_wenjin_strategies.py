"""Tests for the Wenjin strategy ports.

Uses synthetic per-MEL synthon DataFrames so the tests run without
the real csv/all_mels_combined_core.csv (which lives on the user's
Mac as of 2026-05-11). Each test constructs a tiny universe with
known properties so the expected output is computable by hand.

Requires the OpenVsynthes008 mamba env (pandas + numpy).
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _make_synthon_universe(n_mels=4, synthons_per_mel=100, seed=0):
    """Synthetic synthon_ground_truth-style DataFrame.

    Per-MEL score distributions are intentionally different so the
    strategies have something to discriminate on:
      MEL A (rank 1, best): scores N(-35, 5)
      MEL B (rank 2):       scores N(-25, 5)
      MEL C (rank 3):       scores N(-15, 5)
      MEL D (rank 4):       scores N(-5,  5)
    FullLigand_Score correlates with RTCNN_Score with noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    centers = [-35, -25, -15, -5]
    for mel_idx in range(min(n_mels, len(centers))):
        center = centers[mel_idx]
        for j in range(synthons_per_mel):
            rtcnn = rng.normal(center, 5)
            full = rtcnn + rng.normal(0, 2)
            rows.append({
                "key_norm": f"MEL_{mel_idx}",
                "synthon_inchikey": f"S_{mel_idx}_{j:04d}",
                "RTCNN_Score": rtcnn,
                "FullLigand_Score": full,
                "Strain": rng.uniform(5, 25),
                "CoreRmsd": rng.uniform(0.5, 3.0),
            })
    syn = pd.DataFrame(rows)

    mel_rows = []
    for mel_idx, center in enumerate(centers[:n_mels]):
        mel_rows.append({
            "key_norm": f"MEL_{mel_idx}",
            "Score": center,
            "mel_rank": mel_idx + 1,
        })
    mel_ranked = pd.DataFrame(mel_rows)
    return syn, mel_ranked


@unittest.skipUnless(_HAS_DEPS, "pandas+numpy not in this Python env "
                                "(use /home/aoxu/miniconda3/envs/OpenVsynthes008/bin/python)")
class TestWenjinStrategies(unittest.TestCase):

    def setUp(self):
        self.synthons, self.mel_ranked = _make_synthon_universe()

    def test_prepare_scored_pool_option1_uses_rtcnn(self):
        from al_benchmark_gpr91 import prepare_scored_pool
        scored = prepare_scored_pool(self.synthons, option=1)
        self.assertTrue((scored["_score"] == scored["RTCNN_Score"]).all())

    def test_prepare_scored_pool_option2_applies_filter(self):
        from al_benchmark_gpr91 import prepare_scored_pool
        scored = prepare_scored_pool(self.synthons, option=2)
        self.assertTrue((scored["Strain"] <= 15).all())
        self.assertTrue((scored["CoreRmsd"] <= 2.0).all())
        self.assertLess(len(scored), len(self.synthons))

    def test_prepare_scored_pool_rejects_bad_option(self):
        from al_benchmark_gpr91 import prepare_scored_pool
        with self.assertRaises(ValueError):
            prepare_scored_pool(self.synthons, option=99)

    def test_vs_baseline_walks_in_rank_order_until_budget(self):
        from al_benchmark_gpr91 import vs_baseline_rank_walk
        # Budget = 150 synthons → covers MEL 0 (100 synthons) + 50 from MEL 1.
        # vs_baseline_rank_walk INCLUDES the MEL that crosses the budget
        # boundary, so it includes both MEL 0 and MEL 1 (cumulative = 200).
        baseline_df, baseline_ligands = vs_baseline_rank_walk(
            self.mel_ranked, self.synthons, budget=150
        )
        self.assertEqual(list(baseline_df["mel_rank"]), [1, 2])
        self.assertEqual(len(baseline_ligands), 200)

    def test_strategy_a_keeps_top_fraction_globally(self):
        from al_benchmark_gpr91 import (prepare_scored_pool,
                                         strategy_a_global_hard_cutoff)
        scored = prepare_scored_pool(self.synthons, option=1)
        result = strategy_a_global_hard_cutoff(
            scored, self.mel_ranked, top_frac=0.25, budget=1_000_000
        )
        # The cutoff is the 25th percentile of RTCNN_Score across all 400.
        # The pool should be ~25% of the 400 = ~100 synthons; MEL 0 (best
        # scores) should dominate.
        self.assertGreater(result.n_ligands, 0)
        self.assertLessEqual(result.n_ligands, 110)  # ~100 + slop
        # MEL 0 should be in the result; MEL 3 likely should not.
        sel_keys = set(result.selected["key_norm"])
        self.assertIn("MEL_0", sel_keys)
        self.assertNotIn("MEL_3", sel_keys)

    def test_strategy_b_takes_top_fraction_per_mel(self):
        from al_benchmark_gpr91 import (prepare_scored_pool,
                                         strategy_b_greedy_per_mel)
        scored = prepare_scored_pool(self.synthons, option=1)
        result = strategy_b_greedy_per_mel(
            scored, self.mel_ranked, fraction=0.10, budget=1_000_000
        )
        # 10% of 100 = 10 per MEL × 4 MELs = 40 ligands total.
        self.assertEqual(result.n_ligands, 40)
        self.assertEqual(result.n_mels, 4)

    def test_strategy_c_softmax_is_seed_deterministic(self):
        from al_benchmark_gpr91 import (prepare_scored_pool,
                                         strategy_c_softmax_per_mel)
        scored = prepare_scored_pool(self.synthons, option=1)
        a = strategy_c_softmax_per_mel(scored, self.mel_ranked, T=1.0,
                                        per_mel_cap=30, budget=120, seed=42)
        b = strategy_c_softmax_per_mel(scored, self.mel_ranked, T=1.0,
                                        per_mel_cap=30, budget=120, seed=42)
        self.assertEqual(len(a.selected), len(b.selected))
        # Same seed → same selection.
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"])
        )

    def test_strategy_c_respects_per_mel_cap(self):
        from al_benchmark_gpr91 import (prepare_scored_pool,
                                         strategy_c_softmax_per_mel)
        scored = prepare_scored_pool(self.synthons, option=1)
        # PER_MEL_CAP=30 with 4 MELs × 100 synthons each: first pass gives
        # 30 per MEL = 120 total. Budget=120 hits exactly at end of pass 1.
        result = strategy_c_softmax_per_mel(scored, self.mel_ranked, T=1.0,
                                             per_mel_cap=30, budget=120, seed=0)
        self.assertEqual(result.n_ligands, 120)
        counts = result.selected.groupby("key_norm").size()
        for n in counts.values:
            self.assertLessEqual(n, 30)

    def test_strategy_d_takes_global_top_after_per_mel_cap(self):
        from al_benchmark_gpr91 import (prepare_scored_pool,
                                         strategy_d_global_rank_per_mel_cap)
        scored = prepare_scored_pool(self.synthons, option=1)
        result = strategy_d_global_rank_per_mel_cap(
            scored, self.mel_ranked, per_mel_cap=50, budget=100
        )
        self.assertEqual(result.n_ligands, 100)
        # Among the global top-100 by RTCNN, MEL 0 (best center -35) should
        # dominate. Worst-center MEL 3 should contribute 0.
        counts = result.selected.groupby("key_norm").size()
        self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_2", 0))
        self.assertEqual(counts.get("MEL_3", 0), 0)

    def test_ef_metric_baseline_equals_one(self):
        from al_benchmark_gpr91 import compute_ef_vs_baseline
        # Selecting the baseline against itself: EF should be 1.0 everywhere.
        synthons = self.synthons.copy()
        ef = compute_ef_vs_baseline(synthons, synthons, thresholds=[-40, -20, 0])
        # Same set selected from same set → identical hit rates → EF = 1.0.
        # (Skip the threshold that gives baseline rate 0.)
        finite = ef[np.isfinite(ef)]
        self.assertGreater(len(finite), 0)
        for v in finite:
            self.assertAlmostEqual(v, 1.0, places=4)

    def test_ef_auc_handles_all_nan(self):
        from al_benchmark_gpr91 import ef_auc
        self.assertTrue(np.isnan(ef_auc(np.array([np.nan, np.nan]))))


if __name__ == "__main__":
    unittest.main()
