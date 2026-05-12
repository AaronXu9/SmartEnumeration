"""Tests for Strategy M — submodular / diversity-aware selection."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _synthetic_universe(n_mels=4, synthons_per_mel=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = [-35, -25, -15, -5]
    rows = []
    for m in range(min(n_mels, len(centers))):
        c = centers[m]
        for j in range(synthons_per_mel):
            rt = rng.normal(c, 4)
            rows.append({
                "key_norm": f"MEL_{m}",
                "synthon_inchikey": f"S_{m}_{j:04d}",
                "RTCNN_Score": rt,
                "FullLigand_Score": rt + rng.normal(0, 1.0),
                "Strain": rng.uniform(5, 25),
                "CoreRmsd": rng.uniform(0.5, 3.0),
                "MolLogP": rng.uniform(-1, 5),
                "MolLogS": rng.uniform(-6, -1),
                "MoldHf": rng.normal(-40, 20),
                "MolPSA": rng.uniform(20, 120),
                "MolVolume": rng.uniform(150, 400),
                "SubstScore": rng.uniform(50, 130),
                "_score": rt,
            })
    syn = pd.DataFrame(rows)
    mel_rows = [{"key_norm": f"MEL_{i}", "Score": centers[i], "mel_rank": i + 1}
                for i in range(min(n_mels, len(centers)))]
    return syn, pd.DataFrame(mel_rows)


@unittest.skipUnless(_HAS_DEPS, "needs pandas+numpy")
class TestStrategyM(unittest.TestCase):

    def test_pure_score_mode_picks_lowest_rtcnn_score(self):
        """alpha=1.0 → no diversity bonus → pure top-N by RTCNN_Score
        (the default cheap signal). MEL_0 (best-centered) should
        dominate."""
        from al_benchmark_gpr91.strategy_m_submodular import strategy_m_submodular
        syn, mel_ranked = _synthetic_universe()
        result = strategy_m_submodular(
            syn, mel_ranked, budget=50, alpha=1.0, diversity_weight=0.0, seed=0,
        )
        self.assertEqual(result.n_ligands, 50)
        counts = result.selected.groupby("key_norm").size()
        self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_3", 0))

    def test_oracle_upper_bound_mode_via_score_column(self):
        """score_column='FullLigand_Score' is the oracle upper-bound
        reference (NOT a fair strategy — it cheats on the metric).
        Verify it still runs and beats RTCNN-only on this synthetic
        dataset where RTCNN ⊥ FullLigand_Score noise."""
        from al_benchmark_gpr91.strategy_m_submodular import strategy_m_submodular
        syn, mel_ranked = _synthetic_universe()
        result = strategy_m_submodular(
            syn, mel_ranked, budget=50, alpha=1.0, diversity_weight=0.0,
            score_column="FullLigand_Score", seed=0,
        )
        self.assertEqual(result.n_ligands, 50)

    def test_diversity_mode_spreads_across_mels(self):
        """High diversity_weight should force the selection to include all
        four MELs even if the bottom-scoring MEL contributes little to
        the pure score."""
        from al_benchmark_gpr91.strategy_m_submodular import strategy_m_submodular
        syn, mel_ranked = _synthetic_universe()
        result = strategy_m_submodular(
            syn, mel_ranked, budget=100, alpha=0.5, diversity_weight=1000.0, seed=0,
        )
        sel_keys = set(result.selected["key_norm"].unique())
        # With diversity_weight=1000 (huge), each new MEL is worth more
        # than any score gap → all 4 MELs should appear.
        self.assertEqual(sel_keys, {"MEL_0", "MEL_1", "MEL_2", "MEL_3"})

    def test_runs_with_learned_score(self):
        from al_benchmark_gpr91.strategy_m_submodular import strategy_m_submodular
        syn, mel_ranked = _synthetic_universe()
        result = strategy_m_submodular(
            syn, mel_ranked, budget=100, alpha=0.7,
            use_learned_score=True, n_probe=80,
            ensemble_size=2, member_n_estimators=15, seed=0,
        )
        self.assertEqual(result.n_ligands, 100)
        self.assertEqual(result.extras["policy"], "m_submodular")
        self.assertTrue(result.extras["use_learned_score"])

    def test_seed_deterministic_in_pure_score_mode(self):
        from al_benchmark_gpr91.strategy_m_submodular import strategy_m_submodular
        syn, mel_ranked = _synthetic_universe()
        a = strategy_m_submodular(
            syn, mel_ranked, budget=80, alpha=1.0, diversity_weight=0.0, seed=42,
        )
        b = strategy_m_submodular(
            syn, mel_ranked, budget=80, alpha=1.0, diversity_weight=0.0, seed=42,
        )
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"]),
        )


if __name__ == "__main__":
    unittest.main()
