"""Tests for Strategy K — iterative AL with model retraining."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _synthetic_universe(n_mels=4, synthons_per_mel=500, seed=0):
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


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestStrategyK(unittest.TestCase):

    def test_iterative_loop_runs(self):
        from al_benchmark_gpr91.strategy_k_iterative_al import strategy_k_iterative_al
        syn, mel_ranked = _synthetic_universe()
        # Total 2000 synthons; budget=400; n_initial=100, batch_size=100.
        result = strategy_k_iterative_al(
            syn, mel_ranked, budget=400,
            n_initial=100, batch_size=100,
            ensemble_size=2, member_n_estimators=15,
            seed=0,
        )
        self.assertEqual(result.n_ligands, 400)
        self.assertEqual(result.extras["policy"], "k_iterative")
        # 1 probe round + 3 commit rounds.
        self.assertGreaterEqual(result.extras["n_rounds"], 2)

    def test_respects_per_mel_cap(self):
        from al_benchmark_gpr91.strategy_k_iterative_al import strategy_k_iterative_al
        syn, mel_ranked = _synthetic_universe()
        result = strategy_k_iterative_al(
            syn, mel_ranked, budget=400,
            n_initial=100, batch_size=100,
            per_mel_cap=200, ensemble_size=2, member_n_estimators=10,
            seed=0,
        )
        counts = result.selected.groupby("key_norm").size()
        for n in counts.values:
            self.assertLessEqual(n, 200)

    def test_seed_deterministic(self):
        from al_benchmark_gpr91.strategy_k_iterative_al import strategy_k_iterative_al
        syn, mel_ranked = _synthetic_universe()
        a = strategy_k_iterative_al(
            syn, mel_ranked, budget=300,
            n_initial=80, batch_size=80,
            ensemble_size=2, member_n_estimators=10,
            seed=42,
        )
        b = strategy_k_iterative_al(
            syn, mel_ranked, budget=300,
            n_initial=80, batch_size=80,
            ensemble_size=2, member_n_estimators=10,
            seed=42,
        )
        self.assertEqual(len(a.selected), len(b.selected))
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"]),
        )

    def test_prefers_better_mels(self):
        from al_benchmark_gpr91.strategy_k_iterative_al import strategy_k_iterative_al
        syn, mel_ranked = _synthetic_universe()
        # No per-MEL cap so the better MEL should dominate.
        result = strategy_k_iterative_al(
            syn, mel_ranked, budget=400,
            n_initial=100, batch_size=100,
            per_mel_cap=10_000,
            ensemble_size=2, member_n_estimators=15,
            seed=0,
        )
        counts = result.selected.groupby("key_norm").size()
        self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_3", 0))


if __name__ == "__main__":
    unittest.main()
