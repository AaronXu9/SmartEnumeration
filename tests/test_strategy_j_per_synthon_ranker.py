"""Tests for Strategy J — per-synthon learned ranker."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
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


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestStrategyJ(unittest.TestCase):

    def test_baseline_alloc_runs(self):
        from al_benchmark_gpr91.strategy_j_per_synthon_ranker import (
            strategy_j_synthon_ranker_baseline_alloc,
        )
        syn, mel_ranked = _synthetic_universe()
        result = strategy_j_synthon_ranker_baseline_alloc(
            syn, mel_ranked, budget=400, n_probe=20, seed=0, min_commit=0,
        )
        self.assertGreater(result.n_ligands, 350)
        self.assertLessEqual(result.n_ligands, 405)
        self.assertEqual(result.extras["policy"], "baseline")
        # With enough probe data, the model should train.
        self.assertEqual(result.extras["picker"], "learned")

    def test_ucb_alloc_runs(self):
        from al_benchmark_gpr91.strategy_j_per_synthon_ranker import (
            strategy_j_synthon_ranker_ucb_alloc,
        )
        syn, mel_ranked = _synthetic_universe()
        result = strategy_j_synthon_ranker_ucb_alloc(
            syn, mel_ranked, budget=400, n_probe=20, seed=0, min_commit=0,
        )
        self.assertGreater(result.n_ligands, 350)
        self.assertLessEqual(result.n_ligands, 405)
        self.assertEqual(result.extras["policy"], "ucb")
        self.assertEqual(result.extras["picker"], "learned")

    def test_seed_deterministic(self):
        from al_benchmark_gpr91.strategy_j_per_synthon_ranker import (
            strategy_j_synthon_ranker_ucb_alloc,
        )
        syn, mel_ranked = _synthetic_universe()
        a = strategy_j_synthon_ranker_ucb_alloc(
            syn, mel_ranked, budget=400, n_probe=20, seed=42, min_commit=0,
        )
        b = strategy_j_synthon_ranker_ucb_alloc(
            syn, mel_ranked, budget=400, n_probe=20, seed=42, min_commit=0,
        )
        self.assertEqual(len(a.selected), len(b.selected))
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"]),
        )

    def test_too_few_probes_falls_back_to_softmax(self):
        """If probe size is tiny (< 20 rows total) the trainer returns
        None and the picker falls back to softmax sampling. The result
        is still valid; just the `picker` extras field reports
        'softmax_fallback'."""
        from al_benchmark_gpr91.strategy_j_per_synthon_ranker import (
            strategy_j_synthon_ranker_ucb_alloc,
        )
        syn, mel_ranked = _synthetic_universe(n_mels=2, synthons_per_mel=50)
        # n_probe=2, 2 MELs → 4 total observations < 20 → fallback.
        result = strategy_j_synthon_ranker_ucb_alloc(
            syn, mel_ranked, budget=80, n_probe=2, seed=0, min_commit=0,
        )
        self.assertEqual(result.extras["picker"], "softmax_fallback")

    def test_prefers_better_mels(self):
        from al_benchmark_gpr91.strategy_j_per_synthon_ranker import (
            strategy_j_synthon_ranker_ucb_alloc,
        )
        syn, mel_ranked = _synthetic_universe()
        result = strategy_j_synthon_ranker_ucb_alloc(
            syn, mel_ranked, budget=600, n_probe=20, seed=0, min_commit=0,
        )
        counts = result.selected.groupby("key_norm").size()
        # MEL 0 (center -35) should get more budget than MEL 3 (center -5).
        self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_3", 0))


if __name__ == "__main__":
    unittest.main()
