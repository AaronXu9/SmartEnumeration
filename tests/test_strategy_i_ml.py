"""Tests for Strategy I — chemistry-aware ML allocator + softmax picker."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    from rdkit import Chem  # noqa: F401
    import sklearn  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _synthetic_universe(n_mels=4, synthons_per_mel=200, seed=0):
    """Synthon DataFrame + MEL ranking compatible with the al_benchmark_gpr91
    strategy signatures."""
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
                "FullLigand_Score": rt + rng.normal(0, 2),
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


def _synthetic_mel_features(mel_keys, seed=0):
    """Per-MEL chemistry features stand-in (random for unit-test purposes).
    Indexed by key_norm to match the real `_mel_features.compute_mel_features`
    output shape."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(mel_keys), 32)),
        index=pd.Index(mel_keys, name="key_norm"),
        columns=[f"fp_{k}" for k in range(32)],
    )


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn+rdkit (OpenVsynthes008 env)")
class TestStrategyI(unittest.TestCase):

    def test_strategy_i_runs_with_chemistry_features(self):
        from al_benchmark_gpr91.strategy_i_ml_alloc import (
            strategy_i_ml_alloc_softmax_pick,
        )
        syn, mel_ranked = _synthetic_universe()
        mel_keys = sorted(syn["key_norm"].unique())
        mel_features = _synthetic_mel_features(mel_keys)
        result = strategy_i_ml_alloc_softmax_pick(
            syn, mel_ranked, budget=400, n_probe=20,
            mel_features_df=mel_features, seed=0, min_commit=0,
        )
        # Selection should approximately match budget.
        self.assertGreaterEqual(result.n_ligands, 350)
        self.assertLessEqual(result.n_ligands, 405)
        self.assertEqual(result.extras["policy"], "ml")
        self.assertEqual(result.extras["n_probe_total"], 80)

    def test_strategy_i_runs_without_chemistry_features_fallback(self):
        """If mel_features_df is None, Strategy I should still work
        (falls back to V1 features-only mode)."""
        from al_benchmark_gpr91.strategy_i_ml_alloc import (
            strategy_i_ml_alloc_softmax_pick,
        )
        syn, mel_ranked = _synthetic_universe()
        result = strategy_i_ml_alloc_softmax_pick(
            syn, mel_ranked, budget=400, n_probe=20,
            mel_features_df=None, seed=0, min_commit=0,
        )
        self.assertGreater(result.n_ligands, 0)
        self.assertEqual(result.extras["policy"], "ml")

    def test_strategy_i_seed_deterministic(self):
        from al_benchmark_gpr91.strategy_i_ml_alloc import (
            strategy_i_ml_alloc_softmax_pick,
        )
        syn, mel_ranked = _synthetic_universe()
        mel_keys = sorted(syn["key_norm"].unique())
        mel_features = _synthetic_mel_features(mel_keys)
        a = strategy_i_ml_alloc_softmax_pick(
            syn, mel_ranked, budget=400, n_probe=20,
            mel_features_df=mel_features, seed=42, min_commit=0,
        )
        b = strategy_i_ml_alloc_softmax_pick(
            syn, mel_ranked, budget=400, n_probe=20,
            mel_features_df=mel_features, seed=42, min_commit=0,
        )
        self.assertEqual(len(a.selected), len(b.selected))
        self.assertEqual(
            sorted(a.selected["synthon_inchikey"]),
            sorted(b.selected["synthon_inchikey"]),
        )

    def test_strategy_i_prefers_better_mels_with_signal(self):
        """With informative MEL features (correlated with the MEL's score
        center), Strategy I should put more budget on the best MEL than
        on the worst."""
        from al_benchmark_gpr91.strategy_i_ml_alloc import (
            strategy_i_ml_alloc_softmax_pick,
        )
        syn, mel_ranked = _synthetic_universe()
        # Construct mel_features such that fp_0 = -mel_rank (informative).
        mel_features = pd.DataFrame(
            {"fp_0": [-1.0, -2.0, -3.0, -4.0], "fp_1": [0.0, 0.0, 0.0, 0.0]},
            index=pd.Index([f"MEL_{i}" for i in range(4)], name="key_norm"),
        )
        result = strategy_i_ml_alloc_softmax_pick(
            syn, mel_ranked, budget=600, n_probe=20,
            mel_features_df=mel_features, seed=0, min_commit=0,
        )
        counts = result.selected.groupby("key_norm").size()
        # MEL 0 (best, center -35) should get more than MEL 3 (worst).
        self.assertGreater(counts.get("MEL_0", 0), counts.get("MEL_3", 0))


if __name__ == "__main__":
    unittest.main()
