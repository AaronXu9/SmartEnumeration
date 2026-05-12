"""Tests for `al_benchmark_gpr91/_ml_common.py`."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _make_synthetic_scored(n_per_mel=20, n_mels=3, seed=0):
    """A scored_df mock with the columns _ml_common.synthon_features expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mels):
        for j in range(n_per_mel):
            rt = rng.normal(-25 - m * 5, 4)
            rows.append({
                "key_norm": f"MEL_{m}",
                "synthon_inchikey": f"S_{m}_{j:03d}",
                "RTCNN_Score": rt,
                "FullLigand_Score": rt + rng.normal(0, 2),
                "Strain": rng.uniform(2, 20),
                "CoreRmsd": rng.uniform(0.3, 2.5),
                "MolLogP": rng.uniform(-1, 5),
                "MolLogS": rng.uniform(-7, -1),
                "MoldHf": rng.normal(-40, 20),
                "MolPSA": rng.uniform(20, 120),
                "MolVolume": rng.uniform(150, 400),
                "SubstScore": rng.uniform(50, 130),
                "_score": rt,
            })
    return pd.DataFrame(rows)


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestSynthonFeatures(unittest.TestCase):

    def test_shape_and_no_nans(self):
        from al_benchmark_gpr91._ml_common import synthon_features, SYNTHON_NUMERIC_COLS
        df = _make_synthetic_scored()
        X = synthon_features(df)
        self.assertEqual(X.shape, (len(df), len(SYNTHON_NUMERIC_COLS)))
        self.assertFalse(np.isnan(X).any())

    def test_missing_column_filled_with_median(self):
        from al_benchmark_gpr91._ml_common import synthon_features
        df = _make_synthetic_scored().drop(columns=["MolLogS"])
        X = synthon_features(df)
        # Column reindex inserted NaN, then filled with median (= 0 here
        # since the column was all-missing → median is NaN → fallback 0).
        col_idx = 4  # MolLogS is column index 4 in SYNTHON_NUMERIC_COLS
        self.assertTrue(np.allclose(X[:, col_idx], 0.0))


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestJointFeatures(unittest.TestCase):

    def _mel_features(self, n_mels=3, n_mel_feat=5):
        return pd.DataFrame(
            np.arange(n_mels * n_mel_feat).reshape(n_mels, n_mel_feat).astype(float),
            index=pd.Index([f"MEL_{m}" for m in range(n_mels)], name="key_norm"),
            columns=[f"mf_{k}" for k in range(n_mel_feat)],
        )

    def test_joint_concat_shape(self):
        from al_benchmark_gpr91._ml_common import joint_features
        scored = _make_synthetic_scored()
        mel_feats = self._mel_features()
        X = joint_features(scored, mel_feats)
        # 9 synthon features + 5 mel features.
        self.assertEqual(X.shape, (len(scored), 9 + 5))

    def test_joint_lookup_aligns_per_row(self):
        from al_benchmark_gpr91._ml_common import joint_features
        scored = _make_synthetic_scored()
        mel_feats = self._mel_features(n_mels=3, n_mel_feat=2)
        X = joint_features(scored, mel_feats)
        # The mel block should match the row's MEL identity. Per-row check:
        # MEL_0 → mel features [0, 1]; MEL_1 → [2, 3]; MEL_2 → [4, 5].
        expected = {"MEL_0": [0.0, 1.0], "MEL_1": [2.0, 3.0], "MEL_2": [4.0, 5.0]}
        for i, k in enumerate(scored["key_norm"]):
            self.assertTrue(np.allclose(X[i, 9:11], expected[k]),
                              f"row {i}: MEL={k} got {X[i, 9:11]}")

    def test_unknown_mel_row_gets_zero_features(self):
        from al_benchmark_gpr91._ml_common import joint_features
        scored = _make_synthetic_scored()
        # Inject a row with a MEL key that's NOT in mel_features_df.
        new = scored.iloc[0].copy()
        new["key_norm"] = "MEL_UNKNOWN"
        scored2 = pd.concat([scored, new.to_frame().T], ignore_index=True)
        mel_feats = self._mel_features(n_mels=3, n_mel_feat=4)
        X = joint_features(scored2, mel_feats)
        # Last row's MEL block should be all zeros.
        self.assertTrue(np.allclose(X[-1, 9:], 0.0))


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestBaggedRegressor(unittest.TestCase):

    def test_fit_predict_shapes(self):
        from al_benchmark_gpr91._ml_common import BaggedRegressor
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 8)).astype(np.float32)
        # Make the target a noisy linear combo so the ensemble can learn.
        w = rng.normal(size=8)
        y = (X @ w + rng.normal(scale=0.2, size=200)).astype(np.float32)
        bag = BaggedRegressor(n_bags=3, member_n_estimators=20, seed=42)
        bag.fit(X, y)
        mu, sigma = bag.predict_with_std(X)
        self.assertEqual(mu.shape, (200,))
        self.assertEqual(sigma.shape, (200,))
        # The ensemble should have non-trivial uncertainty somewhere.
        self.assertGreater(sigma.max(), 0.0)
        # Mean prediction should correlate reasonably with the true y.
        self.assertGreater(np.corrcoef(mu, y)[0, 1], 0.5)

    def test_predict_alias(self):
        from al_benchmark_gpr91._ml_common import BaggedRegressor
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 4)).astype(np.float32)
        y = rng.normal(size=80).astype(np.float32)
        bag = BaggedRegressor(n_bags=2, member_n_estimators=10, seed=0).fit(X, y)
        self.assertTrue(np.array_equal(bag.predict(X), bag.predict_with_std(X)[0]))

    def test_seed_determinism(self):
        from al_benchmark_gpr91._ml_common import BaggedRegressor
        rng = np.random.default_rng(7)
        X = rng.normal(size=(150, 5)).astype(np.float32)
        y = rng.normal(size=150).astype(np.float32)
        a = BaggedRegressor(n_bags=3, member_n_estimators=15, seed=99).fit(X, y).predict(X)
        b = BaggedRegressor(n_bags=3, member_n_estimators=15, seed=99).fit(X, y).predict(X)
        self.assertTrue(np.allclose(a, b))


@unittest.skipUnless(_HAS_DEPS, "needs pandas+sklearn (OpenVsynthes008 env)")
class TestExtractProbeObservations(unittest.TestCase):

    def test_concatenates_per_mel_probes(self):
        from al_benchmark_gpr91._ml_common import extract_probe_observations
        rng = np.random.default_rng(0)
        probes = {}
        for m in range(3):
            df = _make_synthetic_scored(n_per_mel=10, n_mels=1, seed=m)
            df["key_norm"] = f"MEL_{m}"
            probes[f"MEL_{m}"] = df
        X, y = extract_probe_observations(probes, mel_features_df=None)
        # 30 rows, 9 synthon features.
        self.assertEqual(X.shape, (30, 9))
        self.assertEqual(y.shape, (30,))

    def test_empty_probes_returns_empty_arrays(self):
        from al_benchmark_gpr91._ml_common import extract_probe_observations
        X, y = extract_probe_observations({}, mel_features_df=None)
        self.assertEqual(X.shape, (0, 9))
        self.assertEqual(y.shape, (0,))


if __name__ == "__main__":
    unittest.main()
