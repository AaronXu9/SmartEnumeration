"""Tests for `al_benchmark_gpr91/_mel_features.py`.

Skipped if RDKit / pandas / pyarrow aren't available in the active
Python interpreter (system /usr/bin/python3 lacks them; the
`OpenVsynthes008` mamba env has them).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


@unittest.skipUnless(_HAS_DEPS, "needs pandas+rdkit (OpenVsynthes008 env)")
class TestMelFeatures(unittest.TestCase):

    def setUp(self):
        """Build a tiny synthetic MEL ranking CSV in a temp dir."""
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "tiny_mel_ranking.csv"
        # 3 simple molecules with known InChIKeys.
        smiles_list = ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"]  # benzene, ethanol, naphthalene
        rows = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            ik = Chem.MolToInchiKey(mol)
            # Wenjin's CSV stores binaries as the *repr* of bytes:
            # b'\xef\xbe...'  (with the b prefix kept).
            binary_repr = repr(mol.ToBinary())
            rows.append({
                "icm_inchikey": ik,
                "icm_rdmol_binary": binary_repr,
                # 10 Stage-1 columns (per-MEL ICM docking energies).
                "Score": -30.0 - i,
                "RTCNNscore": -40.0 - i,
                "dEel": -5.0,
                "dEgrid": -10.0,
                "dEhb": -2.0,
                "dEhp": -3.0,
                "dEin": 2.0,
                "dEsurf": 8.0,
                "dTSsc": 1.5,
                "mfScore": -130.0,
                # 3 physchem.
                "MW": 78.0 + 10 * i,
                "Tox_Score": 0.5 - 0.1 * i,
                "molPAINS": 0.1,
                # 3 pool descriptors.
                "Mapped_Occupied_Synthon_Num": 1,
                "Nat": 6 + 4 * i,
                "Nva": 2 + i,
            })
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_compute_morgan_returns_expected_columns(self):
        from al_benchmark_gpr91._mel_features import (
            compute_mel_features, PROBE_PLACEHOLDER_COLS,
        )
        df = compute_mel_features(self.csv_path, fp_kind="morgan",
                                   fp_radius=2, fp_n_bits=1024)
        # Index = key_norm (hyphenated).
        self.assertEqual(df.index.name, "key_norm")
        # 1024 FP bits + 10 Stage-1 + 3 physchem + 3 pool + 7 probe placeholders = 1047.
        self.assertEqual(df.shape, (3, 1024 + 10 + 3 + 3 + 7))
        # FP columns are named fp_0..fp_1023.
        self.assertIn("fp_0", df.columns)
        self.assertIn("fp_1023", df.columns)
        # Probe placeholder columns are present and zero.
        for c in PROBE_PLACEHOLDER_COLS:
            self.assertIn(c, df.columns)
            self.assertTrue((df[c] == 0.0).all())

    def test_compute_maccs_returns_167_bits(self):
        from al_benchmark_gpr91._mel_features import compute_mel_features
        df = compute_mel_features(self.csv_path, fp_kind="maccs",
                                   fp_radius=2, fp_n_bits=0)
        # 167 MACCS bits + 10 Stage-1 + 3 physchem + 3 pool + 7 probe placeholders.
        self.assertEqual(df.shape[1], 167 + 10 + 3 + 3 + 7)

    def test_fingerprints_differ_across_distinct_molecules(self):
        from al_benchmark_gpr91._mel_features import compute_mel_features
        df = compute_mel_features(self.csv_path, fp_kind="morgan",
                                   fp_radius=2, fp_n_bits=1024)
        fps = df.filter(regex=r"^fp_\d+$").values
        # Pairwise Hamming distances should be > 0 between the three.
        for i in range(3):
            for j in range(i + 1, 3):
                d = (fps[i] != fps[j]).sum()
                self.assertGreater(d, 0,
                                    f"MELs {i} and {j} have identical FPs")

    def test_cache_round_trip(self):
        from al_benchmark_gpr91 import _mel_features as mf
        # Redirect the cache dir to the temp dir so we don't pollute oracle/.
        orig_cache = mf.CACHE_DIR
        try:
            mf.CACHE_DIR = Path(self.tmp.name)
            df1 = mf.load_or_compute(self.csv_path, fp_kind="morgan",
                                       fp_radius=2, fp_n_bits=1024,
                                       use_cache=True)
            cache_file = mf._cache_path("morgan", 2, 1024)
            self.assertTrue(cache_file.is_file())
            # Second call hits the cache (load_or_compute is idempotent).
            df2 = mf.load_or_compute(self.csv_path, fp_kind="morgan",
                                       fp_radius=2, fp_n_bits=1024,
                                       use_cache=True)
            pd.testing.assert_frame_equal(df1, df2)
        finally:
            mf.CACHE_DIR = orig_cache

    def test_undecodable_binary_yields_zero_fingerprint(self):
        from al_benchmark_gpr91._mel_features import compute_mel_features
        # Add one row with a deliberately broken binary.
        df = pd.read_csv(self.csv_path)
        df.loc[len(df)] = df.iloc[0].copy()
        df.loc[len(df) - 1, "icm_rdmol_binary"] = "not a valid binary"
        df.loc[len(df) - 1, "icm_inchikey"] = "BROKEN-INVALID-KEY"
        broken_csv = self.csv_path.with_name("with_broken.csv")
        df.to_csv(broken_csv, index=False)
        feats = compute_mel_features(broken_csv, fp_kind="morgan",
                                      fp_radius=2, fp_n_bits=1024)
        # Broken row should have all-zero fingerprint.
        fp_cols = [c for c in feats.columns if c.startswith("fp_")]
        broken_row = feats.loc["BROKEN-INVALID-KEY", fp_cols]
        self.assertEqual(broken_row.sum(), 0,
                          "expected all-zero fingerprint for undecodable row")


if __name__ == "__main__":
    unittest.main()
