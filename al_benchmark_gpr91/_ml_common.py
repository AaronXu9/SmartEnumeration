"""Shared ML scaffolding for the AL sophisticated strategies (J/K/L/N).

Three pieces:

- `synthon_features(df)` — per-row numeric features from the oracle/
  scored_df: RTCNN_Score, Strain, CoreRmsd, MolLogP, MolLogS, MoldHf,
  MolPSA, MolVolume, SubstScore. 9 dims.
- `joint_features(scored_df, mel_features_df, mel_ranked)` — joint
  per-row matrix combining synthon features + per-MEL features
  (from `_mel_features.compute_mel_features`). The output rows
  align 1:1 with `scored_df` rows.
- `_BaggedRegressor` — ensemble of GradientBoostingRegressors fit on
  bagged subsamples. `predict_with_std` returns (mean, std) for
  UCB-style acquisition functions.

Reused by K (iterative AL), L (multi-fidelity), N (joint UCB).
J uses `synthon_features` + a flat per-MEL feature lookup; see its
own module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


# 9 per-synthon numeric columns (present in `csv/all_mels_combined_core.csv`).
SYNTHON_NUMERIC_COLS = [
    "RTCNN_Score",
    "Strain",
    "CoreRmsd",
    "MolLogP",
    "MolLogS",
    "MoldHf",
    "MolPSA",
    "MolVolume",
    "SubstScore",
]


def synthon_features(df: pd.DataFrame) -> np.ndarray:
    """Extract the 9 numeric synthon features from a scored DataFrame.

    Missing values are filled with the column median (computed across the
    passed-in df). Infinite values are clipped to ±float32-max-stably-
    representable (1e30) so the downstream sklearn validation passes.
    Wenjin's oracle has some `inf`/`-inf` in `MoldHf` for synthons that
    failed to compute heat of formation — about 0.0006% of rows."""
    X = df.reindex(columns=SYNTHON_NUMERIC_COLS).copy()
    for c in SYNTHON_NUMERIC_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        # Replace ±inf with NaN, then fill NaN with the column median.
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        med = X[c].median()
        X[c] = X[c].fillna(med if not pd.isna(med) else 0.0)
    arr = X.values.astype(np.float32)
    # Final safety net: any value that still overflows float32 gets
    # clipped. (median of a column with all-finite values can't itself
    # overflow, but be defensive.)
    return np.clip(arr, -1e30, 1e30)


def joint_features(
    scored_df: pd.DataFrame,
    mel_features_df: pd.DataFrame,
    mel_ranked: pd.DataFrame | None = None,
) -> np.ndarray:
    """Concatenate per-row synthon features with the row's MEL features
    (looked up by `key_norm`).

    `scored_df` must have a `key_norm` column matching `mel_features_df`'s
    index. Rows whose `key_norm` is missing from `mel_features_df` get
    zero MEL features (consistent with the un-decodable-binary fallback
    in `_mel_features._fingerprint_array`).

    Returns float32 ndarray of shape (len(scored_df), 9 + n_mel_features).
    For 10M-row scored_df, this is the hot path — does a single pandas
    merge (vectorized), no Python loop.
    """
    if "key_norm" not in scored_df.columns:
        raise KeyError("scored_df missing required 'key_norm' column")

    # Per-row synthon features (always cheap-ish).
    syn = synthon_features(scored_df)

    # MEL features: vectorized lookup via pandas merge.
    mel_cols = mel_features_df.columns
    mel_df = mel_features_df.reset_index()  # key_norm becomes a regular column
    # Use a left merge so all scored_df rows get a row (NaN for unknown MELs).
    merged = scored_df[["key_norm"]].merge(mel_df, on="key_norm", how="left")
    # Fill NaN (unknown MELs OR pre-existing NaN in features) with 0,
    # replace ±inf with 0, and clip to float32-stable range.
    mel_block = merged[mel_cols].replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0.0).values.astype(np.float32)
    mel_block = np.clip(mel_block, -1e30, 1e30)

    return np.hstack([syn, mel_block])


def extract_probe_observations(
    probes_by_key: dict[str, pd.DataFrame],
    mel_features_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for training from the probe phase output.

    `probes_by_key`: as returned by `_probe_each_mel` in al_ext_strategies.py
    — dict {key_norm: DataFrame of probed synthons with `_score` and
    `FullLigand_Score` columns}.

    If `mel_features_df` is None, uses synthon-only features (9 dims).
    Otherwise uses joint features (9 + n_mel dims).

    `y` is FullLigand_Score per probed synthon. **Rows with NaN
    `FullLigand_Score` are dropped** — sklearn regressors reject NaN
    targets, and Wenjin's oracle has ~5% NaN rate where Stage-5 docking
    didn't converge for that ligand.
    """
    parts = []
    ys = []
    for key, probe in probes_by_key.items():
        if probe is None or len(probe) == 0:
            continue
        sub = probe.copy()
        # Ensure the probe rows have the lookup key column.
        if "key_norm" not in sub.columns:
            sub["key_norm"] = key
        parts.append(sub)
        if "FullLigand_Score" in sub.columns:
            ys.append(sub["FullLigand_Score"].astype(np.float32).values)
        else:
            # Fall back to _score (e.g., during tests with synthetic data
            # where FullLigand_Score isn't separately stored).
            ys.append(sub["_score"].astype(np.float32).values)
    if not parts:
        n_mel_dim = mel_features_df.shape[1] if mel_features_df is not None else 0
        return (
            np.zeros((0, len(SYNTHON_NUMERIC_COLS) + n_mel_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    big = pd.concat(parts, ignore_index=True)
    y = np.concatenate(ys).astype(np.float32)
    if mel_features_df is None:
        X = synthon_features(big)
    else:
        X = joint_features(big, mel_features_df)
    # Drop rows where the target is NaN (or infinite) — sklearn refuses
    # to train on those, and the project's Stage-5 oracle has some
    # missing FullLigand_Score values where docking didn't converge.
    finite = np.isfinite(y)
    if not finite.all():
        X = X[finite]
        y = y[finite]
    return X, y


class BaggedRegressor:
    """Bagged GradientBoostingRegressor ensemble for uncertainty estimates.

    `fit(X, y)` trains `n_estimators` independent regressors on bootstrap
    samples of (X, y). `predict_with_std(X)` returns (mu, sigma) where
    sigma is the across-ensemble std. UCB-style acquisition functions
    consume both.

    Hyperparameters: each member is a small GBR (n_estimators=50,
    max_depth=2) — keeps total fit time ≲ 30s on ~95K × ~1k inputs.
    """

    def __init__(
        self,
        n_bags: int = 5,
        member_n_estimators: int = 50,
        member_max_depth: int = 2,
        learning_rate: float = 0.1,
        subsample: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.n_bags = n_bags
        self.member_n_estimators = member_n_estimators
        self.member_max_depth = member_max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.seed = seed
        self.models_: list[GradientBoostingRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedRegressor":
        if len(X) == 0:
            raise ValueError("BaggedRegressor.fit got empty X")
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.models_ = []
        for i in range(self.n_bags):
            # Bootstrap sample: WITH replacement, size n.
            idx = rng.choice(n, size=n, replace=True)
            model = GradientBoostingRegressor(
                n_estimators=self.member_n_estimators,
                max_depth=self.member_max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.seed + i,
            )
            model.fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu, sigma): per-row mean and across-ensemble std."""
        if not self.models_:
            raise RuntimeError("BaggedRegressor.fit must be called before predict")
        # Stack predictions: shape (n_bags, n_rows).
        preds = np.stack([m.predict(X) for m in self.models_], axis=0)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0, ddof=0)
        return mu.astype(np.float32), sigma.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        mu, _ = self.predict_with_std(X)
        return mu


__all__ = [
    "BaggedRegressor",
    "SYNTHON_NUMERIC_COLS",
    "extract_probe_observations",
    "joint_features",
    "synthon_features",
]
