"""MEL chemistry-feature pipeline for the AL sophisticated strategies.

Decodes `icm_rdmol_binary` (RDKit binary mol, stored as a stringified
bytes literal in `csv/Top1K_2Comp_MEL_Frags_With_VS_OpenVS_Mapping.csv`)
and produces a per-MEL DataFrame with ~1043 features:

- 1024-bit Morgan fingerprint (radius 2) of the MEL scaffold
- 3 physchem columns (MW, Tox_Score, molPAINS)
- 6 Stage-1 docking signals (Score, RTCNNscore, dEgrid, dEhb, dEhp, Strain)
- 3 pool descriptors (Mapped_Occupied_Synthon_Num, Nat, Nva)
- 7 placeholder columns for probe summary stats (filled in by the allocator
  at runtime; included here so the feature schema is stable)

Cached as parquet at `oracle/mel_features_<fp_kind>_<radius>_<bits>.parquet`
so repeated benchmark runs don't re-fingerprint.

Reuses RDKit's MorganGenerator (the post-2024 API) to avoid the
`GetMorganFingerprintAsBitVect` deprecation warning.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator

# Import via the project root (works through the local→NAS symlink).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "oracle"

# Stage-1 docking columns we promote into the MEL feature vector.
# Note: `Strain` is per-synthon (in the oracle CSV), NOT per-MEL — so it's
# excluded here. The MEL-ranking CSV has these decomposed-energy columns
# from the Stage-1 ICM docking run.
_STAGE1_COLS = ["Score", "RTCNNscore", "dEel", "dEgrid", "dEhb",
                "dEhp", "dEin", "dEsurf", "dTSsc", "mfScore"]
# Physchem columns (small, scale-mixed; let the model handle scaling).
_PHYSCHEM_COLS = ["MW", "Tox_Score", "molPAINS"]
# Pool descriptors.
_POOL_COLS = ["Mapped_Occupied_Synthon_Num", "Nat", "Nva"]

# Probe-summary placeholder columns the allocator fills at runtime.
PROBE_PLACEHOLDER_COLS = [
    "probe_n",
    "probe_mean",
    "probe_min",
    "probe_median",
    "probe_p10",
    "probe_stdev",
    "probe_expected_hits",
]


def _decode_mol(raw) -> Chem.Mol | None:
    """Convert a stringified bytes literal (b'\\x..') or actual bytes into
    an RDKit Mol. Returns None on failure (caller decides whether to skip
    or fall back to zero fingerprint)."""
    try:
        if isinstance(raw, str):
            if raw.startswith("b'") and raw.endswith("'"):
                raw = ast.literal_eval(raw)
            else:
                return None
        if not isinstance(raw, (bytes, bytearray)):
            return None
        mol = Chem.Mol(raw)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return mol
    except Exception:
        return None


def _fingerprint_array(mol: Chem.Mol, fp_kind: str, fp_radius: int,
                       fp_n_bits: int) -> np.ndarray:
    """Return an integer 0/1 fingerprint array. Falls back to zeros if the
    mol is None — keeps the feature matrix rectangular even when a row's
    binary fails to decode."""
    if mol is None:
        if fp_kind == "maccs":
            return np.zeros(167, dtype=np.uint8)
        return np.zeros(fp_n_bits, dtype=np.uint8)
    if fp_kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_n_bits
        )
        fp = gen.GetFingerprint(mol)
        out = np.zeros(fp_n_bits, dtype=np.uint8)
        # Bit-by-bit copy; numpy-friendly extraction:
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, out)
        return out
    if fp_kind == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
        out = np.zeros(167, dtype=np.uint8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, out)
        return out
    raise ValueError(f"unknown fp_kind {fp_kind!r}; use 'morgan' or 'maccs'")


def compute_mel_features(
    mel_ranking_csv_path: Path | str,
    fp_kind: str = "morgan",
    fp_radius: int = 2,
    fp_n_bits: int = 1024,
) -> pd.DataFrame:
    """Read the MEL ranking CSV, decode binaries, compute fingerprints, and
    return a DataFrame indexed by `key_norm` (the hyphenated ICM InChIKey).

    The returned DataFrame's columns are:
      - fp_0 .. fp_<N-1>           : N-bit fingerprint (N = fp_n_bits or 167)
      - mw / tox_score / molpains  : 3 physchem
      - score / rtcnnscore / ...   : 6 Stage-1 columns
      - mapped_occupied / nat / nva: 3 pool descriptors
      - probe_n / probe_mean / ... : 7 placeholder columns (filled at runtime)
    """
    df = pd.read_csv(mel_ranking_csv_path)
    if "icm_inchikey" not in df.columns:
        raise ValueError(
            f"{mel_ranking_csv_path} missing required column icm_inchikey"
        )
    df["key_norm"] = df["icm_inchikey"].astype(str).str.replace("_", "-")

    # Decode every row's MEL binary and compute its fingerprint.
    n_failed_decode = 0
    fps = []
    for raw in df["icm_rdmol_binary"]:
        mol = _decode_mol(raw)
        if mol is None:
            n_failed_decode += 1
        fps.append(_fingerprint_array(mol, fp_kind, fp_radius, fp_n_bits))
    fps = np.stack(fps, axis=0)
    fp_cols = [f"fp_{i}" for i in range(fps.shape[1])]
    fp_df = pd.DataFrame(fps, columns=fp_cols, index=df.index)

    # Stage-1 docking + physchem + pool descriptors (renamed lowercase
    # for the joint feature matrix).
    rename_map = {
        "MW": "mw", "Tox_Score": "tox_score", "molPAINS": "molpains",
        "Score": "score", "RTCNNscore": "rtcnnscore",
        "dEel": "deel", "dEgrid": "degrid", "dEhb": "dehb",
        "dEhp": "dehp", "dEin": "dein", "dEsurf": "desurf",
        "dTSsc": "dtssc", "mfScore": "mfscore",
        "Mapped_Occupied_Synthon_Num": "mapped_occupied",
        "Nat": "nat", "Nva": "nva",
    }
    keep_cols = _STAGE1_COLS + _PHYSCHEM_COLS + _POOL_COLS
    other = df[keep_cols].rename(columns=rename_map).copy()

    # Numeric columns: NaN-fill with column median (some rows may have
    # missing molPAINS etc.).
    for c in other.columns:
        other[c] = pd.to_numeric(other[c], errors="coerce")
        med = other[c].median()
        other[c] = other[c].fillna(med if not pd.isna(med) else 0.0)

    out = pd.concat([fp_df, other], axis=1)

    # Add zero placeholders for probe summary stats. The allocator overlays
    # these at fit time — keeping the column ORDER stable so feature
    # alignment between training and prediction is bulletproof.
    for c in PROBE_PLACEHOLDER_COLS:
        out[c] = 0.0

    out["key_norm"] = df["key_norm"].values
    out = out.set_index("key_norm")

    if n_failed_decode > 0:
        print(
            f"_mel_features: {n_failed_decode}/{len(df)} rows had un-decodable "
            f"icm_rdmol_binary; their fingerprints are zero-rows",
            file=sys.stderr,
        )
    return out


def _cache_path(fp_kind: str, fp_radius: int, fp_n_bits: int) -> Path:
    if fp_kind == "morgan":
        return CACHE_DIR / f"mel_features_morgan_r{fp_radius}_b{fp_n_bits}.parquet"
    if fp_kind == "maccs":
        return CACHE_DIR / "mel_features_maccs_b167.parquet"
    raise ValueError(f"unknown fp_kind {fp_kind!r}")


def load_or_compute(
    mel_ranking_csv_path: Path | str,
    fp_kind: str = "morgan",
    fp_radius: int = 2,
    fp_n_bits: int = 1024,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Cache wrapper around `compute_mel_features`. Reads from a parquet
    snapshot if it exists; recomputes and writes it otherwise."""
    cache = _cache_path(fp_kind, fp_radius, fp_n_bits)
    if use_cache and cache.is_file():
        return pd.read_parquet(cache)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = compute_mel_features(
        mel_ranking_csv_path, fp_kind=fp_kind,
        fp_radius=fp_radius, fp_n_bits=fp_n_bits,
    )
    out.to_parquet(cache)
    return out


__all__ = [
    "PROBE_PLACEHOLDER_COLS",
    "compute_mel_features",
    "load_or_compute",
]
