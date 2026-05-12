#!/usr/bin/env python3
"""Build an offline SRG score oracle from existing batch results.

Walks `results_*/**/enumerated.sdf` directly (the per-batch
`batch_manifest.tsv` files turn out to be sparse one-row dry-run
snapshots, not a complete index — see comment in `_find_enumerated_sdfs`).
For each SDF, infers the MEL rank from the parent dir name
(`MEL_<row>_Rank<N>`) and looks up the MEL ICMInChIKey via the
per-MEL surviving-synthon SDF filename in `compatible_syntons/`
(pattern: `Rank<N>_ICMInChiKey_<KEY_underscored>_OpenVSInChiKey_...`).

Writes:

  oracle/srg_scores.csv   one row per (mel_inchikey, synthon_inchikey)
  oracle/coverage.tsv     per-MEL coverage summary

Deduplicates (mel_inchikey, synthon_inchikey) keeping the most-negative
rtcnn_score across all observed runs.

Pure stdlib SDF parsing, no RDKit. Mirrors the style of edit_mel_cap.py.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# Import via the project root (works under the local→NAS symlink).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROJECT_ROOT, SYNTHONS_DIR  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / "oracle"

# Filename of a per-MEL surviving-synthon SDF:
#   Rank<N>_ICMInChiKey_<KEY_underscored>_OpenVSInChiKey_<HASH>_..._APO.sdf
_SYNTHON_RE = re.compile(
    r"^Rank(?P<rank>\d+)_ICMInChiKey_(?P<icm_ik>[A-Z0-9_]+)_OpenVSInChiKey_"
)
# Parent dir of an enumerated.sdf:
#   MEL_<row>_Rank<N>                      (canonical)
#   MEL_<row>_Rank<N>_<suffix>             (e.g., MEL_2_Rank2_committed)
#   MEL_<row>_combiDock_R1                 (legacy debug fixture — skip; no MEL InChIKey)
_OUT_DIR_RE = re.compile(r"^MEL_(?P<row>\d+)_Rank(?P<rank>\d+)(?:_[A-Za-z0-9]+)?$")

# SD tags we extract per record. Order is the CSV column order (after the
# MEL identifier columns).
SCORE_TAGS = (
    "RTCNN_Score",
    "DockScore",
    "Strain",
    "CoreRmsd",
    "SubstScore",
    "VlsScore",
)
# Identifier tags. InchiKey is the synthon InChIKey (hyphenated already);
# full_synthon_id is Enamine's short ID (e.g. "s21771012").
ID_TAGS = ("InchiKey", "full_synthon_id")


def _normalize_mel_inchikey(raw: str) -> str:
    """Manifest stores MEL InChIKey as 'ABCD_EFGH_N' (ICM-style underscored).
    Convert to standard 'ABCD-EFGH-N'. Two underscores become two hyphens;
    no other transformation."""
    return raw.replace("_", "-")


def _iter_sdf_records(sdf_path: Path):
    """Yield dict-of-tag-values for each SDF record in the file.

    Only extracts the SCORE_TAGS and ID_TAGS we care about. Skips records
    missing critical tags. Pure-string parsing — no RDKit."""
    want = set(SCORE_TAGS) | set(ID_TAGS)
    cur: dict[str, str] = {}
    tag: str | None = None
    with sdf_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "$$$$":
                if cur:
                    yield cur
                cur = {}
                tag = None
                continue
            if line.startswith("> <") and line.endswith(">"):
                # Next line is the value.
                t = line[3:-1]
                tag = t if t in want else None
                continue
            if tag is not None:
                # SD-tag values can span multiple lines until the next blank
                # line. We take only the first line — sufficient for the
                # scalar fields we want.
                if line == "":
                    tag = None
                else:
                    cur[tag] = line
                    tag = None
        if cur:
            yield cur


def _find_enumerated_sdfs() -> list[Path]:
    """All enumerated.sdf files under any results_* subtree of the project.

    We don't use batch_manifest.tsv as the index because most manifests are
    sparse one-row dry-run snapshots from `--only-row N` invocations — they
    do not list every MEL whose enumerated.sdf actually exists."""
    roots = [p for p in PROJECT_ROOT.iterdir() if p.is_dir() and p.name.startswith("results")]
    out: list[Path] = []
    for r in roots:
        out.extend(sorted(r.rglob("enumerated.sdf")))
    return out


def _build_rank_to_inchikey_map() -> dict[str, str]:
    """Look up MEL ICMInChIKey (hyphen-normalized) by rank string ("Rank2", "Rank10").

    Reads filenames in `compatible_syntons/` under the active target
    (debug or production — see paths.py for which). Combines both the
    debug and production dirs so the oracle covers any MEL whose
    synthon file is on the NAS."""
    rank_map: dict[str, str] = {}
    candidates: list[Path] = []
    # Active target's synthon dir.
    if SYNTHONS_DIR.is_dir():
        candidates.append(SYNTHONS_DIR)
    # Also peek at the production dir if we're in debug mode (and vice versa).
    for sib_name in ("CB2_5ZTY", "CB2_5ZTY_debug"):
        sib = PROJECT_ROOT / sib_name / "compatible_syntons"
        if sib.is_dir() and sib not in candidates:
            candidates.append(sib)
    for d in candidates:
        for f in d.iterdir():
            m = _SYNTHON_RE.match(f.name)
            if not m:
                continue
            rank_key = "Rank" + m.group("rank")
            icm_ik_underscored = m.group("icm_ik")
            # ICM uses ABC_DEFG_N (underscores); standard form is ABC-DEFG-N.
            hyphenated = icm_ik_underscored.replace("_", "-")
            # Don't clobber: if multiple sources, keep the first.
            rank_map.setdefault(rank_key, hyphenated)
    return rank_map


def main() -> int:
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)

    rank_map = _build_rank_to_inchikey_map()
    print(f"rank → MEL InChIKey map: {len(rank_map)} entries", file=sys.stderr)
    if not rank_map:
        print(
            "no compatible_syntons SDFs found — can't infer MEL InChIKey from rank",
            file=sys.stderr,
        )
        return 1

    sdfs = _find_enumerated_sdfs()
    if not sdfs:
        print("no enumerated.sdf found under results_*/", file=sys.stderr)
        return 1
    print(f"scanning {len(sdfs)} enumerated.sdf files", file=sys.stderr)

    # (mel_inchikey, synthon_inchikey) -> best row dict
    best: dict[tuple[str, str], dict] = {}
    # mel_inchikey -> set of synthon_inchikeys observed
    coverage: dict[str, dict[str, set]] = defaultdict(
        lambda: {"synthons": set(), "ranks": set(), "rows": set(), "sources": set()}
    )
    n_records_total = 0
    n_records_with_inchi = 0
    n_records_kept = 0
    n_sdfs_seen = 0
    n_sdfs_skipped = 0

    for sdf in sdfs:
        parent_name = sdf.parent.name
        m = _OUT_DIR_RE.match(parent_name)
        if not m:
            # e.g., MEL_2_combiDock_R1 — debug fixture without a Rank → InChIKey
            # mapping; skip rather than guess.
            n_sdfs_skipped += 1
            continue
        rank = "Rank" + m.group("rank")
        mel_row = m.group("row")
        mel_inchikey = rank_map.get(rank)
        if not mel_inchikey:
            n_sdfs_skipped += 1
            continue
        n_sdfs_seen += 1
        src_label = str(sdf.relative_to(PROJECT_ROOT))
        for rec in _iter_sdf_records(sdf):
            n_records_total += 1
            synth_ik = rec.get("InchiKey", "").strip()
            rt = rec.get("RTCNN_Score", "").strip()
            if not synth_ik or not rt:
                continue
            try:
                rt_f = float(rt)
            except ValueError:
                continue
            n_records_with_inchi += 1
            key = (mel_inchikey, synth_ik)
            cur = best.get(key)
            if cur is None or rt_f < cur["_rt_f"]:
                cov = coverage[mel_inchikey]
                cov["synthons"].add(synth_ik)
                cov["ranks"].add(rank)
                cov["rows"].add(mel_row)
                cov["sources"].add(src_label)
                record = {
                    "mel_inchikey": mel_inchikey,
                    "synthon_inchikey": synth_ik,
                    "rtcnn_score": rt_f,
                    "mel_rank": rank,
                    "mel_row": mel_row,
                    "full_synthon_id": rec.get("full_synthon_id", "").strip(),
                    "source_manifest": src_label,
                    "_rt_f": rt_f,
                }
                for tag in SCORE_TAGS:
                    if tag == "RTCNN_Score":
                        continue
                    v = rec.get(tag, "").strip()
                    try:
                        record[tag.lower()] = float(v) if v else ""
                    except ValueError:
                        record[tag.lower()] = ""
                best[key] = record
                if cur is None:
                    n_records_kept += 1

    # Write scores CSV.
    csv_path = ORACLE_DIR / "srg_scores.csv"
    fieldnames = [
        "mel_inchikey",
        "synthon_inchikey",
        "rtcnn_score",
        "dockscore",
        "strain",
        "corermsd",
        "substscore",
        "vlsscore",
        "mel_rank",
        "mel_row",
        "full_synthon_id",
        "source_manifest",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in best.values():
            rec.pop("_rt_f", None)
            w.writerow(rec)

    # Write coverage TSV.
    cov_path = ORACLE_DIR / "coverage.tsv"
    with cov_path.open("w", encoding="utf-8") as f:
        f.write("mel_inchikey\tn_synthons_observed\tranks_seen\trows_seen\tn_source_manifests\n")
        for ik in sorted(coverage):
            c = coverage[ik]
            f.write(
                f"{ik}\t{len(c['synthons'])}\t"
                f"{','.join(sorted(r for r in c['ranks'] if r))}\t"
                f"{','.join(sorted(r for r in c['rows'] if r))}\t"
                f"{len(c['sources'])}\n"
            )

    print(
        f"SDFs ingested: {n_sdfs_seen}",
        f"SDFs skipped (unparseable parent dir / unknown rank): {n_sdfs_skipped}",
        f"records scanned: {n_records_total}",
        f"with InchiKey + RTCNN_Score: {n_records_with_inchi}",
        f"unique (MEL, synthon) pairs kept: {n_records_kept}",
        f"distinct MELs in oracle: {len(coverage)}",
        sep="\n", file=sys.stderr,
    )
    print(f"wrote {csv_path}", file=sys.stderr)
    print(f"wrote {cov_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
