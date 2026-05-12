"""Stdlib SDF helpers for the adaptive (probe-and-commit) SRG batch.

Records are opaque text blocks separated by `$$$$` terminators — no RDKit,
no chemistry awareness, no sanitization. We only need to shuffle records
between files and pull two tag values (`full_synthon_id`, `RTCNN_Score`).

Matches the existing convention in run_srg_batch.py:parse_mel_sdf where
records are obtained by `text.split("$$$$")` and empties are discarded.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable


RECORD_SEP = "$$$$"


def split_sdf(path: Path) -> list[str]:
    """Return record blocks (no `$$$$` terminator). Empty trailing block dropped."""
    text = path.read_text()
    return [r for r in text.split(RECORD_SEP) if r.strip()]


def write_sdf(records: Iterable[str], path: Path) -> int:
    """Write records back with `$$$$\\n` terminators. Returns count."""
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(r.strip("\n") + "\n" + RECORD_SEP + "\n")
            n += 1
    return n


def _tag_value(record: str, tag: str) -> str | None:
    """Return the line immediately after `> <tag>` in `record`, or None."""
    needle = f"> <{tag}>"
    lines = record.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == needle:
            if i + 1 < len(lines):
                return lines[i + 1].strip()
            return None
    return None


def get_synthon_id(record: str) -> str | None:
    """Output-SDF tag (ICM chemSdfExport). Input synthon SDFs do not carry
    this tag — use `get_title` for those."""
    return _tag_value(record, "full_synthon_id")


def get_title(record: str) -> str:
    """First non-blank line of the record (MOL block title). Present in both
    input synthon SDFs and ICM-emitted enumerated SDFs."""
    for line in record.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def get_rtcnn_score(record: str) -> float | None:
    v = _tag_value(record, "RTCNN_Score")
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def subsample(records: list[str], n: int,
              seed: int) -> tuple[list[str], list[str]]:
    """Random subsample without replacement.

    Returns (probe, remainder). Remainder preserves the original order of
    records that were NOT sampled. `n` is clamped to len(records).
    """
    total = len(records)
    n = min(n, total)
    rng = random.Random(seed)
    picked = set(rng.sample(range(total), n))
    probe = [records[i] for i in sorted(picked)]
    remainder = [records[i] for i in range(total) if i not in picked]
    return probe, remainder


def compute_probe_size(n_total: int, probe_n: int, probe_frac: float) -> int:
    """probe size = max(probe_n, ceil(probe_frac * n_total)), clamped to n_total."""
    n = max(probe_n, math.ceil(probe_frac * n_total))
    return min(n, n_total)


def iter_rtcnn(sdf_path: Path):
    """Yield (synthon_id, rtcnn_score) per record. Skips records missing either."""
    for rec in split_sdf(sdf_path):
        sid = get_synthon_id(rec)
        score = get_rtcnn_score(rec)
        if sid is None or score is None:
            continue
        yield sid, score


def merge_sorted_by_rtcnn(sdfs: list[Path], out: Path) -> int:
    """Concat records from all input SDFs, sort by RTCNN_Score ascending, write.

    RTCNN_Score is a binding-energy proxy: lower = better. Ascending order
    puts the best (most-negative) records first.

    Records without a parseable RTCNN_Score go to the tail in original order.
    Returns the number of records written.
    """
    records: list[str] = []
    for p in sdfs:
        if p.exists():
            records.extend(split_sdf(p))
    scored = []
    unscored = []
    for r in records:
        s = get_rtcnn_score(r)
        if s is None:
            unscored.append(r)
        else:
            scored.append((s, r))
    scored.sort(key=lambda kv: kv[0])
    ordered = [r for _, r in scored] + unscored
    return write_sdf(ordered, out)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (stdlib, numpy-compatible for pct in [0,100])."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (pct / 100.0) * (len(xs) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
