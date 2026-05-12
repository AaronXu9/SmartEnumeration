"""Compare adaptive probe-and-commit SRG output against the classic baseline.

Reads both batch manifests and per-MEL `enumerated.sdf` files, joins on row,
and reports:

  * overlap between synthon-ID sets (pass MELs should be ~1.0; aborts are
    the probe subsample fraction ~10%)
  * top-10/100/1000 recovery of classic's top hits inside adaptive's output
  * score parity on common IDs (ICM re-scoring jitter)
  * abort safety — how many classic hits at the hit_threshold were missed
    by the probe subsample
  * per-MEL adaptive elapsed_s (classic per-MEL timing was not recorded)

Writes adaptive_vs_classic.tsv + adaptive_vs_classic.png next to the
adaptive manifest.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sdf_utils


CLASSIC_MANIFEST = Path("results/batch_manifest.tsv")
ADAPT_MANIFEST = Path("results_local_macos/adaptive/batch_manifest.tsv")
# RTCNN_Score is a binding-energy proxy: lower = better. Match the default in
# run_srg_batch.py --hit-threshold; the Stage-4 filter cutoff is -25.
DEFAULT_HIT_THR = -25.0
CLASSIC_TOTAL_MINUTES = 54.0


@dataclass
class MelRow:
    row: int
    rank: str
    inchikey: str
    classic_status: str
    classic_dir: Path | None
    adapt_status: str
    adapt_decision: str
    adapt_dir: Path | None
    adapt_elapsed_s: float | None
    n_total: int | None
    n_probe: int | None


@dataclass
class Comparison:
    row: int
    rank: str
    adapt_decision: str
    n_classic: int = 0
    n_adaptive: int = 0
    overlap_pct: float = 0.0
    missing_in_adapt: int = 0
    extra_in_adapt: int = 0
    classic_top1: float = float("nan")
    adaptive_top1: float = float("nan")
    top1_delta: float = float("nan")
    top10_recovery: float = float("nan")
    top100_recovery: float = float("nan")
    top1000_recovery: float = float("nan")
    score_diff_mean: float = float("nan")
    score_diff_p95: float = float("nan")
    missed_hits_at_thr: int = 0
    adapt_elapsed_s: float = float("nan")
    score_diffs: list[float] = field(default_factory=list)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        return [r for r in rdr]


def _join_manifests(classic: list[dict[str, str]],
                    adapt: list[dict[str, str]]) -> list[MelRow]:
    ac_by_row = {int(r["row"]): r for r in classic}
    aa_by_row = {int(r["row"]): r for r in adapt}
    rows = sorted(set(ac_by_row) | set(aa_by_row))
    out: list[MelRow] = []
    for r in rows:
        c = ac_by_row.get(r, {})
        a = aa_by_row.get(r, {})
        out.append(MelRow(
            row=r,
            rank=(c.get("rank") or a.get("rank") or "-"),
            inchikey=(c.get("inchikey") or a.get("inchikey") or "-"),
            classic_status=c.get("status", "-"),
            classic_dir=Path(c["out_dir"]) if c.get("out_dir", "-") != "-" else None,
            adapt_status=a.get("status", "-"),
            adapt_decision=a.get("decision", "-"),
            adapt_dir=Path(a["out_dir"]) if a.get("out_dir", "-") != "-" else None,
            adapt_elapsed_s=_maybe_float(a.get("elapsed_s")),
            n_total=_maybe_int(a.get("n_total")),
            n_probe=_maybe_int(a.get("n_probe")),
        ))
    return out


def _maybe_float(v: str | None) -> float | None:
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _maybe_int(v: str | None) -> int | None:
    f = _maybe_float(v)
    return int(f) if f is not None else None


def _top_n_recovery(classic: dict[str, float], adapt_ids: set[str], n: int) -> float:
    """Fraction of classic's top-N (lowest-score = best under lower-is-better)
    found in adapt_ids."""
    if len(classic) < n:
        n = len(classic)
    if n == 0:
        return float("nan")
    top_ids = [sid for sid, _ in
               sorted(classic.items(), key=lambda kv: kv[1])[:n]]
    return sum(1 for sid in top_ids if sid in adapt_ids) / n


def compare_one(mel: MelRow, hit_thr: float) -> Comparison | None:
    if mel.classic_dir is None or mel.adapt_dir is None:
        return None
    classic_path = mel.classic_dir / "enumerated.sdf"
    adapt_path = mel.adapt_dir / "enumerated.sdf"
    if not classic_path.exists() or not adapt_path.exists():
        return None

    classic = dict(sdf_utils.iter_rtcnn(classic_path))
    adapt = dict(sdf_utils.iter_rtcnn(adapt_path))

    common = set(classic) & set(adapt)
    union = set(classic) | set(adapt)
    missing = set(classic) - set(adapt)
    extra = set(adapt) - set(classic)

    diffs = [classic[s] - adapt[s] for s in common]
    diffs.sort()

    comp = Comparison(
        row=mel.row,
        rank=mel.rank,
        adapt_decision=mel.adapt_decision,
        n_classic=len(classic),
        n_adaptive=len(adapt),
        overlap_pct=(100 * len(common) / len(union)) if union else float("nan"),
        missing_in_adapt=len(missing),
        extra_in_adapt=len(extra),
        # RTCNN_Score is lower = better, so "top-1" is the minimum score.
        classic_top1=min(classic.values()) if classic else float("nan"),
        adaptive_top1=min(adapt.values()) if adapt else float("nan"),
        top10_recovery=_top_n_recovery(classic, set(adapt), 10),
        top100_recovery=_top_n_recovery(classic, set(adapt), 100),
        top1000_recovery=_top_n_recovery(classic, set(adapt), 1000),
        missed_hits_at_thr=sum(1 for sid in missing if classic[sid] <= hit_thr),
        adapt_elapsed_s=(mel.adapt_elapsed_s if mel.adapt_elapsed_s is not None
                         else float("nan")),
        score_diffs=diffs,
    )
    comp.top1_delta = comp.classic_top1 - comp.adaptive_top1
    if diffs:
        comp.score_diff_mean = statistics.fmean(diffs)
        comp.score_diff_p95 = sdf_utils.percentile([abs(d) for d in diffs], 95)
    return comp


def _fmt(v: float | int | str, w: int = 10, prec: int = 3) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "-".rjust(w)
        return f"{v:.{prec}f}".rjust(w)
    return str(v).rjust(w)


def print_summary(comps: list[Comparison]) -> None:
    hdr = [
        ("row", 4), ("rank", 6), ("decision", 10),
        ("n_cls", 7), ("n_adp", 7), ("overlap%", 9),
        ("miss", 5), ("extra", 6),
        ("c_top1", 8), ("a_top1", 8),
        ("top10", 7), ("top100", 7), ("top1k", 7),
        ("diff_p95", 9), ("missed≥thr", 11), ("elapsed_s", 10),
    ]
    row_fmt = [w for _, w in hdr]
    print("  ".join(lbl.rjust(w) for lbl, w in hdr))
    print("-" * (sum(row_fmt) + 2 * len(hdr)))
    for c in comps:
        cells = [
            str(c.row).rjust(4),
            c.rank.rjust(6),
            c.adapt_decision.rjust(10),
            str(c.n_classic).rjust(7),
            str(c.n_adaptive).rjust(7),
            _fmt(c.overlap_pct, 9, 2),
            str(c.missing_in_adapt).rjust(5),
            str(c.extra_in_adapt).rjust(6),
            _fmt(c.classic_top1, 8, 2),
            _fmt(c.adaptive_top1, 8, 2),
            _fmt(c.top10_recovery, 7, 2),
            _fmt(c.top100_recovery, 7, 2),
            _fmt(c.top1000_recovery, 7, 2),
            _fmt(c.score_diff_p95, 9, 3),
            str(c.missed_hits_at_thr).rjust(11),
            _fmt(c.adapt_elapsed_s, 10, 1),
        ]
        print("  ".join(cells))


def write_tsv(comps: list[Comparison], mels: list[MelRow], path: Path,
              hit_thr: float) -> None:
    cols = [
        "row", "rank", "classic_status", "adapt_status", "adapt_decision",
        "n_classic", "n_adaptive", "overlap_pct",
        "missing_in_adapt", "extra_in_adapt",
        "classic_top1", "adaptive_top1", "top1_delta",
        "top10_recovery", "top100_recovery", "top1000_recovery",
        "score_diff_mean", "score_diff_p95",
        f"missed_hits_at_{hit_thr:g}", "adapt_elapsed_s",
    ]
    comps_by_row = {c.row: c for c in comps}
    with path.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for m in mels:
            c = comps_by_row.get(m.row)
            if c is None:
                w.writerow([m.row, m.rank, m.classic_status, m.adapt_status,
                            m.adapt_decision] + ["-"] * (len(cols) - 5))
                continue
            w.writerow([
                c.row, c.rank, m.classic_status, m.adapt_status, c.adapt_decision,
                c.n_classic, c.n_adaptive, f"{c.overlap_pct:.3f}",
                c.missing_in_adapt, c.extra_in_adapt,
                f"{c.classic_top1:.4f}", f"{c.adaptive_top1:.4f}",
                f"{c.top1_delta:.4f}",
                f"{c.top10_recovery:.4f}", f"{c.top100_recovery:.4f}",
                f"{c.top1000_recovery:.4f}",
                f"{c.score_diff_mean:.4f}", f"{c.score_diff_p95:.4f}",
                c.missed_hits_at_thr,
                f"{c.adapt_elapsed_s:.1f}",
            ])


def write_png(comps: list[Comparison], path: Path, hit_thr: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # A: top-N recovery bars ------------------------------------------------
    ax = axes[0, 0]
    labels = [f"MEL{c.row}" for c in comps]
    x = range(len(comps))
    width = 0.25
    top10 = [c.top10_recovery for c in comps]
    top100 = [c.top100_recovery for c in comps]
    top1000 = [c.top1000_recovery for c in comps]
    ax.bar([i - width for i in x], top10, width, label="top-10", color="#2b7bb9")
    ax.bar(x, top100, width, label="top-100", color="#f39c12")
    ax.bar([i + width for i in x], top1000, width, label="top-1000",
           color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lbl}\n{c.adapt_decision}"
                        for lbl, c in zip(labels, comps)], fontsize=9)
    ax.set_ylabel("fraction of classic top-N found in adaptive")
    ax.set_title("A. Top-N recovery (pass→≈1.0, abort→probe fraction)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower left", fontsize=9)

    # B: elapsed_s per MEL colored by decision ------------------------------
    ax = axes[0, 1]
    colors = ["#27ae60" if c.adapt_decision == "pass" else "#c0392b"
              for c in comps]
    ax.bar(x, [c.adapt_elapsed_s for c in comps], color=colors)
    total_s = sum(c.adapt_elapsed_s for c in comps
                  if c.adapt_elapsed_s == c.adapt_elapsed_s)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("adaptive elapsed_s")
    ax.set_title(
        f"B. Adaptive wall-clock per MEL (total {total_s / 60:.1f} min "
        f"vs classic ≈{CLASSIC_TOTAL_MINUTES:.0f} min)"
    )
    for i, c in enumerate(comps):
        ax.text(i, c.adapt_elapsed_s, f"{c.adapt_elapsed_s:.0f}s",
                ha="center", va="bottom", fontsize=8)

    # C: top1 scatter -------------------------------------------------------
    ax = axes[1, 0]
    xs = [c.classic_top1 for c in comps]
    ys = [c.adaptive_top1 for c in comps]
    for c, xv, yv in zip(comps, xs, ys):
        color = "#27ae60" if c.adapt_decision == "pass" else "#c0392b"
        ax.scatter(xv, yv, color=color, s=80, edgecolor="black", linewidth=0.5,
                   label=None)
        ax.annotate(f"MEL{c.row}", (xv, yv), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    lo = min(xs + ys) - 5
    hi = max(xs + ys) + 5
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=1)
    ax.axvline(hit_thr, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("classic top-1 RTCNN_Score")
    ax.set_ylabel("adaptive top-1 RTCNN_Score")
    ax.set_title("C. Top-1 parity (dashed: hit_threshold)")

    # D: pooled score diff histogram on committed MELs ---------------------
    ax = axes[1, 1]
    pooled = []
    for c in comps:
        if c.adapt_decision == "pass":
            pooled.extend(c.score_diffs)
    if pooled:
        ax.hist(pooled, bins=60, color="#2b7bb9", edgecolor="black",
                linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("classic − adaptive RTCNN_Score (common IDs)")
        ax.set_ylabel("count")
        ax.set_title(
            f"D. Score parity on common IDs, committed MELs only\n"
            f"n={len(pooled)}  mean={statistics.fmean(pooled):+.3f}  "
            f"p95|Δ|={sdf_utils.percentile([abs(d) for d in pooled], 95):.3f}"
        )
    else:
        ax.text(0.5, 0.5, "no committed MELs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("D. Score parity (no data)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classic-manifest", type=Path, default=CLASSIC_MANIFEST)
    ap.add_argument("--adapt-manifest", type=Path, default=ADAPT_MANIFEST)
    ap.add_argument("--hit-threshold", type=float, default=DEFAULT_HIT_THR,
                    help="RTCNN_Score cutoff for 'missed hit' accounting "
                         "(should match the value used in the adaptive run)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write adaptive_vs_classic.{tsv,png} "
                         "(default: dir of --adapt-manifest)")
    args = ap.parse_args()

    classic_rows = _load_manifest(args.classic_manifest)
    adapt_rows = _load_manifest(args.adapt_manifest)
    mels = _join_manifests(classic_rows, adapt_rows)

    comps: list[Comparison] = []
    for m in mels:
        c = compare_one(m, args.hit_threshold)
        if c is not None:
            comps.append(c)

    out_dir = args.out_dir or args.adapt_manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "adaptive_vs_classic.tsv"
    png_path = out_dir / "adaptive_vs_classic.png"

    print_summary(comps)
    write_tsv(comps, mels, tsv_path, args.hit_threshold)
    write_png(comps, png_path, args.hit_threshold)
    print(f"\nWrote: {tsv_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
