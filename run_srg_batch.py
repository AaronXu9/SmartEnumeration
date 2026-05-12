#!/usr/bin/env python3
"""Batch wrapper around run_srg_single_apo_export_diskmaps.icm.

Runs ICM Screen Replacement Group for each MEL entry in final_table_edited.sdf
with a matching synthon library in synthons/. Each ICM call is a separate
subprocess for failure isolation. Outputs land in
results/MEL_<row>_<rank>/enumerated.sdf so the analysis notebook only has
to swap one path to switch between MELs.

Two modes:
  --adaptive : two-pass probe-and-commit. Phase 1 probes every MEL with a
               random sample; Phase 2 allocates a shared commit budget
               across passing MELs weighted by probe quality; Phase 3
               commits per passing MEL. Rich MELs get more resources, not
               less (opposite of the old target-K behavior).
  classic    : single-pass docking of the full synthon library per MEL.

RTCNN_Score is a binding-energy proxy — lower = better. All hit counts use
`score <= threshold`, merges sort ascending.

Usage:
    python3 run_srg_batch.py                  # classic flow
    python3 run_srg_batch.py --adaptive       # probe/allocate/commit flow
    python3 run_srg_batch.py --dry-run        # print plan, no ICM
    python3 run_srg_batch.py --only-row 2     # single MEL row
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import sdf_utils
from paths import (
    APO_TSV,
    ICM_BIN,
    ICM_FLAGS,
    MAPS_DIR,
    MEL_SDF,
    PROJECT_ROOT,
    RESULTS_DIR,
    SYNTHONS_DIR,
    TEMPLATE_CONVERGE_NOGUI,
    TEMPLATE_ICM,
    TEMPLATE_ICM_HEADLESS,
)
from srg_core import (
    MelEntry,
    Renderer,
    SynthonFile,
    SYNTHON_FILENAME_RE,
    check_nn_score,
    count_sdf_records,
    invoke_icm,
    parse_apo_tsv,
    parse_mel_sdf,
    render_icm,
    render_icm_converge,
    render_icm_headless,
    run_one_classic,
    scan_synthons,
    select_template_and_renderer,
)

ICM_DEFAULT = str(ICM_BIN)


@dataclass
class ProbeResult:
    """Outcome of a single MEL's probe pass. Fed into the allocator and the
    commit pass. For failed/aborted probes, `scores` is empty and
    `decision_dict["decision"]` is one of '-' or 'abort'."""
    mel: MelEntry
    synth: SynthonFile
    out_dir: Path
    probe_sdf: Path
    remainder_records: list[str]
    n_total: int
    n_probe: int
    n_probe_scored: int
    probe_elapsed: float
    scores: list[float]
    decision_dict: dict
    status: str                   # "pass" / "abort" / "fail_probe" / "warn_nn_zero" / "dry_run"

    # Fields read by the allocator (SimpleNamespace-compatible duck-typing).
    @property
    def row(self) -> int:
        return self.mel.row

    @property
    def remainder(self) -> int:
        return len(self.remainder_records)

    @property
    def expected_hits(self) -> float:
        v = self.decision_dict.get("expected_hits_total", 0.0)
        return float(v) if v is not None else 0.0

    @property
    def probe_hits(self) -> int:
        v = self.decision_dict.get("probe_hits_at_threshold", 0)
        return int(v) if v is not None else 0

    @property
    def hit_density(self) -> float:
        return (self.probe_hits / self.n_probe_scored) if self.n_probe_scored else 0.0


def evaluate_probe(scores: list[float], n_total: int, n_probe: int,
                   args: argparse.Namespace) -> dict:
    """Evaluate a probe's RTCNN score distribution against the selected
    stopping criterion and return a JSON-safe stats/decision dict.

    RTCNN_Score is a binding-energy proxy — lower = better. Hits are
    `score <= hit_threshold`; the probe's 'best' score is `min(scores)`;
    the percentile criterion uses a low-tail percentile (e.g. P5).
    """
    if not scores:
        return {
            "criterion": args.stop_criterion,
            "n_probe_scored": 0, "n_probe_input": n_probe, "n_total": n_total,
            "decision": "abort", "reason": "no parseable RTCNN scores in probe output",
        }
    probe_best = min(scores)
    median = statistics.median(scores)
    hits = sum(1 for s in scores if s <= args.hit_threshold)
    expected = hits * n_total / len(scores)
    stats = {
        "criterion": args.stop_criterion,
        "n_probe_scored": len(scores),
        "n_probe_input": n_probe,
        "n_total": n_total,
        "probe_best": probe_best,
        "probe_median": median,
        "hit_threshold": args.hit_threshold,
        "probe_hits_at_threshold": hits,
        "expected_hits_total": expected,
    }
    if args.stop_criterion == "expected-hits":
        passed = expected >= args.min_expected_hits
        stats["min_expected_hits"] = args.min_expected_hits
        reason = (f"expected_hits={expected:.2f} "
                  f"{'>=' if passed else '<'} min={args.min_expected_hits}")
    elif args.stop_criterion == "top-score":
        passed = probe_best <= args.top_threshold
        stats["top_threshold"] = args.top_threshold
        reason = (f"probe_best={probe_best:.2f} "
                  f"{'<=' if passed else '>'} threshold={args.top_threshold}")
    elif args.stop_criterion == "percentile":
        pct_val = sdf_utils.percentile(scores, args.percentile)
        passed = pct_val <= args.pct_threshold
        stats["percentile"] = args.percentile
        stats["percentile_value"] = pct_val
        stats["pct_threshold"] = args.pct_threshold
        reason = (f"P{args.percentile:g}={pct_val:.2f} "
                  f"{'<=' if passed else '>'} threshold={args.pct_threshold}")
    else:
        raise ValueError(f"unknown --stop-criterion: {args.stop_criterion!r}")
    stats["decision"] = "pass" if passed else "abort"
    stats["reason"] = reason
    return stats


def allocate_budget(passing, budget: int, alpha: float,
                    min_commit: int) -> dict[int, int]:
    """Share `budget` synthons across `passing` MELs weighted by
    `expected_hits ** alpha`, respecting per-MEL caps (remainder) and a
    per-MEL floor (min_commit, clipped by remainder).

    Parameters:
      passing: iterable of objects with `.row`, `.remainder`,
               `.expected_hits` (duck-typed — a ProbeResult or a
               SimpleNamespace both work).
      budget: total commit synthons to distribute.
      alpha: 0 = uniform (explore); 1 = linear (proportional exploit);
             >1 = steeper exploit.
      min_commit: per-MEL floor. Clipped by remainder — a MEL with
                  remainder < min_commit gets its remainder, not the floor.

    Returns: {row: commit_n}. Sum may slightly exceed `budget` once floors
    are applied; sum may be below `budget` if every MEL caps at its
    remainder (no more room to spill). Both are acceptable.
    """
    passing = list(passing)
    if not passing:
        return {}

    remainders = {p.row: int(p.remainder) for p in passing}
    weights = {p.row: max(float(p.expected_hits), 1e-9) ** alpha for p in passing}

    commit_n: dict[int, int] = {}
    active = set(remainders.keys())
    B = int(budget)

    # Iterative cap-and-spill: when a MEL's raw allocation >= its remainder,
    # cap it at remainder, subtract from B, and redistribute to the rest.
    while active:
        total_w = sum(weights[r] for r in active)
        if total_w <= 0:
            for r in list(active):
                commit_n[r] = 0
            break
        capped = False
        for r in list(active):
            raw = int(round(B * weights[r] / total_w))
            if raw >= remainders[r]:
                commit_n[r] = remainders[r]
                B -= remainders[r]
                active.discard(r)
                capped = True
        if not capped:
            for r in list(active):
                commit_n[r] = max(0, int(round(B * weights[r] / total_w)))
            break

    # Apply floor, clipped by remainder. Floors may overshoot B slightly.
    for p in passing:
        r = p.row
        commit_n[r] = min(max(commit_n.get(r, 0), min_commit), remainders[r])

    return commit_n


def _compute_raw_alloc(passing, budget: int, alpha: float) -> dict[int, int]:
    """Pre-cap, pre-floor raw allocation for manifest visibility.
    `round(B * w_i / sum(w))` without iteration."""
    passing = list(passing)
    if not passing:
        return {}
    weights = {p.row: max(float(p.expected_hits), 1e-9) ** alpha for p in passing}
    total_w = sum(weights.values())
    if total_w <= 0:
        return {p.row: 0 for p in passing}
    return {p.row: int(round(budget * weights[p.row] / total_w))
            for p in passing}


def probe_mel(mel: MelEntry, synth: SynthonFile, out_dir: Path,
              template: str, icm_bin: str, dry_run: bool,
              args: argparse.Namespace,
              renderer: Renderer = render_icm) -> ProbeResult:
    """Run the probe pass for one MEL and return a ProbeResult."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = sdf_utils.split_sdf(synth.path)
    n_total = len(all_records)
    n_probe = sdf_utils.compute_probe_size(n_total, args.probe_n, args.probe_frac)
    probe_records, remainder_records = sdf_utils.subsample(
        all_records, n_probe, seed=args.probe_seed)

    probe_input = out_dir / "probe_input.sdf"
    sdf_utils.write_sdf(probe_records, probe_input)
    print(f"  [row {mel.row}] probe {n_probe}/{n_total} synthons "
          f"(seed={args.probe_seed})")

    probe_run = out_dir / "probe_run.icm"
    probe_run.write_text(renderer(template, mel.row, probe_input, out_dir,
                                  probe_input.stem,
                                  out_sdf_name="probe_enumerated.sdf"))
    probe_sdf = out_dir / "probe_enumerated.sdf"

    if dry_run:
        print(f"  [row {mel.row}] DRY-RUN: wrote probe_input.sdf and probe_run.icm")
        return ProbeResult(
            mel=mel, synth=synth, out_dir=out_dir, probe_sdf=probe_sdf,
            remainder_records=remainder_records, n_total=n_total,
            n_probe=n_probe, n_probe_scored=0, probe_elapsed=0.0, scores=[],
            decision_dict={"decision": "dry_run", "reason": "dry-run"},
            status="dry_run",
        )

    probe_log = out_dir / "probe_icm.log"
    exit_code, probe_elapsed = invoke_icm(icm_bin, probe_run, probe_log, out_dir)
    n_probe_out, n_nn = check_nn_score(probe_sdf)
    print(f"  [row {mel.row}] probe: exit={exit_code}  "
          f"elapsed={probe_elapsed:.1f}s  records={n_probe_out}  nonzero_NN={n_nn}")

    if exit_code != 0 or n_probe_out == 0:
        return ProbeResult(
            mel=mel, synth=synth, out_dir=out_dir, probe_sdf=probe_sdf,
            remainder_records=remainder_records, n_total=n_total,
            n_probe=n_probe, n_probe_scored=n_probe_out,
            probe_elapsed=probe_elapsed, scores=[],
            decision_dict={"decision": "-", "reason": f"probe ICM exit={exit_code}"},
            status="fail_probe",
        )

    if n_nn == 0 and n_probe_out > 10:
        print(f"    WARN: all NN_Score=0 in probe — RTCNN unbound (see CLAUDE.md)")
        return ProbeResult(
            mel=mel, synth=synth, out_dir=out_dir, probe_sdf=probe_sdf,
            remainder_records=remainder_records, n_total=n_total,
            n_probe=n_probe, n_probe_scored=n_probe_out,
            probe_elapsed=probe_elapsed, scores=[],
            decision_dict={"decision": "-", "reason": "all RTCNN_Score=0 (unbound)"},
            status="warn_nn_zero",
        )

    scores = [s for _, s in sdf_utils.iter_rtcnn(probe_sdf)]
    decision_dict = evaluate_probe(scores, n_total, n_probe, args)
    (out_dir / "probe_decision.json").write_text(
        json.dumps(decision_dict, indent=2) + "\n")
    print(f"  [row {mel.row}] decision={decision_dict['decision']}  "
          f"({decision_dict['reason']})")

    return ProbeResult(
        mel=mel, synth=synth, out_dir=out_dir, probe_sdf=probe_sdf,
        remainder_records=remainder_records, n_total=n_total,
        n_probe=n_probe, n_probe_scored=len(scores),
        probe_elapsed=probe_elapsed, scores=scores,
        decision_dict=decision_dict,
        status=decision_dict["decision"],
    )


def commit_mel(pr: ProbeResult, commit_n: int, template: str, icm_bin: str,
               dry_run: bool, args: argparse.Namespace,
               renderer: Renderer = render_icm) -> dict:
    """Run the commit pass for a passing probe and merge probe+commit outputs.

    Returns a dict of manifest fields (status, final_n, elapsed_s).
    """
    if dry_run:
        return {"status": "dry_run", "elapsed_s": 0.0, "final_n": 0}

    mel = pr.mel
    out_dir = pr.out_dir
    probe_sdf = pr.probe_sdf
    remainder_records = pr.remainder_records
    remainder_max = len(remainder_records)
    commit_n = min(max(0, int(commit_n)), remainder_max)

    if commit_n == 0:
        # remainder exhausted or budget=0+floor=0 — probe is the whole output.
        shutil.copy(probe_sdf, out_dir / "enumerated.sdf")
        return {"status": "ok_probe_only", "elapsed_s": pr.probe_elapsed,
                "final_n": pr.n_probe_scored}

    commit_input = out_dir / "commit_input.sdf"
    if commit_n < remainder_max:
        commit_records, _ = sdf_utils.subsample(
            remainder_records, commit_n, seed=args.probe_seed + 1)
    else:
        commit_records = remainder_records
    sdf_utils.write_sdf(commit_records, commit_input)

    commit_run = out_dir / "commit_run.icm"
    commit_run.write_text(renderer(template, mel.row, commit_input, out_dir,
                                   commit_input.stem,
                                   out_sdf_name="commit_enumerated.sdf"))
    commit_log = out_dir / "commit_icm.log"
    exit_code, commit_elapsed = invoke_icm(icm_bin, commit_run, commit_log, out_dir)

    commit_sdf = out_dir / "commit_enumerated.sdf"
    n_commit_out, _ = check_nn_score(commit_sdf)
    print(f"  [row {mel.row}] commit: exit={exit_code}  "
          f"elapsed={commit_elapsed:.1f}s  records={n_commit_out}")

    if exit_code != 0 or n_commit_out == 0:
        shutil.copy(probe_sdf, out_dir / "enumerated.sdf")
        return {"status": "fail_commit",
                "elapsed_s": pr.probe_elapsed + commit_elapsed,
                "final_n": pr.n_probe_scored}

    final_n = sdf_utils.merge_sorted_by_rtcnn(
        [probe_sdf, commit_sdf], out_dir / "enumerated.sdf")
    print(f"  [row {mel.row}] merged -> enumerated.sdf  records={final_n}")

    return {"status": "ok_committed",
            "elapsed_s": pr.probe_elapsed + commit_elapsed,
            "final_n": final_n}


MANIFEST_HEADER = (
    "row", "inchikey", "rank", "apo_idx", "out_dir", "status",
    "n_total", "n_probe", "probe_best", "probe_median",
    "probe_hits", "hit_density", "expected_hits", "remainder",
    "decision", "alloc_weight", "alloc_raw", "commit_n",
    "final_n", "elapsed_s",
)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.4f}"
    return str(v)


def _skip_manifest_row(mel: MelEntry, action: str) -> tuple[str, ...]:
    """Pad a skipped MEL row to len(MANIFEST_HEADER)."""
    dash = "-"
    head = (str(mel.row), mel.icm_inchikey, dash, mel.apo_idx or dash,
            dash, action)
    return head + (dash,) * (len(MANIFEST_HEADER) - len(head))


def _classic_manifest_row(mel: MelEntry, synth: SynthonFile, out_dir: Path,
                          result: dict) -> tuple[str, ...]:
    """Classic path has no probe/allocation fields — fill with dashes."""
    dash = "-"
    head = (str(mel.row), mel.icm_inchikey, synth.rank_label, mel.apo_idx,
            str(out_dir.relative_to(PROJECT_ROOT)), result["status"])
    tail = (_fmt(result.get("final_n")), _fmt(result.get("elapsed_s")))
    middle = (dash,) * (len(MANIFEST_HEADER) - len(head) - len(tail))
    return head + middle + tail


def _adaptive_manifest_row(mel: MelEntry, synth: SynthonFile, pr: ProbeResult,
                           commit_n: int, alloc_raw: int, alloc_weight: float,
                           result: dict) -> tuple[str, ...]:
    d = pr.decision_dict
    return (
        str(mel.row), mel.icm_inchikey, synth.rank_label, mel.apo_idx,
        str(pr.out_dir.relative_to(PROJECT_ROOT)), result["status"],
        _fmt(pr.n_total), _fmt(pr.n_probe),
        _fmt(d.get("probe_best")), _fmt(d.get("probe_median")),
        _fmt(d.get("probe_hits_at_threshold")),
        _fmt(pr.hit_density), _fmt(d.get("expected_hits_total")),
        _fmt(pr.remainder),
        _fmt(d.get("decision")),
        _fmt(alloc_weight), _fmt(alloc_raw), _fmt(commit_n),
        _fmt(result.get("final_n")), _fmt(result.get("elapsed_s")),
    )


def run_batch_classic(plan, results_dir: Path, template: str,
                      args: argparse.Namespace,
                      renderer: Renderer = render_icm) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for mel, synth, action in plan:
        if action != "run":
            rows.append(_skip_manifest_row(mel, action))
            continue
        out_dir = results_dir / f"MEL_{mel.row}_{synth.rank_label}"
        print(f"MEL row {mel.row} -> {out_dir.relative_to(PROJECT_ROOT)}")
        result = run_one_classic(mel, synth, out_dir, template,
                                 args.icm, args.dry_run, renderer=renderer)
        rows.append(_classic_manifest_row(mel, synth, out_dir, result))
    return rows


def run_batch_adaptive(plan, results_dir: Path, template: str,
                       args: argparse.Namespace,
                       renderer: Renderer = render_icm) -> list[tuple[str, ...]]:
    """Two-pass orchestrator: probe every 'run' MEL, allocate a shared budget
    over passing MELs, then commit.

    Rows are appended in the same order as `plan` so the manifest TSV mirrors
    the input order.
    """
    # ---- Phase 1: probe all "run" MELs ----------------------------------
    probe_results: dict[int, ProbeResult] = {}
    for mel, synth, action in plan:
        if action != "run":
            continue
        out_dir = results_dir / f"MEL_{mel.row}_{synth.rank_label}"
        print(f"MEL row {mel.row} -> {out_dir.relative_to(PROJECT_ROOT)}  [probe]")
        probe_results[mel.row] = probe_mel(
            mel, synth, out_dir, template, args.icm, args.dry_run, args,
            renderer=renderer)

    # ---- Phase 2: allocate budget across passing MELs -------------------
    passing = [p for p in probe_results.values() if p.status == "pass"]
    if args.commit_budget_n is not None:
        budget = max(0, int(args.commit_budget_n))
    else:
        total_remainder = sum(p.remainder for p in passing)
        budget = math.ceil(args.commit_budget_frac * total_remainder)
    allocations = allocate_budget(
        passing, budget=budget, alpha=args.alloc_alpha,
        min_commit=args.min_commit)
    raw_alloc = _compute_raw_alloc(passing, budget, args.alloc_alpha)
    weights = {p.row: max(float(p.expected_hits), 1e-9) ** args.alloc_alpha
               for p in passing}

    print(f"\nAllocation: budget={budget}  alpha={args.alloc_alpha}  "
          f"min_commit={args.min_commit}  passing={len(passing)}")
    for p in passing:
        print(f"  MEL {p.row:>3}: remainder={p.remainder:>6}  "
              f"E_hits={p.expected_hits:>8.2f}  "
              f"weight={weights[p.row]:>10.4g}  "
              f"raw={raw_alloc[p.row]:>6}  commit_n={allocations[p.row]:>6}")

    # ---- Phase 3: commit per passing MEL; short-circuit others ----------
    rows: list[tuple[str, ...]] = []
    for mel, synth, action in plan:
        if action != "run":
            rows.append(_skip_manifest_row(mel, action))
            continue
        pr = probe_results[mel.row]
        commit_n = allocations.get(mel.row, 0)
        if pr.status == "pass":
            print(f"\nMEL row {mel.row} -> commit (n={commit_n})")
            result = commit_mel(pr, commit_n, template, args.icm,
                                args.dry_run, args, renderer=renderer)
        elif pr.status == "abort":
            # Expose probe as final output so the analysis path still resolves.
            if not args.dry_run:
                shutil.copy(pr.probe_sdf, pr.out_dir / "enumerated.sdf")
            result = {"status": "ok_aborted",
                      "elapsed_s": pr.probe_elapsed,
                      "final_n": pr.n_probe_scored}
        elif pr.status == "dry_run":
            result = {"status": "dry_run", "elapsed_s": 0.0, "final_n": 0}
        elif pr.status in ("fail_probe", "warn_nn_zero"):
            result = {"status": pr.status,
                      "elapsed_s": pr.probe_elapsed,
                      "final_n": pr.n_probe_scored}
        else:
            result = {"status": f"unknown_{pr.status}",
                      "elapsed_s": pr.probe_elapsed,
                      "final_n": pr.n_probe_scored}
        rows.append(_adaptive_manifest_row(
            mel, synth, pr, commit_n,
            raw_alloc.get(mel.row, 0),
            weights.get(mel.row, 0.0),
            result))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan and render scripts, do not invoke ICM")
    ap.add_argument("--only-row", type=int, default=None,
                    help="run only this MEL row (1-based index into final_table_edited.sdf)")
    ap.add_argument("--icm", default=ICM_DEFAULT,
                    help=f"ICM executable path (default: {ICM_DEFAULT})")
    ap.add_argument("--synthon-path", type=Path, default=None,
                    help="Override synthon SDF for every planned MEL (useful for "
                         "fast smoke tests — e.g. combiDock_R1.sdf, 8 records). "
                         "Bypasses the synthons/ filename scan; out_dir becomes "
                         "MEL_<row>_<stem>/ so it doesn't clobber real results.")
    ap.add_argument("--template", choices=("default", "headless", "converge"),
                    default="default",
                    help="which ICM template to drive. 'default' = "
                         "run_srg_single_apo_export_diskmaps.icm (our working "
                         "parallel path, l_bg=yes nProc=0; uses openFile so "
                         "it needs a GUI ICM like icm64 on Mac / Linux KatLab). "
                         "'headless' = run_srg_single_apo_export_diskmaps_headless.icm "
                         "(same flow as default but with read object/read map/"
                         "read table mol in place of openFile; required for "
                         "icmng on CARC, also works with icm64 -g / -s). "
                         "'converge' = run_srg_single_apo_export_diskmaps_converge_noGUI.icm "
                         "(Wenjin's variant: foreground processLigandICM, "
                         "skips e3dSetReceptor so scores ~3 units more-negative "
                         "than default; noGUI so it's safe under icm64 -g / icmng).")

    adg = ap.add_argument_group("adaptive probe-and-commit")
    adg.add_argument("--adaptive", action="store_true",
                     help="two-pass: probe all MELs, allocate a shared commit "
                          "budget weighted by probe quality, then commit. "
                          "Outputs are routed to <results_dir>/adaptive/.")
    adg.add_argument("--probe-n", type=int, default=500,
                     help="floor for probe size (default: 500)")
    adg.add_argument("--probe-frac", type=float, default=0.05,
                     help="probe size = max(--probe-n, ceil(--probe-frac * N_total)) "
                          "(default: 0.05)")
    adg.add_argument("--probe-seed", type=int, default=42,
                     help="RNG seed for deterministic subsampling (default: 42)")
    adg.add_argument("--stop-criterion",
                     choices=("expected-hits", "top-score", "percentile"),
                     default="expected-hits",
                     help="decision rule applied to probe RTCNN scores "
                          "(default: expected-hits). RTCNN_Score is a binding-"
                          "energy proxy — lower = better.")
    adg.add_argument("--hit-threshold", type=float, default=-25.0,
                     help="RTCNN_Score cutoff; a synthon is a hit if "
                          "score <= threshold (lower = better; default: -25.0, "
                          "matches the Stage-4 filter)")
    adg.add_argument("--min-expected-hits", type=float, default=10.0,
                     help="commit iff extrapolated total hits >= this "
                          "(expected-hits criterion; default: 10.0)")
    adg.add_argument("--top-threshold", type=float, default=-30.0,
                     help="commit iff min(probe scores) <= this "
                          "(top-score criterion; lower = better; default: -30.0)")
    adg.add_argument("--percentile", type=float, default=5.0,
                     help="percentile used by the percentile criterion "
                          "(default: 5.0 — the low tail / best percentile "
                          "under lower = better)")
    adg.add_argument("--pct-threshold", type=float, default=-28.0,
                     help="commit iff Pxx(probe) <= this "
                          "(percentile criterion; lower = better; default: -28.0)")

    adg.add_argument("--commit-budget-frac", type=float, default=0.5,
                     help="commit budget = ceil(frac * sum(remainder_i)) summed "
                          "over passing MELs (default: 0.5). Ignored if "
                          "--commit-budget-n is set.")
    adg.add_argument("--commit-budget-n", type=int, default=None,
                     help="absolute commit budget (synthon count). Overrides "
                          "--commit-budget-frac.")
    adg.add_argument("--alloc-alpha", type=float, default=1.0,
                     help="exponent on expected_hits_i when weighting commit "
                          "shares. 0 = uniform (pure exploration), "
                          "1 = linear (proportional exploit), "
                          ">1 = steeper exploit. Default: 1.0.")
    adg.add_argument("--min-commit", type=int, default=500,
                     help="floor commit_n per passing MEL, clipped by remainder "
                          "(default: 500).")
    args = ap.parse_args()

    if not args.dry_run and not Path(args.icm).exists():
        print(f"ERROR: ICM binary not found: {args.icm}", file=sys.stderr)
        return 2

    synthon_override: SynthonFile | None = None
    if args.synthon_path is not None:
        override_path = args.synthon_path
        if not override_path.is_absolute():
            override_path = (PROJECT_ROOT / override_path).resolve()
        if not override_path.is_file():
            print(f"ERROR: --synthon-path not found: {override_path}", file=sys.stderr)
            return 2
        # `rank_label` becomes the synthon symlink's stem (via `synthons_<label>.sdf`)
        # and from there, the ICM table name. ICM constraints:
        #   - table names truncate at ~60 chars → keep label short
        #   - `$s_synthon_table` re-parses hyphens as subtraction → no hyphens
        # Prefer the `RankN` prefix if the override file matches our naming
        # convention; otherwise sanitize & truncate the stem.
        m = re.match(r"^(Rank\d+)", override_path.stem)
        if m:
            rank_label = m.group(1)
        else:
            rank_label = re.sub(r"[^A-Za-z0-9_]", "_", override_path.stem)[:32]
        synthon_override = SynthonFile(
            path=override_path, rank_label=rank_label,
            icm_inchikey="<override>",
        )
        print(f"Synthon override: {override_path} "
              f"({override_path.stat().st_size} bytes)  rank_label={rank_label}")

    results_dir = RESULTS_DIR / "adaptive" if args.adaptive else RESULTS_DIR

    entries   = parse_mel_sdf(MEL_SDF)
    parse_apo_tsv(APO_TSV, entries)
    synthons  = scan_synthons(SYNTHONS_DIR)
    if args.template == "converge":
        template_path = TEMPLATE_CONVERGE_NOGUI
        renderer: Renderer = render_icm_converge
    elif args.template == "headless":
        template_path = TEMPLATE_ICM_HEADLESS
        renderer = render_icm_headless
    else:
        template_path = TEMPLATE_ICM
        renderer = render_icm
    template = template_path.read_text()

    print(f"MEL entries: {len(entries)}   Synthon files: {len(synthons)}")
    print(f"Template:    {template_path.name}   renderer: {renderer.__name__}")
    print(f"Output root: {results_dir}"
          + ("   mode: adaptive" if args.adaptive else "   mode: classic"))
    if args.adaptive:
        print(f"Probe:     floor={args.probe_n}  frac={args.probe_frac}  "
              f"seed={args.probe_seed}")
        print(f"Criterion: {args.stop_criterion}  hit_threshold={args.hit_threshold}  "
              f"min_expected_hits={args.min_expected_hits}  "
              f"top_threshold={args.top_threshold}  "
              f"P{args.percentile:g}_threshold={args.pct_threshold}")
        print(f"Budget:    frac={args.commit_budget_frac}  "
              f"abs={args.commit_budget_n}  alpha={args.alloc_alpha}  "
              f"min_commit={args.min_commit}")
    print(f"{'row':>3}  {'apo':>4}  {'synthon':>8}  inchikey")
    print("-" * 72)
    plan: list[tuple[MelEntry, SynthonFile | None, str]] = []
    for e in entries:
        if args.only_row is not None and e.row != args.only_row:
            continue
        if not e.apo_idx:
            plan.append((e, None, "skip_no_apo"))
        elif synthon_override is not None:
            plan.append((e, synthon_override, "run"))
        elif e.icm_inchikey not in synthons:
            plan.append((e, None, "skip_no_synthon"))
        else:
            plan.append((e, synthons[e.icm_inchikey], "run"))

    for e, synth, action in plan:
        rank = synth.rank_label if synth else "-"
        print(f"{e.row:>3}  {e.apo_idx or '-':>4}  {rank:>8}  {e.icm_inchikey}  [{action}]")
    print()

    results_dir.mkdir(parents=True, exist_ok=True)
    if args.adaptive:
        manifest_rows = run_batch_adaptive(plan, results_dir, template, args,
                                           renderer=renderer)
    else:
        manifest_rows = run_batch_classic(plan, results_dir, template, args,
                                          renderer=renderer)

    manifest = results_dir / "batch_manifest.tsv"
    with manifest.open("w") as f:
        f.write("\t".join(MANIFEST_HEADER) + "\n")
        for row in manifest_rows:
            f.write("\t".join(row) + "\n")

    n_ok = n_fail = n_skip = 0
    for row in manifest_rows:
        status = row[5]
        if status.startswith("ok") or status == "dry_run":
            n_ok += 1
        elif status.startswith("skip"):
            n_skip += 1
        else:
            n_fail += 1

    print(f"\nManifest: {manifest}")
    print(f"Summary: ok={n_ok}  fail={n_fail}  skipped={n_skip}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
