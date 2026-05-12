#!/bin/bash
# =============================================================================
# submit.sh — submit a Stage-3 SRG array job to SLURM on CARC.
#
# Counts MELs from final_table_edited.sdf, renders srg_array.sbatch with the
# right --array spec + template variant + job name, and sbatches it.
#
# Usage:
#   ./scripts/submit.sh                          # full run, default concurrency=16, headless
#   ./scripts/submit.sh --template converge      # use Wenjin's converge template
#   ./scripts/submit.sh --array 0,5,17           # resume specific array indices
#   ./scripts/submit.sh --resume                 # auto-scan status.json, re-queue failures
#   ./scripts/submit.sh --concurrency 8          # override %N
#   ./scripts/submit.sh --dry-run                # render & print the sbatch, don't submit
#   ./scripts/submit.sh --job-name my_test       # override SLURM job name
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEMPLATE_VARIANT="headless"
CONCURRENCY=16
ARRAY_OVERRIDE=""
RESUME=0
DRY_RUN=0
JOB_NAME=""

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --template)      TEMPLATE_VARIANT="$2"; shift 2 ;;
        --array)         ARRAY_OVERRIDE="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --resume)        RESUME=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --job-name)      JOB_NAME="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "${TEMPLATE_VARIANT}" in
    headless|converge) ;;
    default)
        echo "ERROR: --template default uses openFile which is GUI-only; " >&2
        echo "       icmng on CARC cannot run it. Use headless or converge." >&2
        exit 2 ;;
    *)  echo "ERROR: unknown --template: ${TEMPLATE_VARIANT}" >&2; exit 2 ;;
esac

MEL_SDF="$(cd "${PROJECT_ROOT}" && python3 -c 'from paths import MEL_SDF; print(MEL_SDF)')"
if [[ ! -f "${MEL_SDF}" ]]; then
    echo "ERROR: ${MEL_SDF} not found. Stage-0 preprocessing must run first." >&2
    exit 2
fi

# Count MELs by counting `$$$$` separators in final_table_edited.sdf. This
# matches srg_core.parse_mel_sdf's row indexing — row 1 = first record.
N_MELS=$(grep -c '^\$\$\$\$' "${MEL_SDF}" || true)
if [[ "${N_MELS}" -eq 0 ]]; then
    echo "ERROR: ${MEL_SDF} has no MEL records" >&2
    exit 2
fi

# Build the --array spec.
if [[ -n "${ARRAY_OVERRIDE}" ]]; then
    ARRAY_SPEC="${ARRAY_OVERRIDE}%${CONCURRENCY}"
elif [[ "${RESUME}" -eq 1 ]]; then
    # Scan results/MEL_<row>_*/status.json; queue rows missing a status or
    # whose status is not in the done-set. run_one_mel.py is idempotent so
    # running the full array would also work, but this is cheaper.
    echo "Scanning ${PROJECT_ROOT}/results_carc/ for completed rows..."
    PENDING=()
    for row in $(seq 1 "${N_MELS}"); do
        status_file=$(ls -1 "${PROJECT_ROOT}/results_carc/MEL_${row}_"*/status.json 2>/dev/null | head -1 || true)
        if [[ -z "${status_file}" ]]; then
            PENDING+=($((row - 1)))
            continue
        fi
        status=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('status',''))" "${status_file}" 2>/dev/null || echo "")
        case "${status}" in
            ok|ok_dirty_exit|ok_committed|ok_probe_only|ok_aborted|dry_run) : ;;
            *) PENDING+=($((row - 1))) ;;
        esac
    done
    if [[ ${#PENDING[@]} -eq 0 ]]; then
        echo "All ${N_MELS} MELs already completed successfully. Nothing to submit."
        exit 0
    fi
    ARRAY_CSV=$(IFS=,; echo "${PENDING[*]}")
    ARRAY_SPEC="${ARRAY_CSV}%${CONCURRENCY}"
    echo "Resume: ${#PENDING[@]} pending tasks"
else
    ARRAY_SPEC="0-$((N_MELS - 1))%${CONCURRENCY}"
fi

# Job name: default to srg_<project-dir-basename>_<template> for easy spotting
# in squeue; override via --job-name.
if [[ -z "${JOB_NAME}" ]]; then
    JOB_NAME="srg_$(basename "${PROJECT_ROOT}")_${TEMPLATE_VARIANT}"
fi

# Render the sbatch template via envsubst-style sed (avoids bash heredoc
# quoting pitfalls). The template has sentinels __N_MELS__, __CONCURRENCY__,
# __TEMPLATE_VARIANT__, __JOB_NAME__, __PROJECT_ROOT__.
mkdir -p "${PROJECT_ROOT}/logs"
RENDERED="${PROJECT_ROOT}/logs/srg_array.rendered.sbatch"
sed \
    -e "s|__ARRAY_SPEC__|${ARRAY_SPEC}|g" \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    -e "s|__TEMPLATE_VARIANT__|${TEMPLATE_VARIANT}|g" \
    -e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
    "${SCRIPT_DIR}/srg_array.sbatch" > "${RENDERED}"
chmod +x "${RENDERED}"

echo "Rendered: ${RENDERED}"
echo "  array:    ${ARRAY_SPEC}"
echo "  template: ${TEMPLATE_VARIANT}"
echo "  job:      ${JOB_NAME}"
echo "  mels:     ${N_MELS}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "--- rendered sbatch (first 30 lines) ---"
    head -30 "${RENDERED}"
    echo "--- (dry-run: not submitting) ---"
    exit 0
fi

cd "${PROJECT_ROOT}"
sbatch "${RENDERED}"
