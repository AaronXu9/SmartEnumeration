#!/bin/bash
# =============================================================================
# srg_array.sbatch — SLURM array template for the CARC Stage-3 SRG pipeline.
#
# One array task = one MEL row. submit.sh renders this template (substituting
# __N_MELS__, __CONCURRENCY__, headless, srg_pocketInformedV-SYNTHES_headless) and
# submits it via sbatch.
#
# The #SBATCH directives below mirror V-SYNTHES Stage 1 lab norms:
# epyc-64 partition, 16 ntasks (ICM _ligedit_bg auto-chunks across these),
# 4h walltime (per-MEL wall is minutes; 4h is generous headroom), %N
# concurrency cap (starts conservative at 16, tune up after first run).
# =============================================================================
#SBATCH --account=katritch_502
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1%1
#SBATCH --job-name=srg_pocketInformedV-SYNTHES_headless
#SBATCH --output=logs/slurm-%A_%a.out

set -eo pipefail

# ---- env ---------------------------------------------------------------------
# VSYNTHES_ENV=carc tells paths.py to pick the carc branch (ICM_BIN=icmng,
# ICMHOME=/project2/katritch_223/icm-3.9-4a, ICM_FLAGS=["-s"]).
export VSYNTHES_ENV=carc

# Per-task scratch dir for ICM's tempfiles (lig*_out.icb chunks land here).
# /scratch1 is node-local fast storage; /project2 would bottleneck IO for
# the 16 parallel synthon-docking workers on a single SRG.
TMPROOT="/scratch1/${USER}/icm_tmp/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPROOT}"
export TMPDIR="${TMPROOT}"

# ICM's processLigandICM l_bg=yes nProc=0 auto-chunks across OMP threads.
export OMP_NUM_THREADS=${SLURM_NTASKS:-16}

# Array task ID is 0-based; MEL rows are 1-based.
MEL_ROW=$((SLURM_ARRAY_TASK_ID + 1))

PROJECT_ROOT="/project2/katritch_223/aoxu/pocketInformedV-SYNTHES"
TEMPLATE_VARIANT="headless"

cd "${PROJECT_ROOT}"
echo "[$(date -Iseconds)] host=$(hostname)  job=${SLURM_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  row=${MEL_ROW}  template=${TEMPLATE_VARIANT}  tmpdir=${TMPDIR}"

python3 scripts/run_one_mel.py --row "${MEL_ROW}" --template "${TEMPLATE_VARIANT}"
rc=$?

# Clean per-task scratch regardless of outcome so we don't fill /scratch1.
rm -rf "${TMPROOT}"

exit ${rc}
