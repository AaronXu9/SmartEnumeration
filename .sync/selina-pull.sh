#!/bin/bash
# One-way mirror of a slim subset of Wenjin Liu ("Selina")'s SmartEnum tree
# from CARC into the NAS-side selina/ dir. Resolves the CARC-side symlink
# (selina -> /project2/katritch_223/selina/SmartEnum) and copies only the
# items in INCLUDES below.
#
# Run manually (not from cron). The data isn't changing fast enough to need
# a 2-h cycle, and we don't want surprise NAS-fill events.
#
# Excluded by design: 150 GB of *_enumerated_products.pkl (Stage 5 input,
# pre-docking structures) and all targets other than CB2 (5TH2A, GPR91, MEL).
# Edit INCLUDES below to widen scope.
set -euo pipefail

LOCAL_DIR="/mnt/katritch_lab2/pocketInformedV-SYNTHES/selina"
REMOTE_USER_HOST="aoxu@discovery.usc.edu"
REMOTE_ROOT="/project2/katritch_223/selina/SmartEnum"

# Each entry is a path relative to REMOTE_ROOT. Trailing slash = sync contents,
# no slash = sync the named file/dir.
INCLUDES=(
  "AutomateICMScreenReplacement/"
  "FullLigandICMDocking/"
  "CB2/CB2-5ZTY-TopN-MELFrags/"
  "CB2/CB2-5ZTY-ICM3.9.3-Docked-2Comp-MEL-Fully-Enumerated-Top1KMEL/5ZTY-PKL-Entry-Count-CARC.csv"
  # Production per-MEL surviving-synthon SDFs. Verified 2026-05-11 at 77 GB
  # across 1473 files (per-MEL Rank{N}_*_surviving_synthons_ICMReady_APO.sdf,
  # 2 MB–100 MB each). Too big for the default pull cap; the NAS
  # CB2_5ZTY_debug/compatible_syntons/ already has the debug-fixture subset
  # (5 ranks) we need for reading the audit. Uncomment + raise MAX_PULL_GB
  # if you need the full production set on the NAS.
  # "CB2/CB2-5ZTY-Comaptiable-And-Surviving-Syntons/"
)

# Hard cap on total pull size. If the CARC `du` adds up to more than this,
# abort — protects against pulling Wenjin's full 269 GB by accident.
MAX_PULL_GB="${MAX_PULL_GB:-20}"

SYNC_DIR="/mnt/katritch_lab2/pocketInformedV-SYNTHES/.sync"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$SYNC_DIR/logs/selina-pull-$TS.log"
mkdir -p "$SYNC_DIR/logs" "$LOCAL_DIR"

echo "[$(date +%H:%M:%S)] selina-pull starting"
echo "  remote:  $REMOTE_USER_HOST:$REMOTE_ROOT"
echo "  local:   $LOCAL_DIR"
echo "  cap:     ${MAX_PULL_GB} GB"
echo "  log:     $LOG"
echo

# Preflight: size check on CARC.
echo "Preflight: measuring size of include set on CARC..."
SIZE_CMD="cd $REMOTE_ROOT && du -sb $(printf '%q ' "${INCLUDES[@]}") 2>/dev/null | awk '{sum+=\$1} END {print sum}'"
TOTAL_BYTES="$(ssh "$REMOTE_USER_HOST" "$SIZE_CMD")"
TOTAL_GB="$(awk -v b="$TOTAL_BYTES" 'BEGIN{printf "%.2f", b/1024/1024/1024}')"
echo "  total: ${TOTAL_GB} GB"
if awk -v g="$TOTAL_GB" -v cap="$MAX_PULL_GB" 'BEGIN{exit !(g+0 > cap+0)}'; then
  echo "ERROR: ${TOTAL_GB} GB exceeds cap of ${MAX_PULL_GB} GB. Aborting." >&2
  echo "       Set MAX_PULL_GB=<higher> to override, or narrow INCLUDES." >&2
  exit 2
fi

# Build an --include filter chain that picks out only the listed items,
# anchored at REMOTE_ROOT, then -av --delete-excluded keeps the result tidy.
# Approach: rsync from $REMOTE_ROOT/ to $LOCAL_DIR/ with --files-from would
# work but file/dir mix is awkward. Use one rsync per include instead — small
# overhead per invocation, much cleaner semantics.
echo
for item in "${INCLUDES[@]}"; do
  echo "[$(date +%H:%M:%S)] PULL  $item"
  dest="$LOCAL_DIR/$item"
  # rsync doesn't create parent dirs on the receiver; do it ourselves.
  # For trailing-slash items dest is a dir; for files dest is a file in a dir.
  if [[ "$item" == */ ]]; then
    mkdir -p "$dest"
  else
    mkdir -p "$(dirname "$dest")"
  fi
  rsync -ahr \
    --update \
    --partial \
    --info=stats2 \
    --human-readable \
    --log-file="$LOG" \
    "$REMOTE_USER_HOST:$REMOTE_ROOT/$item" \
    "$dest" 2>&1 | tail -5
  echo
done

echo "[$(date +%H:%M:%S)] selina-pull done. Local size:"
du -sh "$LOCAL_DIR"
