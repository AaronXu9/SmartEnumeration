#!/bin/bash
# Bidirectional sync: pull then push, with a single shared timestamp and a lock.
# Strategy: rsync --update (mtime-wins); file-level conflicts preserved under
# .sync/conflicts/<TS>/. Deletions are NOT propagated (no --delete).
set -euo pipefail

LOCAL_DIR="/mnt/katritch_lab2/pocketInformedV-SYNTHES"
SYNC_DIR="$LOCAL_DIR/.sync"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
LOCK="$SYNC_DIR/sync.lock"

mkdir -p "$SYNC_DIR/logs" "$SYNC_DIR/conflicts"

# Pre-flight: fail fast if local storage is dangerously low.
free_gb=$(df -BG --output=avail "$LOCAL_DIR" | tail -1 | tr -dc '0-9')
if [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "ABORT: only ${free_gb}G free on $LOCAL_DIR (threshold ${MIN_FREE_GB}G)." >&2
  echo "Override with MIN_FREE_GB=<n> $0" >&2
  exit 2
fi

# Serialize concurrent runs (cron + manual).
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ABORT: another sync is already running (lock: $LOCK)." >&2
  exit 3
fi

export SYNC_TS="$(date +%Y%m%d-%H%M%S)"
echo "=== sync-both  $SYNC_TS  (free ${free_gb}G) ==="

status=0
"$SYNC_DIR/sync-pull.sh" || { echo "PULL FAILED" >&2; status=1; }
"$SYNC_DIR/sync-push.sh" || { echo "PUSH FAILED" >&2; status=1; }

# Retention: drop old logs and empty/old conflict dirs.
find "$SYNC_DIR/logs"      -type f -mtime +14 -delete 2>/dev/null || true
find "$SYNC_DIR/conflicts" -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
find "$SYNC_DIR/conflicts" -mindepth 1 -type d -empty -delete 2>/dev/null || true

echo "=== sync-both  $SYNC_TS  done (exit $status) ==="
exit $status
