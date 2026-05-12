#!/bin/bash
# Push local -> remote (one-way, --update means mtime-wins).
set -euo pipefail

LOCAL_DIR="/mnt/katritch_lab2/pocketInformedV-SYNTHES"
REMOTE_USER_HOST="aoxu@discovery.usc.edu"
REMOTE_DIR="/project2/katritch_223/aoxu/pocketInformedV-SYNTHES"

SYNC_DIR="$LOCAL_DIR/.sync"
TS="${SYNC_TS:-$(date +%Y%m%d-%H%M%S)}"
LOG="$SYNC_DIR/logs/push-$TS.log"
# --backup-dir on push is evaluated on the remote; use absolute remote path.
BACKUP_REMOTE_ABS="$REMOTE_DIR/.sync/conflicts/$TS/from-local"

mkdir -p "$SYNC_DIR/logs"
ssh "$REMOTE_USER_HOST" "mkdir -p '$BACKUP_REMOTE_ABS'"

echo "[$(date +%H:%M:%S)] PUSH  $LOCAL_DIR/  ->  $REMOTE_USER_HOST:$REMOTE_DIR/"

rsync -ah \
  --update \
  --partial \
  --info=progress2,stats2 \
  --human-readable \
  --exclude-from="$SYNC_DIR/exclude.txt" \
  --backup \
  --backup-dir="$BACKUP_REMOTE_ABS" \
  --log-file="$LOG" \
  "$LOCAL_DIR/" "$REMOTE_USER_HOST:$REMOTE_DIR/"

# Clean up the empty backup dir on the remote if nothing was written.
ssh "$REMOTE_USER_HOST" \
  "find '$REMOTE_DIR/.sync/conflicts/$TS' -type d -empty -delete 2>/dev/null || true"
