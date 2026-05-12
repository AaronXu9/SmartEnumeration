#!/bin/bash
# Pull remote -> local (one-way, --update means mtime-wins).
set -euo pipefail

LOCAL_DIR="/mnt/katritch_lab2/pocketInformedV-SYNTHES"
REMOTE_USER_HOST="aoxu@discovery.usc.edu"
REMOTE_DIR="/project2/katritch_223/aoxu/pocketInformedV-SYNTHES"

SYNC_DIR="$LOCAL_DIR/.sync"
TS="${SYNC_TS:-$(date +%Y%m%d-%H%M%S)}"
LOG="$SYNC_DIR/logs/pull-$TS.log"
BACKUP_REL=".sync/conflicts/$TS/from-remote"

mkdir -p "$SYNC_DIR/logs" "$LOCAL_DIR/$BACKUP_REL"

echo "[$(date +%H:%M:%S)] PULL  $REMOTE_USER_HOST:$REMOTE_DIR/  ->  $LOCAL_DIR/"

rsync -ah \
  --update \
  --partial \
  --info=progress2,stats2 \
  --human-readable \
  --exclude-from="$SYNC_DIR/exclude.txt" \
  --backup \
  --backup-dir="$BACKUP_REL" \
  --log-file="$LOG" \
  "$REMOTE_USER_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

# Remove the backup dir if nothing was written to it (keeps .sync/conflicts tidy).
find "$LOCAL_DIR/$BACKUP_REL" -type d -empty -delete 2>/dev/null || true
find "$LOCAL_DIR/.sync/conflicts/$TS" -type d -empty -delete 2>/dev/null || true
