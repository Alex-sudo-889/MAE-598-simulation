#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$ROOT_DIR/robot_sim"
BACKUP_DIR="$ROOT_DIR/backups"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "robot_sim directory not found at $SOURCE_DIR" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$BACKUP_DIR/robot_sim-$STAMP.tar.gz"

tar -czf "$ARCHIVE" -C "$ROOT_DIR" "robot_sim"

echo "Backup saved to $ARCHIVE"
