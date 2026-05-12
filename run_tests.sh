#!/usr/bin/env bash
# Runs the stdlib unittest suite. Works on both macOS (local) and KatLab.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m unittest discover -s tests -v
