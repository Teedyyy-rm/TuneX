#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# TuneX — Train Stage 2: S2A (semantic → mel) + auto-push checkpoint HF
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "▶ S2A Training (Stage 2) + auto-push HF..."
exec ./.venv/bin/python scripts/train_with_push.py --stage s2a
