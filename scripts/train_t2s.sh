#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# TuneX — Train Stage 1: T2S (text → semantic) + auto-push checkpoint HF
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "▶ T2S Training (Stage 1) + auto-push HF..."
exec ./.venv/bin/python scripts/train_with_push.py --stage t2s
