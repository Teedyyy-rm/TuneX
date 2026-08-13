#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# TuneX — Train Kokoro Stage 1 trên GPU LỚN (2x L40 / 3090 24GB+)
# Dùng configs/kokoro_stage1_l40.yaml (batch 48, bf16-mixed)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d repo/Kokoro-Vietnamese ]; then
    echo "▶ Clone Kokoro-Vietnamese..."
    git clone https://github.com/contextboxai/Kokoro-Vietnamese.git repo/Kokoro-Vietnamese
fi
cd repo/Kokoro-Vietnamese

# Dataset Ngọc Huyền (nếu chưa có)
if [ ! -f training/vi/train_list.txt ]; then
    echo "▶ Setup dataset (11,315 mẫu từ /root/TuneX/data/raw/)..."
    # Tạo train_list.txt từ meta.csv — format: audio/NgocHuyenViVoice_xxx.wav|phonemes|0
    # (script chuẩn bị phonemes riêng — Kokoro cần phoneme, không phải text thường)
    echo "⚠️ Cần script prepare phonemes (xem README Kokoro-Vietnamese)"
fi

echo "▶ Stage 1 Kokoro (batch 48, bf16) — 2x L40..."
# 1 GPU: accelerate launch; 2 GPU: thêm --num_processes 2 --multi_gpu
accelerate launch train_first.py --config_path "$ROOT/configs/kokoro_stage1_l40.yaml" \
    ${NUM_GPUS:+--num_processes $NUM_GPUS} ${NUM_GPUS:+--multi_gpu}
