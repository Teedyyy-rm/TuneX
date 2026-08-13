#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# TuneX — Bộ cài đặt finetune Confucius4-TTS
# Aug 13 (Teedyy): clone repo + venv + deps + pretrained, 1 lệnh chạy
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TuneX — Confucius4-TTS Finetune Setup                   ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── 1. Clone repo code ────────────────────────────────────────────
if [ ! -d "repo/Confucius4-TTS" ]; then
    echo "▶ [1/5] Clone Confucius4-TTS..."
    mkdir -p repo
    git clone --depth 1 https://github.com/netease-youdao/Confucius4-TTS.git repo/Confucius4-TTS
else
    echo "▶ [1/5] Repo đã có (repo/Confucius4-TTS)"
fi

# ── 2. Venv + dependencies ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "▶ [2/5] Tạo venv + cài torch cu124..."
    python3 -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    ./.venv/bin/pip install --quiet torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "▶ [2/5] venv đã có"
fi

echo "▶ [2/5] Cài requirements..."
./.venv/bin/pip install --quiet -r repo/Confucius4-TTS/requirements.txt || \
    ./.venv/bin/pip install --quiet transformers==4.52.4 tokenizers safetensors \
        huggingface_hub sentencepiece pytorch-lightning==2.5.6 librosa soundfile \
        inflect regex jaconv pykakasi ema-pytorch wetext PyYAML tqdm packaging \
        filelock fsspec protobuf==3.19.6 matplotlib scipy tensorboard datasets==4.8.4

# ── 3. Pretrained models ──────────────────────────────────────────
mkdir -p checkpoints pretrained
echo "▶ [3/5] Tải pretrained Confucius4-TTS (HF)..."
./.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
import os
os.makedirs("checkpoints", exist_ok=True)
tok = None
if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
for f in ["t2s_model.safetensors", "s2a_model.pt", "wav2vec2bert_stats.pt",
          "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "config.json"]:
    p = hf_hub_download("netease-youdao/Confucius4-TTS", f, token=tok, local_dir="checkpoints")
    print(f"  ✅ {f} ({os.path.getsize(p)/1e6:.0f}MB)")
PY

# ── 4. Amphion MaskGCT (semantic codec — chuẩn bị data) ───────────
if [ ! -d "external/Amphion" ]; then
    echo "▶ [4/5] Clone Amphion MaskGCT (semantic codec)..."
    mkdir -p external
    git clone --depth 1 https://github.com/open-mmlab/Amphion.git external/Amphion
else
    echo "▶ [4/5] Amphion đã có"
fi

# ── 5. Dataset (từ config.yaml) ────────────────────────────────────
echo "▶ [5/6] Cài deps dataset (pandas/pyarrow)..."
./.venv/bin/pip install --quiet pandas pyarrow 2>/dev/null || true

if [ -f "config.yaml" ] && [ ! -f "data/raw/meta.csv" ]; then
    echo "▶ [5/6] Tải dataset HF (config.yaml → data/raw/)..."
    ./.venv/bin/python scripts/download_datasets.py --config config.yaml || \
        echo "⚠️ Tải dataset lỗi — xem lỗi trên (có thể chạy lại lệnh này sau)"
else
    echo "▶ [5/6] Dataset đã có (data/raw/meta.csv) hoặc không có config.yaml"
fi

# ── 6. Config paths ────────────────────────────────────────────────
echo "▶ [6/6] Trỏ config về thư mục TuneX..."
for f in configs/train_t2s.yaml configs/train_s2a.yaml; do
    sed -i "s|\./checkpoints|$ROOT/checkpoints|g; s|\./data|$ROOT/data|g; s|\./pretrained|$ROOT/pretrained|g; s|logs/|$ROOT/logs/|g" "$f" 2>/dev/null || true
done

echo ""
echo "✅ SETUP XONG!"
echo "   ├─ Repo code:   repo/Confucius4-TTS"
echo "   ├─ Venv:        .venv"
echo "   ├─ Weights:     checkpoints/"
echo "   ├─ Dataset:     data/raw/ (nếu config.yaml có datasets)"
echo "   └─ HF push:     Teedyyy-rm/Confucius4-storycast (tự động khi train)"
echo ""
echo "Bước tiếp theo:"
echo "   1. (nếu dataset chưa có) python scripts/download_datasets.py --config config.yaml"
echo "   2. python scripts/prepare_dataset.py --audio-dir data/raw/audio --meta data/raw/meta.csv --out data/processed"
echo "   3. bash scripts/train_t2s.sh   # tự push checkpoint lên HF"
echo "   4. bash scripts/train_s2a.sh"
