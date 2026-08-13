# 🎛️ TuneX — Confucius4-TTS Finetune Toolkit

Bộ cài đặt **finetune Confucius4-TTS** (765M — T2S GPT-2 24L + S2A flow-matching) cho giọng tiếng Việt mới. Gọn gàng, 1 lệnh setup.

## 📁 Cấu trúc

```
TuneX/
├── setup.sh                  # 1 lệnh: clone repo + venv + deps + pretrained
├── configs/
│   ├── train_t2s.yaml        # Stage 1: text → semantic (LR 1e-4, 20 epochs)
│   └── train_s2a.yaml        # Stage 2: semantic → mel (LR 5e-5, 20 epochs)
├── scripts/
│   ├── prepare_dataset.py    # audio + metadata → TSV + semantic ids (.npy)
│   ├── train_t2s.sh          # chạy Stage 1
│   ├── train_s2a.sh          # chạy Stage 2
│   └── test_clone.py         # test giọng sau finetune (so sánh trước/sau)
├── data/
│   ├── raw/                  # audio gốc + metadata.csv (audio,transcript)
│   └── processed/            # train.tsv / val.tsv / semantic_ids/ / ref/
├── checkpoints/              # pretrained Confucius4 (tự tải trong setup)
├── pretrained/               # w2v-bert-2.0 + campplus (tự tải lúc chạy)
├── external/Amphion/         # MaskGCT semantic codec (tự clone)
├── repo/Confucius4-TTS/      # code gốc (tự clone)
└── logs/                     # tensorboard + checkpoints training
```

## 🚀 Quickstart

```bash
# 1. Cài đặt (clone + venv + torch cu124 + weights)
bash setup.sh

# 2. Chuẩn bị dataset
#    data/raw/audio/*.wav (16k/22.05k mono, 1-25s)
#    data/raw/meta.csv:   audio,transcript
python scripts/prepare_dataset.py \
    --audio-dir data/raw/audio --meta data/raw/meta.csv \
    --out data/processed --lang vi

# 3. Train 2 giai đoạn
bash scripts/train_t2s.sh    # Stage 1: text → semantic
bash scripts/train_s2a.sh    # Stage 2: semantic → mel

# 4. Test giọng mới
python scripts/test_clone.py --text "Xin chào, tôi là giọng mới." --ref data/processed/ref/xxx.wav
```

## ⚙️ Tinh chỉnh (chỉnh trong `configs/`)

| Tham số | V100-16GB | RTX 3090-24GB | V100-32GB |
|---|---|---|---|
| T2S `batch_size` | 4-8 | 8-16 | 16-32 |
| S2A `batch_size` | 8-16 | 16-24 | 24-32 |
| `precision` | `16-mixed` | `bf16-mixed` | `bf16-mixed` |
| `accumulate_grad_batches` | 2-4 | 2 | 1-2 |
| Epochs (1 giọng ~5-10h audio) | 10-20 | 10-20 | 10-20 |

- **V100 (Volta): KHÔNG có bf16 tensor core** → dùng `16-mixed` (fp16)
- **Ampere+ (3090/4090/A100):** dùng `bf16-mixed` (nhanh hơn)
- **OOM:** giảm `batch_size`, tăng `accumulate_grad_batches` (giữ effective batch)

## 📊 Ước tính thời gian (dataset ~10k mẫu, 20 epochs)

| GPU | T2S | S2A | Tổng |
|---|---|---|---|
| V100-16GB | 6-10h | 3-5h | ~10-15h |
| RTX 3090-24GB | 4-6h | 2-3h | ~6-9h |

## 📝 Dataset format (TSV, 5 cột, không header)

```
lang | wav_path | norm_text | semantic_ids_path | ref_audio_paths
vi   | /abs/xxx.wav | "nội dung câu" | /abs/semantic_ids/xxx.npy | /abs/ref/xxx.wav
```

- `semantic_ids_path`: .npy semantic tokens (pre-extract qua w2v-bert layer 17 → MaskGCT codec)
- `ref_audio_paths`: audio tham chiếu giọng (comma-separated nếu nhiều)
- Audio 1-30s (loader cắt 30s), prompt ref ≤ 15s

## ⚠️ Lưu ý

- **Zero-shot vẫn nên thử TRƯỚC khi finetune** — chỉ cần 1 file WAV 3-10s, không tốn GPU. Finetune khi zero-shot chưa đủ sát.
- Confucius4 **KHÔNG hỗ trợ LoRA** — full finetune 2 giai đoạn.
- Checkpoints training lưu ở `logs/t2s_train/` + `logs/s2a_train/` (Lightning .ckpt + safetensors).
