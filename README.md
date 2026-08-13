<div align="center">

# 🎛️ TuneX

**Toolkit fine-tune [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) — giọng tiếng Việt cho StoryCast**

[![GitHub](https://img.shields.io/badge/GitHub-Teedyyy--rm/TuneX-181717?logo=github)](https://github.com/Teedyyy-rm/TuneX)
[![Model](https://img.shields.io/badge/Base-Confucius4--TTS%20765M-blue)](https://huggingface.co/netease-youdao/Confucius4-TTS)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu126-EE4C2C?logo=pytorch&logoColor=white)]()
[![GPU](https://img.shields.io/badge/GPU-RTX%203090%2024GB-green)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow)]()
[![HF Auto-Push](https://img.shields.io/badge/HF%20Push-Auto%20mỗi%20epoch-FFD21E)]()

**1 lệnh setup → dataset → train T2S → train S2A → checkpoint tự push lên Hugging Face**

</div>

---

## ✨ Tính năng nổi bật

| 🚀 | Tính năng | Mô tả |
|---|---|---|
| ⚡ | **One-command setup** | `bash setup.sh` — clone repo + venv + torch + weights + Amphion |
| 🧬 | **Semantic codec CHUẨN** | [RepCodec / MaskGCT](https://huggingface.co/amphion/MaskGCT) — không phải hack `%8192` |
| 🔄 | **2-stage fine-tune** | T2S (Text→Semantic) + S2A (Semantic→Mel) |
| 📤 | **Auto-push HF** | Checkpoint đẩy lên repo private mỗi epoch (async — không nghẽn training) |
| 📊 | **Log UI đẹp** | Epoch bar + step bar + Loss/Acc/LR/Speed/ETA → `/root/tunex.log` |
| 🎙️ | **Test clone nhanh** | 1 lệnh tải HF → export → gen thử → nghe |

---

## 🧠 Kiến trúc

<div align="center">

| | | | | | | | |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 📝 **Text**<br/><sub>tiếng Việt</sub> | ➡️ | 🧠 **STAGE 1 — T2S**<br/><sub>Text → Semantic<br/>GPT-2 24L · 1280 dim</sub><br/><span style="color:#e74c3c;font-weight:bold">🔥 FINETUNE</span> | ➡️ | 🧠 **STAGE 2 — S2A**<br/><sub>Semantic → Mel<br/>Flow Matching</sub><br/><span style="color:#3498db;font-weight:bold">⏸️ pretrained → finetune</span> | ➡️ | 🎧 **Audio**<br/><sub>BigVGAN</sub> |

</div>

<div align="center">
<sub>🎙️ <b>Ref Audio</b> ──── CAMPPlus 192-dim speaker embedding ────▶ cả T2S & S2A</sub>
</div>

---

## 📁 Cấu trúc repo

```text
TuneX/
├── setup.sh                    # 1 lệnh: clone + venv + torch cu126 + weights
├── config.yaml                 # HF repo_id, push_every_steps, logging
├── configs/
│   ├── train_t2s.yaml          # Stage 1: text → semantic (LR 5e-5, bf16, 20 epochs)
│   └── train_s2a.yaml          # Stage 2: semantic → mel (LR 5e-5, load pretrained)
├── scripts/
│   ├── prepare_dataset.py      # audio + metadata → TSV + semantic ids (.npy)
│   ├── extract_semantic_codes.py  # ⭐ RepCodec chuẩn (MaskGCT codec)
│   ├── train_with_push.py      # ⭐ train + monitor + auto-push HF (async)
│   ├── export_checkpoint.py    # Lightning .ckpt → inference safetensors
│   ├── test_hf_checkpoint.py   # tải HF → export → gen thử (1 lệnh)
│   ├── test_clone.py           # test giọng sau finetune
│   └── log_ui.py               # UI log đẹp (epoch bar, step bar, GPU)
├── data/
│   ├── raw/                    # audio gốc + meta.csv (audio,transcript)
│   └── processed/              # train.tsv / val.tsv / semantic_ids/ / ref/
├── checkpoints/                # pretrained + semantic_codec/
├── external/Amphion/           # MaskGCT semantic codec (tự clone)
├── repo/Confucius4-TTS/        # code gốc (tự clone)
└── logs/                       # tensorboard + checkpoints training
```

---

## 🚀 Quickstart

### 1️⃣ Cài đặt

```bash
bash setup.sh
```

> ⚠️ **Yêu cầu:** GPU NVIDIA ≥ 24GB VRAM (đã test: RTX 3090), Python 3.10+.
> ⚠️ Nếu server mới: cài **torch 2.7.1+cu126 TRƯỚC** khi pip requirements (PyPI đã gỡ cudnn cu124).

### 2️⃣ Chuẩn bị dataset

```text
data/raw/
├── audio/
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
└── meta.csv        # audio,transcript
```

```bash
python scripts/prepare_dataset.py \
    --audio-dir data/raw/audio --meta data/raw/meta.csv \
    --out data/processed --lang vi
```

### 3️⃣ Train T2S (Stage 1)

```bash
bash scripts/train_t2s.sh        # = python scripts/train_with_push.py --stage t2s
```

### 4️⃣ Train S2A (Stage 2)

```bash
bash scripts/train_s2a.sh        # = python scripts/train_with_push.py --stage s2a
```

### 5️⃣ Test giọng

```bash
# Tải checkpoint mới nhất từ HF → export → gen thử
python scripts/test_hf_checkpoint.py \
    --text "Chuyện kể rằng, thuở xưa..." \
    --ref data/processed/ref/000001.wav \
    --out test.wav
```

---

## 🧬 Semantic Codec — tại sao chuẩn?

Confucius4 pretrained học token space của **RepCodec (MaskGCT)** — 8192 codebook. Pipeline chuẩn:

```text
Audio → w2v-bert layer 17 → normalize (stats) → RepCodec.quantize() → semantic codes
```

⚠️ **Đừng** dùng phép `abs().sum() % 8192` — model học chuỗi token vô nghĩa (loss giảm nhưng sai). Đã verify: Acc 0.8% (sai) → **25.7% (chuẩn)** ngay epoch 1.

---

## 📤 Auto-push Hugging Face

<div align="center">

```text
GPU Cloud (n1)
    │  train_with_push.py
    ├── training T2S (Lightning)
    ├── checkpoint mỗi epoch (620 steps)
    └── push async → HF (không block training)
             │
             ▼
    https://huggingface.co/Teedyyy-rm/Confucius4-storycast
```

</div>

```yaml
# config.yaml
hf:
  repo_id: "Teedyyy-rm/Confucius4-storycast"
  private: true
  push_every_steps: 620        # = 1 epoch
```

---

## 📊 Training config (đã tinh chỉnh)

<details>
<summary><b>📄 T2S — text → semantic (bấm để xem)</b></summary>

```yaml
data:
  batch_size: 8
  num_workers: 8
  sample_rate: 16000

training:
  precision: bf16-mixed        # 3090 (Ampere)
  epochs: 20
  gradient_clip: 1.0
  accumulate_grad_batches: 2   # effective batch = 16
  save_every_n_steps: 620      # = 1 epoch
  log_every_n_steps: 10

optimizer:
  optimizer_type: adamw
  learning_rate: 5.0e-5        # full finetune dataset nhỏ — chống catastrophic forgetting
  weight_decay: 0.01
  betas: [0.9, 0.95]

scheduler:
  scheduler_type: cosine
  num_warmup_steps: 1000
  num_training_steps: 12390
```

</details>

<details>
<summary><b>📄 S2A — semantic → mel (bấm để xem)</b></summary>

```yaml
paths:
  t2s_checkpoint: ./checkpoints/t2s_model.safetensors   # frozen backbone
  s2a_checkpoint: ./checkpoints/s2a_model.pt            # ⭐ load pretrained (không init mới)

training:
  precision: bf16-mixed
  epochs: 20
  accumulate_grad_batches: 2
  save_every_n_steps: 620

optimizer:
  learning_rate: 5.0e-5
```

</details>

---

## 📈 Kết quả thực tế (Ngọc Huyền, 11,316 mẫu)

| Metric | Codec sai (`%8192`) | **Codec chuẩn (RepCodec)** |
|---|:---:|:---:|
| Loss (epoch 1) | 8.97 | **3.02** |
| Acc (epoch 1) | 0.8% | **25.7%** |
| Acc (epoch 3) | — | **32.2%** |
| GPU | 100% | 100% (19-22GB) |
| Speed | 1.9 s/step | 1.9 s/step |

---

## 🛠️ Troubleshooting

| Lỗi | Giải pháp |
|---|---|
| `nvidia-cudnn-cu12==9.1.0.70` fail | Cài `torch 2.7.1+cu126` TRƯỚC requirements |
| OOM batch 16 | `accumulate_grad_batches: 2` + batch 8 (đã cấu hình) |
| Log UI đứng | `os.read()` fix — đảm bảo `train_with_push.py` mới nhất |
| Push không chạy | Check token `~/.cache/huggingface/token` + `push_every_steps` đúng với accumulate |
| Log không vào tunex.log | `log_ui.init(ui=True)` — `ui=False` làm mất file log! |

---

## 🗺️ Lộ trình

- [x] Semantic codec chuẩn (RepCodec)
- [x] Auto-push HF mỗi epoch (async)
- [x] Config tinh chỉnh (LR 5e-5, bf16, accumulate 2)
- [x] Test pipeline (tải → export → gen)
- [ ] S2A Stage 2 training
- [ ] Voicepack extraction
- [ ] Tích hợp StoryCast

---

<div align="center">

**Made with ❤️ by [Teedyyy](https://github.com/Teedyyy-rm) · phục vụ [StoryCast](https://github.com/Teedyyy-rm/StoryCast)**

<sub>Mọi log → `/root/tunex.log` · reload/restart append — `tail -f` không bao giờ mất follow</sub>

</div>
