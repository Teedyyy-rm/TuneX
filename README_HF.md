---
license: apache-2.0
language:
- vi
tags:
- text-to-speech
- tts
- vietnamese
- finetuned
- confucius
- audiobook
- storycast
pipeline_tag: text-to-speech
base_model: netease-youdao/Confucius4-TTS
library_name: transformers
datasets:
- pnnbao-ump/ngochuyen_voice
- thangnzt/NgocHuyenViVoice
metrics:
- accuracy
---

<div align="center">

# 🎙️ Confucius4-StoryCast

**Fine-tuned Text-to-Semantic (T2S) checkpoint cho giọng kể chuyện tiếng Việt**

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Private%20Model-FFD21E)](https://huggingface.co/Teedyyy-rm/Confucius4-storycast)
[![Model](https://img.shields.io/badge/Base%20Model-Confucius4--TTS%20765M-blue)](https://huggingface.co/netease-youdao/Confucius4-TTS)
[![GPU](https://img.shields.io/badge/GPU-RTX%203090%2024GB-green)]()
[![Stage](https://img.shields.io/badge/Stage-T2S%20(Text%20→%20Semantic)-orange)]()

*Voice clone pipeline phục vụ [StoryCast](https://github.com/Teedyyy-rm) — dự án audiobook tiếng Việt tự động*

</div>

---

## 📖 Giới thiệu

Repo này lưu **checkpoint fine-tune** của model **Confucius4-TTS** cho một giọng đọc cụ thể (Ngọc Huyền), phục vụ pipeline sản xuất audiobook **StoryCast**.

Confucius4-TTS là engine TTS đa ngôn ngữ zero-shot, kiến trúc **2 giai đoạn**:

<div align="center">

| | | | | | | |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 📝 **Text**<br/><sub>tiếng Việt</sub> | ➡️ | 🧠 **STAGE 1 — T2S**<br/><sub>Text → Semantic<br/>GPT-2 24L · 1280 dim</sub><br/><span style="color:#e74c3c;font-weight:bold">🔥 FINETUNE</span> | ➡️ | 🧠 **STAGE 2 — S2A**<br/><sub>Semantic → Mel<br/>Flow Matching</sub><br/><span style="color:#3498db;font-weight:bold">⏸️ pretrained</span> | ➡️ | 🎧 **Audio**<br/><sub>BigVGAN</sub> |

</div>

<div align="center">
<sub>🎙️ <b>Ref Audio</b> (giọng cần clone) ──── CAMPPlus 192-dim speaker embedding ────▶ cả T2S & S2A</sub>
</div>

> ⚠️ **Repo hiện chứa checkpoint giai đoạn T2S** (Text → Semantic). Giai đoạn S2A (Semantic → Mel) sẽ được bổ sung sau.

---

## 🧬 Thông số kỹ thuật

<div align="center">

| Thành phần | Giá trị |
|---|---|
| **Base model** | [netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) |
| **Tổng tham số** | ~765M (1.2B với frozen) |
| **T2S architecture** | GPT-2, 24 layers, 1280 dim, 20 heads |
| **Semantic vocab** | 8,194 tokens (RepCodec / MaskGCT) |
| **Text vocab** | 32,000 |
| **Speaker embedding** | CAMPPlus 192-dim |
| **Ngôn ngữ** | Tiếng Việt (`vi`) |

</div>

---

## 🎯 Mục đích fine-tune

| Mục tiêu | Chi tiết |
|---|---|
| 🗣️ **Giọng đọc** | Ngọc Huyền — 11,316 mẫu (~25 giờ audio) |
| 📚 **Nội dung** | Truyện kể / audiobook tiếng Việt |
| 🎭 **Phong cách** | Kể chuyện chậm rãi, ấm áp, có ngữ điệu |
| 🔄 **Pipeline** | [TuneX](https://github.com/Teedyyy-rm/TuneX) — full fine-tune 2 stage |

---

## 📦 Cấu trúc checkpoint

```text
t2s/
├── step_000000620.ckpt   # Epoch 1 — checkpoint đầu tiên (data chuẩn RepCodec)
├── step_000001240.ckpt   # Epoch 2
├── ...
└── step_000012390.ckpt   # Epoch 20 (final)
```

> 🔬 Checkpoint là **Lightning .ckpt** (chứa optimizer state, ~6.5GB). Sử dụng [TuneX](https://github.com/Teedyyy-rm/TuneX) để export sang định dạng inference.

---

## 🚀 Cách sử dụng

### 1. Tải checkpoint

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download

# Cần token (repo private)
tok = open("~/.cache/huggingface/token").read().strip()

ckpt = hf_hub_download(
    "Teedyyy-rm/Confucius4-storycast",
    "t2s/step_000000620.ckpt",
    token=tok,
)
print(ckpt)
EOF
```

### 2. Export sang inference format

```bash
python scripts/export_checkpoint.py \
    --ckpt t2s/step_000000620.ckpt \
    --out checkpoints/t2s_finetuned.safetensors \
    --stage t2s
```

### 3. Generate giọng

```python
from confuciustts.cli.inference import ConfuciusTTS

model = ConfuciusTTS(config_path="config/inference.yaml", device="cuda")

audio = model.generate(
    text="Chuyện kể rằng, thuở xưa có một nàng công chúa...",
    lang="vi",
    prompt_wav="ref_audio.wav",   # ref giọng Ngọc Huyền
)
```

---

## 📊 Tiến trình training

<div align="center">

| Epoch | Steps | Loss | Accuracy | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 620 | 3.02 | 25.7% | ✅ Pushed |
| 2 | 1,240 | ~3.0 | ~25% | 🔄 Training |
| 3–20 | ... | ... | ... | ⏳ |

</div>

---

## 🛠️ Training config

<details>
<summary><b>📄 T2S — train_t2s.yaml (bấm để xem)</b></summary>

```yaml
data:
  batch_size: 8
  num_workers: 8
  sample_rate: 16000

training:
  precision: bf16-mixed        # RTX 3090 (Ampere)
  epochs: 20
  gradient_clip: 1.0
  accumulate_grad_batches: 2  # effective batch = 16
  save_every_n_steps: 620     # = 1 epoch
  log_every_n_steps: 10

optimizer:
  learning_rate: 5.0e-5       # full finetune, dataset nhỏ — chống catastrophic forgetting

scheduler:
  scheduler_type: cosine
  num_warmup_steps: 1000
  num_training_steps: 12390
```

</details>

---

## 🔍 So sánh hiệu quả

| Metric | Trước (semantic codec sai) | **Sau (RepCodec chuẩn)** |
|---|---|---|
| Loss (epoch 1) | 8.97 | **3.02** |
| Accuracy (epoch 1) | 0.8% | **25.7%** |

> 💡 Semantic IDs được trích bằng **RepCodec (MaskGCT)** — codec chuẩn của Confucius4, thay vì phép `%8192` tạm bợ.

---

## 🗺️ Lộ trình

- [x] Extract dataset 11,316 mẫu (RepCodec chuẩn)
- [x] T2S Stage 1 — training + auto-push HF
- [ ] T2S hoàn tất 20 epochs
- [ ] S2A Stage 2 (Semantic → Mel) — load pretrained
- [ ] Extract voicepack / test chất giọng
- [ ] Tích hợp StoryCast

---

<div align="center">

**Made with ❤️ by [Teedyyy](https://github.com/Teedyyy-rm) · [TuneX](https://github.com/Teedyyy-rm/TuneX)**

*Repo cập nhật tự động mỗi epoch qua `train_with_push.py`*

</div>
