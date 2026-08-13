#!/usr/bin/env python3
"""TuneX — Extract semantic codes CHUẨN (RepCodec MaskGCT) cho Confucius4 finetune.

⚠️ THAY THẾ phép `abs().sum()%8192` SAI — đúng theo source Confucius4:
    w2v-bert layer 17 features → normalize (stats) → RepCodec.quantize() → codes

Source: confuciustts/frontend/semantic_extractor.py (SemanticCodec class)
        Amphion models/codec/kmeans/repcodec_model.py (RepCodec)

Usage:
    python scripts/extract_semantic_codes.py \
        --audio-dir data/raw/audio --out data/processed/semantic_ids \
        [--device cuda] [--batch 16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
AMPHION = ROOT / "external" / "Amphion"
sys.path.insert(0, str(AMPHION))
sys.path.insert(0, str(ROOT / "scripts"))

# ⚠️ MỌI log → /root/tunex.log (Teedyy mandate) — KHÔNG ghi file riêng
# ⚠️ ui=True BẮT BUỘC — ui=False làm _LOG_FILE=None (log() chỉ in stdout, KHÔNG ghi file!)
try:
    from log_ui import log, init as _log_ui_init
    _log_ui_init("/root/tunex.log", ui=True)
except Exception:
    def log(msg=""):
        import sys as _sys
        _sys.stdout.write(str(msg) + "\n")
        _sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed/semantic_ids"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--codec-ckpt", type=Path,
                   default=Path("checkpoints/semantic_codec/model.safetensors"))
    return p.parse_args()


def build_models(args):
    """Load w2v-bert extractor + RepCodec semantic codec (chuẩn Confucius4)."""
    from types import SimpleNamespace
    import json

    from models.codec.kmeans.repcodec_model import RepCodec
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    device = args.device if torch.cuda.is_available() else "cpu"

    # w2v-bert (semantic features layer 17)
    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    w2v = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().to(device)
    stats = torch.load("checkpoints/wav2vec2bert_stats.pt", map_location="cpu")
    sem_mean, sem_std = stats["mean"], torch.sqrt(stats["var"])

    # RepCodec — cfg từ maskgct.json (semantic_codec section)
    mcfg = json.load(open(AMPHION / "models/tts/maskgct/config/maskgct.json"))
    sc = mcfg["model"]["semantic_codec"]
    codec_cfg = SimpleNamespace(**sc)
    codec = RepCodec(cfg=codec_cfg).eval().to(device)
    import safetensors.torch
    safetensors.torch.load_model(codec, str(args.codec_ckpt))
    log(f"✅ Codec loaded: codebook_size={sc['codebook_size']}")

    return processor, w2v, sem_mean, sem_std, codec, device


def extract_codes_batch(wav_paths, processor, w2v, sem_mean, sem_std, codec, device, batch_sr=16000):
    """w2v-bert layer 17 → normalize → RepCodec.quantize → codes (chuẩn)."""
    import torchaudio.functional as TAF
    raw_wavs = []
    for wp in wav_paths:
        wav, sr = torchaudio.load(str(wp))
        if sr != batch_sr:
            wav = TAF.resample(wav, sr, batch_sr)
        raw_wavs.append(wav.squeeze(0).numpy())
    inputs = processor(raw_wavs, sampling_rate=batch_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out_ = w2v(input_features=inputs["input_features"].to(device),
                   attention_mask=inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
                   output_hidden_states=True)
    feats = out_.hidden_states[17].cpu()  # (B, T, D)
    feats = (feats - sem_mean) / sem_std
    with torch.no_grad():
        codes, _ = codec.quantize(feats.to(device))  # (B, T) — semantic codes!
    results = []
    for i, wp in enumerate(wav_paths):
        mask = inputs["attention_mask"][i]
        n_frames = mask.sum().item()
        ids = codes[i][:n_frames].cpu().numpy().astype(np.int64)
        results.append(ids)
    return results


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(args.audio_dir.glob("*.wav"))
    log(f"Audio: {len(wavs)} files")
    if not wavs:
        raise SystemExit("Không có wav")

    processor, w2v, mean, std, codec, device = build_models(args)

    done = 0
    for i in range(0, len(wavs), args.batch):
        batch = wavs[i:i + args.batch]
        ids_list = extract_codes_batch(batch, processor, w2v, mean, std, codec, device)
        for wp, ids in zip(batch, ids_list):
            np.save(args.out / f"{wp.stem}.npy", ids)
        done += len(batch)
        if done % 200 == 0 or done == len(wavs):
            log(f"  {done}/{len(wavs)} ...")
    log(f"✅ XONG: {done} semantic code files → {args.out}")


if __name__ == "__main__":
    main()
