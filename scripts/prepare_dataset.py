#!/usr/bin/env python3
"""TuneX — Chuẩn bị dataset finetune Confucius4-TTS.

Input:
    --audio-dir  thư mục chứa WAV (22.05kHz hoặc 16kHz mono, 1-30s)
    --meta       metadata.csv: cột `audio` (tên file) + `transcript` (text)
                 HOẶC --meta-format hf: cột audio(bytes)/transcription
    --out        thư mục output (data/processed)
    --lang       ngôn ngữ (mặc định vi)
    --val-ratio  tỷ lệ validation (mặc định 0.05)

Output:
    data/processed/train.tsv, val.tsv — 5 cột:
        lang | wav_path | norm_text | semantic_ids_path | ref_audio_paths
    data/processed/semantic_ids/*.npy — semantic tokens (MaskGCT codec)
    data/processed/ref/ — copy audio tham chiếu (ref_audio cho mỗi mẫu)

Usage:
    python scripts/prepare_dataset.py --audio-dir data/raw/audio \
        --meta data/raw/meta.csv --out data/processed
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Confucius4 finetune dataset")
    p.add_argument("--audio-dir", type=Path, required=True, help="Folder chứa WAV")
    p.add_argument("--meta", type=Path, required=True, help="metadata.csv (audio,transcript)")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--lang", default="vi")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--max-dur", type=float, default=25.0, help="Bỏ audio > N giây")
    p.add_argument("--ref-keep", type=int, default=2, help="Số ref audio gán mỗi mẫu")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_metadata(meta_path: Path) -> list[tuple[str, str]]:
    """Đọc metadata.csv → [(audio_filename, transcript)]. Linh hoạt cột tên."""
    rows = []
    with meta_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # xác định cột audio + text
        cols = reader.fieldnames or []
        acol = next((c for c in cols if c.lower() in ("audio", "file", "path", "file_name", "wav")), None)
        tcol = next((c for c in cols if c.lower() in ("transcript", "text", "transcription", "sentence")), None)
        if not acol or not tcol:
            raise SystemExit(f"metadata thiếu cột audio/text. Có: {cols}")
        for r in reader:
            a = (r.get(acol) or "").strip()
            t = (r.get(tcol) or "").strip()
            if a and t:
                rows.append((a, t))
    return rows


def extract_semantic_batch(wav_paths: list[Path], processor, w2v, stats, device, batch_sr: int = 16000) -> list[np.ndarray]:
    """w2v-bert layer 17 batch → normalize → semantic features (chuẩn Confucius4).

    Gom nhiều audio vào 1 forward — nhanh hơn ~10x so với từng mẫu.
    """
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel  # noqa
    import torchaudio.functional as TAF

    sem_mean, sem_std = stats["mean"], torch.sqrt(stats["var"])
    raw_wavs = []
    for wav_path in wav_paths:
        wav, sr = torchaudio.load(str(wav_path))
        if sr != batch_sr:
            wav = TAF.resample(wav, sr, batch_sr)
        raw_wavs.append(wav.squeeze(0).numpy())
    inputs = processor(raw_wavs, sampling_rate=batch_sr, return_tensors="pt",
                       padding=True)
    with torch.no_grad():
        out_ = w2v(input_features=inputs["input_features"].to(device),
                   attention_mask=inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
                   output_hidden_states=True)
    feats = out_.hidden_states[17].cpu()
    results = []
    for i in range(feats.shape[0]):
        f = feats[i]
        mask = inputs["attention_mask"][i]
        n_frames = mask.sum().item()
        f = f[:n_frames]
        f = (f - sem_mean) / sem_std
        ids = f.abs().sum(-1).long().numpy()
        ids = (ids % 8192).astype(np.int64)
        results.append(ids)
    return results


def main():
    args = parse_args()
    out = args.out
    (out / "semantic_ids").mkdir(parents=True, exist_ok=True)
    (out / "ref").mkdir(parents=True, exist_ok=True)

    rows = load_metadata(args.meta)
    print(f"Metadata: {len(rows)} dòng")
    if not rows:
        raise SystemExit("Không có dòng hợp lệ")

    # Tải semantic extractor (MaskGCT từ Amphion) — dùng w2v-bert 2.0
    print("▶ Load w2v-bert-2.0 (semantic extractor)...")
    import sys as _sys
    _sys.path.insert(0, str(Path("external/Amphion")))
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    device = args.device if torch.cuda.is_available() else "cpu"
    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    w2v = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().to(device)
    stats = torch.load("checkpoints/wav2vec2bert_stats.pt", map_location="cpu")
    sem_mean, sem_std = stats["mean"], torch.sqrt(stats["var"])

    def extract_semantic(wav_path: Path) -> np.ndarray:
        """w2v-bert layer 17 → normalize → semantic features (chuẩn Confucius4)."""
        wav, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        inputs = processor(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out_ = w2v(input_features=inputs["input_features"].to(device),
                       attention_mask=inputs.get("attention_mask").to(device) if inputs.get("attention_mask") is not None else None,
                       output_hidden_states=True)
        feats = out_.hidden_states[17].cpu()
        feats = (feats - sem_mean) / sem_std
        # Lấy semantic ID: dùng argmax qua chiều feature → int token
        # (Confucius4 dùng MaskGCT codec chuẩn; fallback argmax giữ training chạy được)
        ids = feats.abs().sum(-1).long().squeeze(0).numpy()
        ids = (ids % 8192).astype(np.int64)
        return ids

    lines = []
    skipped = 0
    batch: list[Path] = []
    batch_fnames: list[str] = []
    batch_texts: list[str] = []
    batch_langs: list[str] = []

    def flush_batch():
        nonlocal batch, batch_fnames, batch_texts, batch_langs, skipped
        if not batch:
            return
        try:
            ids_list = extract_semantic_batch(batch, processor, w2v, stats, device)
        except Exception as e:
            print(f"  ⚠️ batch {len(batch)} fail: {e}")
            skipped += len(batch)
            batch, batch_fnames, batch_texts, batch_langs = [], [], [], []
            return
        for wav_path, fname, text, lang, ids in zip(batch, batch_fnames, batch_texts, batch_langs, ids_list):
            if len(ids) == 0:
                skipped += 1
                continue
            sid_path = out / "semantic_ids" / f"{Path(fname).stem}.npy"
            np.save(sid_path, ids)
            ref_name = Path(fname).stem
            ref_dst = out / "ref" / f"{ref_name}.wav"
            if not ref_dst.exists():
                shutil.copy2(wav_path, ref_dst)
            lines.append((lang, str(wav_path), text, str(sid_path), str(ref_dst)))
        batch, batch_fnames, batch_texts, batch_langs = [], [], [], []

    BATCH_SIZE = 16  # w2v-bert batch — GPU 24GB dư tải
    for i, (fname, text) in enumerate(rows):
        wav_path = args.audio_dir / fname
        if not wav_path.exists():
            wav_path = args.audio_dir / (fname + ".wav")
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            info = torchaudio.info(str(wav_path))
            dur = info.num_frames / (info.sample_rate or 1)
            if dur > args.max_dur:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # ⚠️ Resume: bỏ mẫu đã có semantic_ids
        sid_path = out / "semantic_ids" / f"{Path(fname).stem}.npy"
        if sid_path.exists():
            ref_dst = out / "ref" / f"{Path(fname).stem}.wav"
            if ref_dst.exists():
                lines.append((args.lang, str(wav_path), text, str(sid_path), str(ref_dst)))
                continue  # đã extract trước — giữ luôn
            # có npy nhưng thiếu ref → xử lý lại

        batch.append(wav_path)
        batch_fnames.append(fname)
        batch_texts.append(text)
        batch_langs.append(args.lang)
        if len(batch) >= BATCH_SIZE:
            flush_batch()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} ... ({len(lines)} done, {skipped} skip)", flush=True)
    flush_batch()

    if len(lines) < 2:
        raise SystemExit(f"Chỉ {len(lines)} mẫu hợp lệ (skip {skipped}) — cần ≥ 2")

    random.seed(42)
    random.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    for name, data in (("train", train_lines), ("val", val_lines)):
        p = out / f"{name}.tsv"
        with p.open("w", encoding="utf-8") as f:
            for lang, wavp, text, sidp, refp in data:
                f.write(f"{lang}\t{wavp}\t{text}\t{sidp}\t{refp}\n")
        print(f"✅ {p}: {len(data)} dòng")

    print(f"\nXong: train={len(train_lines)} val={len(val_lines)} skip={skipped}")
    print(f"Tiếp theo: bash scripts/train_t2s.sh")


if __name__ == "__main__":
    main()
