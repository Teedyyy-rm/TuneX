#!/usr/bin/env python3
"""TuneX — Tải dataset HF (parquet thủ công) → data/raw/audio + metadata.csv.

Đọc config.yaml → tải từng dataset → ghi WAV + metadata.csv (audio,transcript).
Giống cơ chế train_kokoro.py (tránh torchcodec — đọc parquet thủ công).

Usage:
    python scripts/download_datasets.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out-audio", default=None, help="override data/raw/audio")
    p.add_argument("--out-meta", default=None, help="override data/raw/meta.csv")
    p.add_argument("--limit", type=int, default=0, help="giới hạn mẫu (test)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    audio_dir = Path(args.out_audio or "data/raw/audio")
    meta_path = Path(args.out_meta or "data/raw/meta.csv")
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — tránh fail nếu thiếu deps lúc setup sớm
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files

    tok = None
    tf = os.path.expanduser(cfg.get("hf", {}).get("token_file", "~/.cache/huggingface/token"))
    if os.path.exists(tf):
        tok = open(tf).read().strip()
    os.environ.setdefault("HF_TOKEN", tok or "")

    rows = []
    for ds in cfg["datasets"]:
        repo = ds["repo"]
        text_col = ds["text_col"]
        audio_col = ds.get("audio_col", "audio")
        lang = ds.get("lang", "vi")
        print(f"▼ {repo} (text={text_col}, audio={audio_col})", flush=True)

        files = [f for f in list_repo_files(repo, repo_type="dataset", token=tok)
                 if f.startswith("data/") and f.endswith(".parquet")]
        if not files:
            files = [f for f in list_repo_files(repo, repo_type="dataset", token=tok)
                     if f.endswith(".parquet")]
        print(f"  {len(files)} parquet files", flush=True)

        n = 0
        for fname in files:
            local = hf_hub_download(repo, fname, repo_type="dataset", token=tok)
            df = pd.read_parquet(local)
            for i, row in df.iterrows():
                text = str(row.get(text_col) or "").strip()
                if not text:
                    continue
                a = row.get(audio_col)
                if isinstance(a, dict) and a.get("bytes"):
                    try:
                        data, sr = sf.read(io.BytesIO(a["bytes"]))
                    except Exception:
                        continue
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if len(data) < int(0.5 * sr):  # < 0.5s — bỏ
                        continue
                    # Tên file ổn định theo repo + index
                    stem = f"{repo.split('/')[-1]}_{n:06d}"
                    wav_path = audio_dir / f"{stem}.wav"
                    try:
                        sf.write(wav_path, data, sr)
                    except Exception:
                        continue
                    rows.append((wav_path.name, text, lang))
                    n += 1
                    if args.limit and n >= args.limit:
                        break
            if args.limit and n >= args.limit:
                break
        print(f"  ✅ {n} samples", flush=True)

    # Ghi metadata.csv
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "transcript", "lang"])
        for name, text, lang in rows:
            w.writerow([name, text, lang])
    print(f"\n✅ XONG: {len(rows)} samples → {audio_dir} + {meta_path}")


if __name__ == "__main__":
    main()
