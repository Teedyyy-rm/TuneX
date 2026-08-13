#!/usr/bin/env python3
"""TuneX — Test checkpoint finetune từ HF (máy chính).

1. Tải checkpoint mới nhất từ Teedyyy-rm/Confucius4-storycast
2. Export Lightning .ckpt → inference safetensors
3. Gen thử câu test → WAV

Usage:
    python scripts/test_hf_checkpoint.py \
        --text "Câu test..." \
        --ref /path/ref.wav \
        [--stage t2s] \
        [--out test.wav]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repo" / "Confucius4-TTS"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Xin chào, đây là giọng sau khi finetune. Hôm nay trời đẹp quá.")
    p.add_argument("--ref", type=Path, default=Path("/home/obito/projects/StoryCast/assets/voices/teedyy-voice/ref_audio_0-7s.wav"))
    p.add_argument("--stage", default="t2s", choices=["t2s", "s2a"])
    p.add_argument("--repo", default="Teedyyy-rm/Confucius4-storycast")
    p.add_argument("--out", type=Path, default=Path("/home/obito/projects/Models/Confucius4-TTS/test_finetuned.wav"))
    p.add_argument("--lang", default="vi")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Lấy file mới nhất từ HF
    import os
    from huggingface_hub import HfApi
    tok = None
    tf = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(tf):
        tok = open(tf).read().strip()
    api = HfApi(token=tok)
    files = api.list_repo_files(args.repo, repo_type="model", token=tok)
    stage_files = [f for f in files if f.startswith(f"{args.stage}/") and f.endswith(".ckpt")]
    if not stage_files:
        print(f"⚠️ Chưa có checkpoint {args.stage} trên HF. Files:", files)
        return
    # Chọn mới nhất (tên step_XXXXXXX → số lớn nhất)
    stage_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    latest = stage_files[-1]
    print(f"▶ Checkpoint mới nhất: {latest}")

    local = api.hf_hub_download(args.repo, latest, repo_type="model", token=tok)
    print(f"▶ Tải về: {local} ({os.path.getsize(local)/1e9:.1f}GB)")

    # 2. Export → safetensors
    from scripts.export_checkpoint import main as export_main
    sys.argv = ["export_checkpoint.py", "--ckpt", local,
                "--out", str(ROOT / "checkpoints" / f"t2s_finetuned_{latest.split('_')[-1].split('.')[0]}.safetensors"),
                "--stage", args.stage]
    export_main()

    # 3. Gen thử — dùng config local nhưng override t2s_checkpoint
    out_safetensors = [p for p in (ROOT / "checkpoints").glob("t2s_finetuned_*.safetensors")]
    if not out_safetensors:
        raise SystemExit("Export fail — không tìm thấy safetensors")
    finetuned = out_safetensors[-1]
    print(f"\n▶ Gen thử với checkpoint: {finetuned}")

    import yaml
    cfg_path = Path("/home/obito/projects/Confucius4-TTS/config/inference_config_local.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["paths"]["t2s_checkpoint"] = str(finetuned)  # override T2S → finetuned
    tmp_cfg = Path("/tmp/inference_test_finetuned.yaml")
    tmp_cfg.write_text(yaml.safe_dump(cfg))

    import torch
    import torchaudio
    from confuciustts.cli.inference import ConfuciusTTS

    model = ConfuciusTTS(config_path=str(tmp_cfg), device="cuda" if torch.cuda.is_available() else "cpu")
    audio = model.generate(text=args.text, lang=args.lang, prompt_wav=str(args.ref), verbose=True)
    torchaudio.save(str(args.out), audio.cpu(), model.sample_rate)
    print(f"✅ Saved: {args.out} ({audio.shape[-1]/model.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
