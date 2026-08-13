#!/usr/bin/env python3
"""TuneX — Test clone giọng sau finetune (dùng checkpoint mới).

So sánh trước/sau finetune với cùng ref audio.

Usage:
    python scripts/test_clone.py --text "câu test" \
        --ref data/processed/ref/xxx.wav \
        [--t2s-checkpoint logs/t2s_train/.../model.safetensors]
        [--s2a-checkpoint logs/s2a_train/.../model.pt]
        [--out test_output.wav]
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
    p.add_argument("--text", default="Xin chào, đây là giọng sau khi finetune.")
    p.add_argument("--ref", type=Path, default=Path("data/processed/ref"))
    p.add_argument("--t2s-checkpoint", type=Path, default=None,
                   help="Checkpoint T2S sau finetune (nếu có)")
    p.add_argument("--s2a-checkpoint", type=Path, default=None,
                   help="Checkpoint S2A sau finetune (nếu có)")
    p.add_argument("--out", type=Path, default=Path("test_output.wav"))
    p.add_argument("--lang", default="vi")
    return p.parse_args()


def main():
    args = parse_args()

    # Ref audio: file đầu tiên trong thư mục (hoặc chính nó)
    if args.ref.is_dir():
        wavs = sorted(args.ref.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"Không có wav trong {args.ref}")
        ref = wavs[0]
    else:
        ref = args.ref
    print(f"Ref audio: {ref}")

    # Load checkpoint finetune nếu có → ghi đè config paths
    import yaml
    cfg_path = ROOT / "configs" / "inference_config.yaml"
    if not cfg_path.exists():
        # dùng config gốc của repo
        cfg_path = ROOT / "repo" / "Confucius4-TTS" / "config" / "inference_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.t2s_checkpoint:
        cfg["paths"]["t2s_checkpoint"] = str(args.t2s_checkpoint)
    if args.s2a_checkpoint:
        cfg["paths"]["s2a_checkpoint"] = str(args.s2a_checkpoint)
    # paths tuyệt đối
    cfg["paths"]["tokenizer_path"] = str((ROOT / "checkpoints").resolve())
    cfg["paths"]["w2v_stat"] = str((ROOT / "checkpoints" / "wav2vec2bert_stats.pt").resolve())
    cfg["paths"]["t2s_checkpoint"] = str((ROOT / cfg["paths"]["t2s_checkpoint"]).resolve() if not Path(cfg["paths"]["t2s_checkpoint"]).is_absolute() else cfg["paths"]["t2s_checkpoint"])
    cfg["paths"]["s2a_checkpoint"] = str((ROOT / cfg["paths"]["s2a_checkpoint"]).resolve() if not Path(cfg["paths"]["s2a_checkpoint"]).is_absolute() else cfg["paths"]["s2a_checkpoint"])
    tmp_cfg = ROOT / "configs" / "inference_test.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg))

    import torch
    import torchaudio
    from confuciustts.cli.inference import ConfuciusTTS

    model = ConfuciusTTS(config_path=str(tmp_cfg), device="cuda" if torch.cuda.is_available() else "cpu")
    audio = model.generate(text=args.text, lang=args.lang, prompt_wav=str(ref), verbose=True)
    torchaudio.save(str(args.out), audio.cpu(), model.sample_rate)
    print(f"✅ Saved: {args.out} ({audio.shape[-1]/model.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
