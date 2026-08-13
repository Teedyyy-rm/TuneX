#!/usr/bin/env python3
"""TuneX — Convert kokoro_vi.pth (KModel) → StyleTTS2 training format.

Verify key-by-key TRƯỚC khi train (bài học: load được nhưng missing keys
im lặng = checkpoint hỏng). Theo recipe iamdinhthuan/Kokoro-Vietnamese:

    kokoro_vi.pth (KModel: bert/bert_encoder/predictor/text_encoder/decoder)
        ↓ strip "module." prefix
    kokoro_vi_base.pth ({net: {...}}) ← StyleTTS2 load_checkpoint format

Usage:
    python scripts/convert_kokoro_vi.py \
        --input training/kokoro_vi.pth \
        --output training/kokoro_vi_base.pth \
        [--verify]   # thử load vào T2S model thật → in missing/unexpected keys
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

GROUPS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def strip_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("training/kokoro_vi.pth"))
    p.add_argument("--output", type=Path, default=Path("training/kokoro_vi_base.pth"))
    p.add_argument("--verify", action="store_true", help="Verify load vào model thật")
    args = p.parse_args()

    raw = torch.load(str(args.input), map_location="cpu", weights_only=False)

    # 1. Kiểm tra top-level keys — phải đủ 5 nhóm
    missing = [g for g in GROUPS if g not in raw]
    if missing:
        raise SystemExit(f"❌ Thiếu nhóm keys: {missing} — file KHÔNG phải KModel kokoro_vi.pth")
    print(f"✅ 5 nhóm keys đủ: {list(raw.keys())}")

    # 2. Kiểm tra prefix module.
    for g in GROUPS:
        keys = list(raw[g].keys())
        has = any(k.startswith("module.") for k in keys[:10])
        print(f"  {g}: {len(keys)} keys | module. prefix: {'có' if has else 'không'}")
        if not has:
            print(f"  ⚠️ {g} không có prefix module. — kiểm tra thủ công")

    # 3. Convert
    net = {g: strip_prefix(raw[g]) for g in GROUPS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net}, str(args.output))
    print(f"✅ Saved: {args.output} ({args.output.stat().st_size/1e6:.0f}MB)")

    # 4. Verify — load vào model thật, in missing/unexpected keys
    if args.verify:
        print("\n=== VERIFY load vào StyleTTS2 model ===")
        import sys
        sys.path.insert(0, str(Path("StyleTTS2").resolve()))
        from models import build_model
        config_path = Path("configs/config_voice_stage1.yml")
        if not config_path.exists():
            config_path = Path("repo/Confucius4-TTS/config/inference_config.yaml")  # placeholder
            print("⚠️ config_voice_stage1.yml chưa có — bỏ qua verify model")
            return
        import yaml
        config = yaml.safe_load(config_path.read_text())
        model = build_model(config["model_params"])
        # Load từng nhóm, in missing/unexpected
        total_missing = 0
        for g in GROUPS:
            sd = net[g]
            target = getattr(model, g)
            result = target.load_state_dict(sd, strict=False)
            if result.missing_keys:
                print(f"  ❌ {g}: {len(result.missing_keys)} missing keys")
                print(f"     {result.missing_keys[:3]}")
                total_missing += len(result.missing_keys)
            if result.unexpected_keys:
                print(f"  ⚠️ {g}: {len(result.unexpected_keys)} unexpected keys")
            else:
                print(f"  ✅ {g}: load OK (0 missing, 0 unexpected)")
        if total_missing:
            raise SystemExit(f"❌ Có {total_missing} missing keys — KHÔNG nên train với checkpoint này")
        print("\n✅ VERIFY PASS — checkpoint an toàn để train")


if __name__ == "__main__":
    main()
