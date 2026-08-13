#!/usr/bin/env python3
"""TuneX — Export checkpoint Lightning (.ckpt) → inference safetensors.

Training lưu:  checkpoint['state_dict'] = {'t2s_model.<k>': v, ...optimizer...}
Inference cần: safetensors với keys KHÔNG prefix (ConfuciusTTS load vào t2s_model).

Usage:
    python scripts/export_checkpoint.py \
        --ckpt logs/t2s_train/ckpt/step_000001239.ckpt \
        --out checkpoints/t2s_finetuned.safetensors \
        [--stage t2s]   # t2s | s2a (mặc định t2s)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import safetensors.torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt (đã tải từ HF)")
    p.add_argument("--out", type=Path, required=True, help="Output .safetensors")
    p.add_argument("--stage", default="t2s", choices=["t2s", "s2a"])
    return p.parse_args()


def main():
    args = parse_args()
    print(f"▶ Load ckpt: {args.ckpt} ({args.ckpt.stat().st_size/1e9:.1f}GB)")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    prefix = f"{args.stage}_model."
    keys = {k.replace(prefix, ""): v for k, v in sd.items() if k.startswith(prefix)}
    if not keys:
        # Thử không prefix (nếu ai đó đã export)
        keys = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    print(f"▶ Extract: {len(keys)} keys (prefix '{prefix}')")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(keys, str(args.out))
    print(f"✅ Saved: {args.out} ({args.out.stat().st_size/1e6:.0f}MB)")
    # Verify: đọc lại + kích thước tensor lớn nhất
    reload = safetensors.torch.load_file(str(args.out))
    biggest = max(reload.items(), key=lambda kv: kv[1].numel())
    print(f"   Verify load OK | tensor lớn nhất: {biggest[0]} {tuple(biggest[1].shape)}")


if __name__ == "__main__":
    main()
