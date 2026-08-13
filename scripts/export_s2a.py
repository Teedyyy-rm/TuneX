#!/usr/bin/env python3
"""TuneX — Export S2A checkpoint (Lightning .ckpt) → inference .pt.

S2A KHÁC T2S:
- Training ckpt keys prefix 'model.' (không phải 's2a_model.')
- Inference load bằng torch.load() trực tiếp (file .pt dict)
- Có EMA model state (ema_model_state_dict) — chất lượng TỐT HƠN → ưu tiên dùng

Usage:
    python scripts/export_s2a.py \
        --ckpt s2a/step_0000413.ckpt \
        --out checkpoints/s2a_finetuned.pt \
        [--use-ema]   # mặc định True
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="Ưu tiên EMA weights (nếu có)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"▶ Load ckpt: {args.ckpt} ({args.ckpt.stat().st_size/1e9:.1f}GB)")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)

    # Ưu tiên EMA (chất lượng tốt hơn — S2A dùng EMA training)
    sd = None
    if args.use_ema and "ema_model_state_dict" in ckpt:
        sd = ckpt["ema_model_state_dict"]
        print("▶ Dùng EMA model state")
    if sd is None:
        sd = ckpt.get("state_dict", ckpt)

    # Strip prefix 'model.' hoặc 'ema_model.' → keys không prefix (inference expects)
    out = {}
    for k, v in sd.items():
        for prefix in ("ema_model.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        out[k] = v
    print(f"▶ Extract: {len(out)} keys (strip 'ema_model.'/'model.' prefix)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args.out))
    print(f"✅ Saved: {args.out} ({args.out.stat().st_size/1e6:.0f}MB)")
    # Verify: load lại + kích thước
    reloaded = torch.load(str(args.out), map_location="cpu", weights_only=False)
    print(f"   Verify load OK | {len(reloaded)} keys | sample: {list(reloaded)[:2]}")


if __name__ == "__main__":
    main()
