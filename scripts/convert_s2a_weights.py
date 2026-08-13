#!/usr/bin/env python3
"""TuneX — Convert S2A finetuned (plain weight, đã remove weight_norm) → inference format.

Training Lightning lưu state sau remove_weight_norm (257 keys, weight plain).
Pretrained/inference model dùng weight_norm (274 keys: weight_g + weight_v).

Chuyển đổi: weight = weight_g * normalize(weight_v, dim=1)  (Conv1d)
→ weight_g = weight.norm(2, dim=1, keepdim=True)
→ weight_v = weight / weight_g

Usage:
    python scripts/convert_s2a_weights.py \
        --pt checkpoints/s2a_finetuned_epoch1.pt \
        --out checkpoints/s2a_finetuned_epoch1_infer.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=Path, required=True, help="S2A pt (đã strip prefix)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    sd = torch.load(str(args.pt), map_location="cpu", weights_only=False)
    out = {}
    n_converted = 0
    for k, v in sd.items():
        if k.endswith(".conv.weight") and v.dim() == 3:  # Conv1d weight_norm
            # ⚠️ Chuẩn pretrained: g = norm(dim=(1,2)) → (out,1,1); v = w/g → (out,in,k)
            g = v.norm(2, dim=(1, 2), keepdim=True).clamp_min(1e-12)
            vv = v / g
            out[k + "_g"] = g
            out[k + "_v"] = vv
            n_converted += 1
        else:
            out[k] = v
    torch.save(out, str(args.out))
    print(f"✅ Saved: {args.out} ({len(out)} keys, {n_converted} weight_norm tách)")
    print("   Verify:", [k for k in out if "res_skip_layers.0.conv" in k])


if __name__ == "__main__":
    main()
