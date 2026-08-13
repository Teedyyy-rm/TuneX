#!/usr/bin/env python3
"""TuneX — Push checkpoint lên HF repo (tự động trong training).

Cơ chế: mỗi N steps (đọc từ config.yaml → hf.push_every_steps) script này
được gọi để upload checkpoint mới nhất lên repo HF.

Usage:
    python scripts/push_checkpoint.py --stage t2s|s2a \
        --file logs/t2s_train/lightning_logs/.../model.safetensors \
        [--step 5000] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["t2s", "s2a"])
    p.add_argument("--file", required=True, help="Path checkpoint cần push")
    p.add_argument("--step", type=int, default=0, help="Step hiện tại (đặt tên)")
    p.add_argument("--epoch", type=int, default=0, help="Epoch hiện tại (đặt tên)")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf_cfg = cfg.get("hf", {})
    repo_id = hf_cfg["repo_id"]
    private = hf_cfg.get("private", True)

    tok = None
    tf = os.path.expanduser(hf_cfg.get("token_file", "~/.cache/huggingface/token"))
    if os.path.exists(tf):
        tok = open(tf).read().strip()
    if not tok:
        raise SystemExit("Không có HF token — tạo repo thủ công: huggingface-cli repo create ...")

    api = HfApi(token=tok)
    # Đảm bảo repo tồn tại (create nếu chưa có — private theo config)
    try:
        api.repo_info(repo_id, repo_type="model", token=tok)
    except Exception:
        print(f"⚠️ Repo {repo_id} chưa có — tạo mới (private={private})...")
        api.create_repo(repo_id, repo_type="model", private=private, token=tok)

    fpath = Path(args.file)
    if not fpath.exists():
        raise SystemExit(f"Checkpoint không có: {fpath}")

    name = f"{args.stage}/step_{args.step:07d}"
    if args.epoch:
        name = f"{args.stage}/epoch_{args.epoch:03d}_step_{args.step:07d}"
    name += fpath.suffix

    print(f"📤 Push {fpath.name} → {repo_id}:{name} ({fpath.stat().st_size/1e6:.0f}MB)...")
    url = api.upload_file(
        path_or_fileobj=str(fpath),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="model",
        token=tok,
    )
    print(f"✅ OK: {url}")


if __name__ == "__main__":
    main()
