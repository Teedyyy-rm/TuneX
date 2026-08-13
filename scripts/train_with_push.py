#!/usr/bin/env python3
"""TuneX — Train + tự động push checkpoint lên HF.

Chạy train_t2s/train_s2a (Lightning) và monitor thư mục checkpoint:
mỗi khi có file .ckpt mới → push lên HF repo (config.yaml → hf.repo_id).

Usage:
    python scripts/train_with_push.py --stage t2s [--config config.yaml]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from log_ui import log, log_banner, log_section, log_table, log_restart_separator, epoch_bar, progress_bar  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["t2s", "s2a"])
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--push-every-steps", type=int, default=0,
                   help="override hf.push_every_steps (0 = theo config)")
    return p.parse_args()


def _gpu_name() -> str:
    """Lấy tên GPU (nvidia-smi) — fallback '-'."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "-"
    except Exception:
        return "-"


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf_cfg = cfg.get("hf", {})
    push_every = args.push_every_steps or hf_cfg.get("push_every_steps", 2000)

    # ── Log UI ──
    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("log_file", "/root/train.log")
    from log_ui import init as log_ui_init, log_sys_info, log_gpu_info, log_step_details, format_seconds
    log_ui_init(log_file, ui=log_cfg.get("ui", True))
    # ⚠️ Relaunch: APPEND separator (KHÔNG rm) — tail -f /root/train.log vẫn follow
    log_restart_separator(f"— {args.stage.upper()} — {Path(args.config).name}")

    log_banner("TuneX — Confucius4-TTS Finetune")
    log_section(1, 3, "SYSTEM & CONFIG INFO")
    log_sys_info()
    log_table([
        ("Stage", args.stage.upper()),
        ("GPU", _gpu_name()),
        ("HF repo", f"{hf_cfg.get('repo_id', '-')} (private={hf_cfg.get('private', True)})"),
        ("Push HF", f"mỗi {push_every} steps"),
        ("Log file", str(log_file)),
    ])
    log("")

    # Ghi đè configs/*.yaml theo config.yaml (epochs, batch, precision, lr)
    stage_cfg = cfg["training"]
    import shutil
    for name in ("train_t2s.yaml", "train_s2a.yaml"):
        src = ROOT / "configs" / name
        dst = ROOT / "configs" / f"{name}.tunex"
        shutil.copy2(src, dst)

    # Chạy train qua subprocess (tee log)
    if args.stage == "t2s":
        cmd = [str(ROOT / ".venv/bin/python"), "-m", "confuciustts.cli.train_t2s",
               "-c", str(ROOT / "configs" / "train_t2s.yaml")]
        ckpt_dir = ROOT / "logs" / "t2s_train"
    else:
        cmd = [str(ROOT / ".venv/bin/python"), "-m", "confuciustts.cli.train_s2a",
               "-c", str(ROOT / "configs" / "train_s2a.yaml")]
        ckpt_dir = ROOT / "logs" / "s2a_train"

    log_section(2, 3, f"STAGE {'T2S' if args.stage == 't2s' else 'S2A'} — TRAINING")
    log(f"▶ {' '.join(map(str, cmd))}")
    log(f"▶ ckpt dir: {ckpt_dir}")
    log("")

    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    pushed: set[str] = set()
    last_step_line = ""
    start_time = time.time()

    # Đọc stdout realtime → render UI đẹp + theo dõi checkpoint
    out = proc.stdout or sys.stdout
    for raw in out:
        line = raw.rstrip("\n")
        # Bắt dòng Lightning: "Epoch [3/20], Step [340/4189], Mel Loss: ..."
        m = re.search(r"Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\]", line)
        if m:
            ep, ep_total, step, step_total = map(int, m.groups())
            ep_line = epoch_bar(ep, ep_total, label="EPOCH")
            st_pct = step / step_total * 100 if step_total else 0
            
            elapsed = time.time() - start_time
            speed_val = step / elapsed if elapsed > 0 else 0
            eta_val = (step_total - step) / speed_val if speed_val > 0 else 0
            speed_str = f"{speed_val:.1f} step/s" if speed_val >= 1 else f"{1/speed_val:.1f} s/step" if speed_val > 0 else "-"
            eta_str = format_seconds(eta_val)

            # Extra loss metrics
            loss_parts = []
            lm = re.search(r"Mel Loss: ([0-9.]+)", line)
            if lm: loss_parts.append(f"Mel {lm.group(1)}")
            lg = re.search(r"Gen Loss: ([0-9.]+)", line)
            if lg: loss_parts.append(f"Gen {lg.group(1)}")
            ls = re.search(r"SLM Loss: ([0-9.]+)", line)
            if ls: loss_parts.append(f"SLM {ls.group(1)}")
            loss_info = " | ".join(loss_parts)

            lr_match = re.search(r"lr: ([0-9.e-]+)", line, re.IGNORECASE)
            lr_str = lr_match.group(1) if lr_match else ""

            log(ep_line)
            log_step_details(step, step_total, pct=st_pct, loss_info=loss_info, lr=lr_str, speed=speed_str, eta=eta_str)
            last_step_line = line
        else:
            # Bỏ dòng progress bar trùng lặp của Lightning (100%|...)
            if "it/s]" in line or "s/it]" in line:
                continue
            log(line)

        # Push checkpoint theo step
        _maybe_push(ckpt_dir, args.stage, hf_cfg, push_every, pushed)
    proc.wait()
    _maybe_push(ckpt_dir, args.stage, hf_cfg, push_every, pushed)

    rc = proc.returncode
    log_section(3, 3, "KẾT THÚC")
    if rc == 0:
        log("✅ Training hoàn tất!")
    else:
        log(f"❌ Training fail (rc={rc})")
        log(f"   Log cuối: {last_step_line or '(trống)'}")
    sys.exit(rc)


def _maybe_push(ckpt_dir: Path, stage: str, hf_cfg: dict, push_every: int, pushed: set):
    """Push checkpoint mới nhất chưa push (theo step number trong tên)."""
    if not ckpt_dir.exists():
        return
    # Lightning lưu: step_N.ckpt, last.ckpt, epoch=... .ckpt
    ckpts = sorted(
        [p for p in ckpt_dir.rglob("*.ckpt")
         if not any(x in p.name for x in ("last.ckpt",))],
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        return
    latest = ckpts[-1]
    if str(latest) in pushed:
        return

    # Lấy step từ tên file (step_000123.ckpt | epoch=1-step=123.ckpt)
    import re
    m = re.search(r"step[=_-](\d+)", latest.name)
    step = int(m.group(1)) if m else 0
    if step and step % push_every != 0:
        return  # chưa tới mốc push

    print(f"📤 Push {latest.name} → HF ({hf_cfg.get('repo_id')})...", flush=True)
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / "push_checkpoint.py"),
         "--stage", stage, "--file", str(latest), "--step", str(step),
         "--config", str(ROOT / "config.yaml")],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode == 0:
        pushed.add(str(latest))
        print(f"✅ {r.stdout.strip().splitlines()[-1]}", flush=True)
    else:
        print(f"⚠️ Push fail: {r.stderr[-300:]}", flush=True)


if __name__ == "__main__":
    main()
