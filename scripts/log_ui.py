#!/usr/bin/env python3
"""TuneX — UI log đẹp cho training (khung viền chữ nhật, ASCII).

- Mọi log ghi CẢ stdout + file log (mặc định /root/train.log) để `tail -f` theo dõi.
- ⚠️ KHÔNG BAO GIỜ rm file log khi relaunch — append separator, tail -f vẫn follow
  (bài học Kokoro: rm làm tail -f mất follow → user phải chạy lại lệnh).
- Dùng: from log_ui import log, log_banner, log_section, log_table, progress_bar

Cấu hình (config.yaml):
    logging:
      log_file: /root/train.log
      ui: true
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

_LOG_FILE: Path | None = None
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def display_width(s: str) -> int:
    """Tính độ rộng hiển thị thực tế trên terminal cho chuỗi chứa ASCII, Emoji, Unicode Wide/Fullwidth, ANSI codes."""
    clean_s = _ANSI_RE.sub('', s)
    w = 0
    for c in clean_s:
        if unicodedata.category(c) in ('Mn', 'Me', 'Cf') or c in ('\ufe0f', '\ufe0e'):
            continue
        ea = unicodedata.east_asian_width(c)
        if ea in ('W', 'F'):
            w += 2
        else:
            w += 1
    return w


def init(log_file: str = "/root/tunex.log", ui: bool = True) -> None:
    """Khởi tạo log file. KHÔNG rm — mở append (giữ tail -f follow)."""
    global _LOG_FILE
    if not ui:
        _LOG_FILE = None
        return
    _LOG_FILE = Path(log_file).expanduser()
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str = "") -> None:
    """Ghi 1 dòng (có timestamp) → stdout + log file (append)."""
    # Không prefix timestamp cho dòng ASCII-art (bắt đầu bằng ký tự khung viền/icon)
    _no_ts = {"╔", "╚", "║", "═", "┌", "└", "│", "├", "┬", "┴", "┐", "┘", "─", "━",
              "▶", "✅", "❌", "⚠", "📤", "🎉", "🔁", "⏳", " "}
    line = msg if (not msg or msg[0] in _no_ts) else f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_banner(title: str) -> None:
    """Banner lớn — ╔══ ... ══╗"""
    dw = display_width(title)
    width = max(dw + 4, 60)
    pad = width - 4 - dw
    log("╔" + "═" * (width - 2) + "╗")
    log(f"║  {title}{' ' * pad}║")
    log("╚" + "═" * (width - 2) + "╝")


def log_section(num: int, total: int, title: str) -> None:
    """Section — ╔══ [1/5] TÊN ══╗"""
    log("")
    log("━" * 64)
    log(f"  ╔══ [{num}/{total}] {title} ══╗")
    log("━" * 64)


def log_table(rows: list[tuple[str, str]]) -> None:
    """Bảng 2 cột — ┌──┬──┐ / │ │ │ / └──┴──┘"""
    if not rows:
        return
    w1 = max(display_width(k) for k, _ in rows) + 2
    w2 = max(display_width(v) for _, v in rows) + 2
    sep = "┌" + "─" * w1 + "┬" + "─" * w2 + "┐"
    end = "└" + "─" * w1 + "┴" + "─" * w2 + "┘"
    log(sep)
    for k, v in rows:
        pad1 = w1 - display_width(k) - 1
        pad2 = w2 - display_width(v) - 1
        log(f"│ {k}{' ' * pad1}│ {v}{' ' * pad2}│")
    log(end)


def progress_bar(pct: float, width: int = 40) -> str:
    """Thanh tiến trình — ████░░░░ (pct 0-100)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def epoch_bar(epoch: int, total: int, label: str = "EPOCH", width: int = 32) -> str:
    """Epoch bar — ══ EPOCH 5/20 ████░░ 25% ══"""
    pct = epoch / total * 100
    bar = progress_bar(pct, width)
    return f"  ══ {label} {epoch}/{total} {bar} {pct:.0f}% ══"


def log_restart_separator(reason: str = "") -> None:
    """Separator khi relaunch — APPEND vào log cũ (không rm → tail -f giữ follow)."""
    log("")
    log("═" * 64)
    log(f"🔁 RELAUNCH {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {reason}")
    log("═" * 64)


import shutil
import subprocess

def format_seconds(seconds: float) -> str:
    """Định dạng giây thành H:M:S dễ đọc (ví dụ: 1h 23m 45s)."""
    if seconds < 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def get_sys_info() -> dict[str, str]:
    """Lấy thông tin hệ thống (Python, PyTorch, CUDA, RAM, Ổ đĩa)."""
    info = {
        "Python": sys.version.split()[0],
        "PyTorch": "-",
        "CUDA": "-",
        "RAM": "-",
        "Free Disk": "-"
    }
    # PyTorch & CUDA
    try:
        import torch
        info["PyTorch"] = torch.__version__
        info["CUDA"] = f"Available ({torch.version.cuda})" if torch.cuda.is_available() else "No CUDA"
    except Exception:
        pass
    
    # RAM
    if os.path.exists('/proc/meminfo'):
        try:
            mem = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].split()[0])
            tot_ram = mem.get('MemTotal', 0) / (1024 * 1024)
            avail_ram = mem.get('MemAvailable', 0) / (1024 * 1024)
            info["RAM"] = f"{avail_ram:.1f}GB / {tot_ram:.1f}GB"
        except Exception:
            pass

    # Disk space
    try:
        total, used, free = shutil.disk_usage('.')
        info["Free Disk"] = f"{free / (1024**3):.1f}GB / {total / (1024**3):.1f}GB"
    except Exception:
        pass

    return info


def log_sys_info() -> None:
    """In bảng thông tin hệ thống chi tiết khi bắt đầu chạy."""
    sys_data = get_sys_info()
    rows = [(k, v) for k, v in sys_data.items()]
    log_table(rows)


def log_gpu_info(gpu_name: str = "", vram_used: str = "", vram_total: str = "",
                 util: str = "", temp: str = "") -> None:
    """Bảng GPU — tự động query nvidia-smi nếu không truyền tham số."""
    if not gpu_name:
        try:
            cmd = ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits']
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                parts = [x.strip() for x in r.stdout.strip().split(',')]
                if len(parts) >= 5:
                    gpu_name = parts[0]
                    vram_used = f"{float(parts[1])/1024:.1f}GB" if parts[1].isdigit() else f"{parts[1]}MB"
                    vram_total = f"{float(parts[2])/1024:.1f}GB" if parts[2].isdigit() else f"{parts[2]}MB"
                    util = f"{parts[3]}%"
                    temp = parts[4]
        except Exception:
            pass

    box_width = 62
    log("┌" + "─" * (box_width + 2) + "┐")
    
    header = "🎧 TuneX · Confucius4-TTS Finetune"
    dw_h = display_width(header)
    left_pad = (box_width - dw_h) // 2
    right_pad = box_width - dw_h - left_pad
    log(f"│ {' ' * left_pad}{header}{' ' * right_pad} │")
    log("├" + "─" * (box_width + 2) + "┤")
    
    if gpu_name:
        gpu_str = f"📊 GPU: {gpu_name}"
        pad_g = max(0, box_width - display_width(gpu_str))
        log(f"│ {gpu_str}{' ' * pad_g} │")
    if vram_used and vram_total:
        vram_str = f"💻 VRAM: {vram_used}/{vram_total} | GPU Util: {util} | Temp: {temp}°C"
        pad_v = max(0, box_width - display_width(vram_str))
        log(f"│ {vram_str}{' ' * pad_v} │")
    log("└" + "─" * (box_width + 2) + "┘")


def log_step_details(step: int, total_steps: int, pct: float, loss_info: str = "",
                     lr: str = "", speed: str = "", eta: str = "") -> None:
    """In card chi tiết tiến trình step, loss, lr, speed và thời gian còn lại (ETA)."""
    bar = progress_bar(pct, width=28)
    log(f"  ▸ Step {step}/{total_steps} {bar} {pct:.1f}%")
    details = []
    if loss_info:
        details.append(f"Loss: {loss_info}")
    if lr:
        details.append(f"LR: {lr}")
    if speed:
        details.append(f"Speed: {speed}")
    if eta:
        details.append(f"ETA: {eta}")
    if details:
        log(f"    └─ 📈 {' | '.join(details)}")


if __name__ == "__main__":
    # Demo — chạy thử: python log_ui.py
    init("/tmp/tunex_demo.log")
    log_banner("TuneX — Confucius4-TTS Finetune (Enhanced Info)")
    
    log_section(1, 3, "SYSTEM & ENVIRONMENT INFO")
    log_sys_info()
    
    log_section(2, 3, "STAGE 1 — T2S (text → semantic)")
    log_table([
        ("GPU", "RTX 3090 24GB"),
        ("Batch", "8 × accumulate 2 = effective 16"),
        ("Precision", "bf16-mixed"),
        ("Epochs", "20"),
        ("HF repo", "Teedyyy-rm/Confucius4-storycast"),
    ])
    log("")
    log(epoch_bar(7, 20))
    log_step_details(340, 4189, pct=8.1, loss_info="Mel 0.241 | Gen 1.102 | SLM 0.054", lr="1e-4", speed="3.2 it/s", eta=format_seconds(1202))
    log("")
    log_gpu_info()
    
    log_section(3, 3, "RELAUNCH MONITOR")
    log_restart_separator("(demo)")
    print("\n✅ Demo xong — xem /tmp/tunex_demo.log")


