#!/usr/bin/env python3
"""TuneX — Train + tự động push checkpoint lên HF.

Chạy train_t2s/train_s2a (Lightning) và monitor thư mục checkpoint:
mỗi khi có file .ckpt mới → push lên HF repo (config.yaml → hf.repo_id).

Usage:
    python scripts/train_with_push.py --stage t2s [--config config.yaml]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from log_ui import log, log_banner, log_section, log_table, log_restart_separator, epoch_bar, progress_bar, format_seconds, log_step_details  # noqa: E402


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


def _handle_line(line: str, LIGHTNING_RE, args, cfg, last_render_step: int, start_time: float) -> int:
    """Xử lý 1 dòng stdout → render UI. Trả về last_render_step mới."""
    from log_ui import log, epoch_bar, log_step_details  # noqa: F401
    m = LIGHTNING_RE.search(line)
    if m:
        ep = int(m.group(1))
        pct = int(m.group(2))
        step = int(m.group(3))
        step_total = int(m.group(4))
        # ⚠️ Dedup: Lightning in progress bar 2 lần cho cùng step → chỉ render khi step MỚI
        if step == last_render_step:
            return last_render_step
        last_render_step = step
        ep_total = int(cfg["training"].get("t2s_epochs" if args.stage == "t2s" else "s2a_epochs", 20))
        ep_line = epoch_bar(min(ep + 1, ep_total), ep_total, label="EPOCH")

        elapsed = time.time() - start_time
        speed_val = step / elapsed if elapsed > 0 else 0
        eta_val = (step_total - step) / speed_val if speed_val > 0 else 0
        speed_str = f"{speed_val:.1f} step/s" if speed_val >= 1 else f"{1/speed_val:.1f} s/step" if speed_val > 0 else "-"
        eta_str = format_seconds(eta_val)

        # Extra loss metrics — T2S: train_loss_step | S2A: train/flow_loss_step
        loss_parts = []
        lm = re.search(r"(?:train_loss_step|train/flow_loss_step|flow_loss_step|flow_loss)=([0-9.]+)", line)
        if lm: loss_parts.append(f"Loss {lm.group(1)}")
        lg = re.search(r"train_acc_step=([0-9.]+)", line)
        if lg: loss_parts.append(f"Acc {lg.group(1)}")
        loss_info = " | ".join(loss_parts)

        lr_match = re.search(r"lr=([0-9.e-]+)", line, re.IGNORECASE)
        lr_str = lr_match.group(1) if lr_match else ""

        log(ep_line)
        log_step_details(step, step_total, pct=pct, loss_info=loss_info, lr=lr_str, speed=speed_str, eta=eta_str)
    else:
        # Bỏ progress bar trùng lặp của Lightning (dòng chứa it/s hoặc s/it + %)
        if re.search(r"\dit/s|\ds/it", line) or ("%" in line and "it/s" in line):
            return last_render_step
        log(line)
    return last_render_step


def _poll_tensorboard(ckpt_dir: Path, args, cfg, last_tb_step: int, start_time: float, resume_base: int = 0, speed_state: dict = None) -> int:
    """Đọc TensorBoard events (Lightning luôn ghi train_loss_step mỗi step).

    Dùng khi stdout im lặng (Rich progress bar tắt khi non-TTY sau resume).
    Trả về step mới nhất đã render.

    ⚠️ Aug 15 (Teedyy báo ETA sai): speed tính theo DELTA giữa 2 lần poll
    (speed_state) — KHÔNG dùng elapsed từ start (bao gồm thời gian load model
    ~2 phút chưa train → speed thấp → ETA phóng đại 15h thay vì ~6h).
    """
    from log_ui import log, epoch_bar, log_step_details  # noqa: F401
    try:
        import yaml as _yaml
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = sorted(
            (ckpt_dir / "t2s_training").glob("version_*/events.out.tfevents.*")
            if (ckpt_dir / "t2s_training").exists()
            else ckpt_dir.glob("events.out.tfevents.*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not events:
            return last_tb_step
        ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 10000})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        # T2S: train_loss_step | S2A: train/flow_loss_step
        loss_tag = next((t for t in tags if t in ("train_loss_step", "train/flow_loss_step", "flow_loss_step", "flow_loss")), None)
        if not loss_tag:
            return last_tb_step
        evs = ea.Scalars(loss_tag)
        if not evs:
            return last_tb_step
        e = evs[-1]
        if e.step <= last_tb_step:
            return last_tb_step
        last_tb_step = e.step

        # ⚠️ step_total = save_every_n_steps THẬT từ config (T2S 620 / S2A 413 — optimizer steps)
        stage_yaml = ROOT / "configs" / ("train_t2s.yaml" if args.stage == "t2s" else "train_s2a.yaml")
        try:
            step_total = int(_yaml.safe_load(stage_yaml.read_text(encoding="utf-8"))["training"]["save_every_n_steps"])
        except Exception:
            step_total = 620 if args.stage == "t2s" else 413
        ep_total = int(cfg["training"].get("t2s_epochs" if args.stage == "t2s" else "s2a_epochs", 20))
        total_steps = step_total * ep_total
        ep = e.step // step_total  # global_step / steps-per-epoch → epoch index thật
        pct = int((e.step % step_total) / step_total * 100) if step_total else 0
        log(epoch_bar(min(ep + 1, ep_total), ep_total, label="EPOCH"))
        loss_parts = [f"Loss {e.value:.4f}"]
        acc_tag = next((t for t in tags if t == "train_acc_step"), None)
        if acc_tag:
            a = ea.Scalars(acc_tag)[-1]
            loss_parts.append(f"Acc {a.value:.4f}")
        elapsed = time.time() - start_time
        # ⚠️⚠️ Aug 15 (ETA sai): speed theo DELTA cửa sổ ≥5 steps (anchor) — KHÔNG dùng
        # elapsed từ start (gồm ~2 phút load model chưa train → ETA phóng đại).
        # Cửa sổ 5 steps → ổn định hơn delta 1-step/3s (nhiễu). Lần đầu chỉ seed.
        now_t = time.time()
        if speed_state is not None and speed_state.get("seeded"):
            d_steps = e.step - speed_state.get("step", e.step)
            d_time = now_t - speed_state.get("t", now_t)
            if d_steps >= 5 and d_time > 1:
                speed_val = d_steps / d_time
                # cập nhật anchor — chỉ khi đủ 5 steps mới
                speed_state.update(step=e.step, t=now_t)
            else:
                speed_val = speed_state.get("last_speed", 0.0)  # giữ speed cũ nếu chưa đủ window
        else:
            speed_val = 0.0
            if speed_state is not None:
                speed_state.update(seeded=True, step=e.step, t=now_t, last_speed=0.0)
        if speed_state is not None and speed_val > 0:
            speed_state["last_speed"] = speed_val
        speed_str = f"{speed_val:.1f} step/s" if speed_val >= 1 else f"{1/speed_val:.1f} s/step" if speed_val > 0 else "-"
        eta_val = (total_steps - e.step) / speed_val if speed_val > 0 else 0
        # ⚠️ Hiển thị step GLOBAL thật (vd 3799/12390) — không modulo theo epoch
        log_step_details(e.step, total_steps, pct=pct,
                         loss_info=" | ".join(loss_parts), lr="", speed=speed_str,
                         eta=format_seconds(eta_val))
    except Exception as ex:
        log(f"⚠️ TB poll: {ex}")
    return last_tb_step


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf_cfg = cfg.get("hf", {})
    push_every = args.push_every_steps or hf_cfg.get("push_every_steps", 2000)

    # ── Log UI ──
    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("log_file", "/root/tunex.log")
    from log_ui import init as log_ui_init, log_sys_info, log_gpu_info, log_step_details, format_seconds
    log_ui_init(log_file, ui=log_cfg.get("ui", True))
    # ⚠️ Relaunch: APPEND separator (KHÔNG rm) — tail -f /root/tunex.log vẫn follow
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
    import os as _os
    _env = dict(_os.environ)
    # ⚠️ confuciustts nằm trong repo/Confucius4-TTS — cần PYTHONPATH
    _repo_pkg = ROOT / "repo" / "Confucius4-TTS"
    if _repo_pkg.exists():
        _env["PYTHONPATH"] = str(_repo_pkg) + _os.pathsep + _env.get("PYTHONPATH", "")
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
                            bufsize=0, env=_env)  # ⚠️ binary mode — đọc chunk bytes, tách \r/\n
    pushed: set[str] = set()
    # ⚠️ Aug 15: file ckpt CÓ SẴN lúc launch (resume source tải từ HF, đã push ở run trước)
    # → thêm vào pushed ngay — tránh push lại 6.5GB mỗi restart
    # ⚠️⚠️ Dedup theo STEP NUMBER (không theo path): Lightning khi start tạo
    # step_000003720.ckpt (9 số) — cùng step 3720, content giống step_0003720.ckpt đã push
    if ckpt_dir.exists():
        for p in ckpt_dir.rglob("*.ckpt"):
            if not p.name.startswith("last"):
                pushed.add(str(p))
    pushed_steps: set[int] = set()
    for p in list(pushed):
        m = re.search(r"step[=_-](\d+)", Path(p).name)
        if m:
            pushed_steps.add(int(m.group(1)))
    pushing: set[str] = set()
    # ⚠️ resume_base = global_step từ ckpt có sẵn (resume source) — để tính speed steps MỚI
    resume_base = 0
    for p in list(pushed):
        try:
            import torch as _torch
            ck = _torch.load(str(p), map_location="cpu")
            resume_base = max(resume_base, int(ck.get("global_step", 0)))
        except Exception:
            pass
    # ⚠️ Aug 15: speed_state — delta speed giữa 2 lần poll TB (ETA chính xác)
    speed_state: dict = {"seeded": False, "step": resume_base, "t": time.time()}
    last_step_line = ""
    start_time = time.time()
    # ⚠️ Aug 15: chỉ poll TB khi stdout IM LẶNG (không render step mới ≥ 60s).
    # Trước đó poll định kỳ 30s → stdout parser + TB poll render CÙNG step → log in 2 lần (Teedyy báo)
    last_stdout_render = time.time()

    # Đọc stdout realtime → render UI đẹp + theo dõi checkpoint
    # ⚠️ Aug 13: đọc BINARY + tách cả \r (Lightning progress bar dùng \r, không \n
    # sau resume từ last.ckpt → readline() kẹt mãi → log đứng, GPU vẫn chạy)
    LIGHTNING_RE = re.compile(
        r"Epoch (\d+):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?,\s*([0-9.]+)(it/s|s/it),.*?(?:train_loss_step=([0-9.]+))?"
    )
    LOG_EVERY = 1  # ⚠️ Teedyy: render MỖI step, nhưng DEDUP — 1 step chỉ render 1 lần (Lightning in 2 dòng/step)
    last_render_step = -1

    out_buf = b""
    out = proc.stdout
    import select
    last_tb_poll = 0.0
    last_tb_step = -1
    while True:
        # ⚠️ Đọc stdout với timeout — nếu Lightning im lặng (Rich non-TTY sau resume)
        # thì poll TensorBoard events (nguồn loss/step LUÔN có)
        rlist, _, _ = select.select([out], [], [], 3)
        # ⚠️⚠️ Aug 15: poll TB khi stdout im ≥ 3s — render MỌI step mới
        # (dedup bằng last_tb_step) → mỗi step đúng 1 dòng. select timeout 3s
        # (không 10s) → delta time speed chính xác (10s làm d_time phóng đại → ETA sai).
        # stdout parser render chuẩn 1 lần/step; TB poll chỉ là fallback khi resume "0/?"
        now = time.time()
        if now - last_stdout_render > 3:
            last_tb_step = _poll_tensorboard(ckpt_dir, args, cfg, last_tb_step, start_time, resume_base, speed_state)
            # Push checkpoint theo step (async — không block training)
            _maybe_push(ckpt_dir, args.stage, hf_cfg, push_every, pushed, pushing, pushed_steps)
        if not rlist:
            continue
        chunk = os.read(out.fileno(), 4096) if out else b""  # ⚠️ os.read: trả data CÓ SẴN (FileIO raw vì bufsize=0) — read() block
        if not chunk:
            if out_buf.strip():
                _handle_line(out_buf.decode("utf-8", errors="replace"), LIGHTNING_RE, args, cfg,
                             last_render_step, start_time)
            break
        out_buf += chunk
        # Tách theo \n hoặc \r — progress bar update bằng \r
        while b"\n" in out_buf or b"\r" in out_buf:
            nl = out_buf.find(b"\n")
            cr = out_buf.find(b"\r")
            if nl == -1:
                idx = cr + 1
            elif cr == -1:
                idx = nl + 1
            else:
                idx = min(nl, cr) + 1
            raw_line = out_buf[:idx]
            out_buf = out_buf[idx:]
            _prev_step = last_render_step
            last_render_step = _handle_line(
                raw_line.decode("utf-8", errors="replace").rstrip("\r\n"),
                LIGHTNING_RE, args, cfg, last_render_step, start_time,
            )
            # stdout đang render step mới → hoãn TB poll (tránh in 2 lần cùng step)
            if last_render_step != _prev_step:
                last_stdout_render = time.time()
        # Push checkpoint theo step (async — không block training)
        _maybe_push(ckpt_dir, args.stage, hf_cfg, push_every, pushed, pushing, pushed_steps)
    proc.wait()
    _maybe_push(ckpt_dir, args.stage, hf_cfg, push_every, pushed, pushing, pushed_steps)

    rc = proc.returncode
    log_section(3, 3, "KẾT THÚC")
    if rc == 0:
        log("✅ Training hoàn tất!")
    else:
        log(f"❌ Training fail (rc={rc})")
        log(f"   Log cuối: {last_step_line or '(trống)'}")
    sys.exit(rc)


def _maybe_push(ckpt_dir: Path, stage: str, hf_cfg: dict, push_every: int, pushed: set, pushing: set, pushed_steps: set = None):
    """Push checkpoint mới nhất chưa push (theo step number trong tên).

    ⚠️ Aug 13: chạy ASYNC (thread) — push 6.5GB mất 10-20 phút, nếu đồng bộ
    sẽ block vòng đọc stdout → training nghẹt (pipe đầy, GPU 0%).
    """
    if not ckpt_dir.exists():
        return
    if pushed_steps is None:
        pushed_steps = set()
    # Lightning lưu: step_N.ckpt, last.ckpt, last-v1.ckpt, epoch=... .ckpt
    # ⚠️ Aug 13: lọc MỌI file bắt đầu "last" (last.ckpt, last-v1.ckpt...) — trước chỉ lọc "last.ckpt" → push nhầm last-v1!
    ckpts = sorted(
        [p for p in ckpt_dir.rglob("*.ckpt")
         if not p.name.startswith("last")],
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        return
    latest = ckpts[-1]
    if str(latest) in pushed or str(latest) in pushing:
        return

    # Lấy step từ tên file (step_000123.ckpt | epoch=1-step=123.ckpt)
    import re
    m = re.search(r"step[=_-](\d+)", latest.name)
    step = int(m.group(1)) if m else 0
    if step and step % push_every != 0:
        return  # chưa tới mốc push
    if step and step in pushed_steps:
        pushed.add(str(latest))  # đánh dấu đã push — không push lại bản trùng step
        return

    pushing.add(str(latest))
    log(f"📤 Push {latest.name} → HF ({hf_cfg.get('repo_id')})... (async)")

    def _do_push():
        import traceback
        try:
            r = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / "push_checkpoint.py"),
                 "--stage", stage, "--file", str(latest), "--step", str(step),
                 "--config", str(ROOT / "config.yaml")],
                capture_output=True, text=True, timeout=3600,  # 6.5GB cần 10-20 phút
            )
            if r.returncode == 0:
                pushed.add(str(latest))
                log(f"✅ {r.stdout.strip().splitlines()[-1]}")
            else:
                log(f"⚠️ Push fail: {r.stderr[-300:]}")
        except Exception as e:
            log(f"⚠️ Push exception: {e}")
        finally:
            pushing.discard(str(latest))

    import threading
    threading.Thread(target=_do_push, daemon=True).start()


if __name__ == "__main__":
    main()
