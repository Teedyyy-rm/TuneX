#!/usr/bin/env bash
# TuneX — Chuyển từ T2S sang S2A sau khi T2S đạt (Teedyy Aug 13).
# Khi epoch 6 xong (step_000003720.ckpt):
#   1. Export T2S finetuned epoch 6 -> safetensors (frozen backbone cho S2A)
#   2. Sua train_s2a.yaml: t2s_checkpoint -> finetuned epoch 6
#   3. Kill T2S training
#   4. Chay S2A training
# Usage (tren n1): bash scripts/switch_to_s2a.sh
set -e
cd /root/TuneX

echo "=== 1. Export T2S finetuned (epoch 6) -> safetensors ==="
CKPT=$(ls -t logs/t2s_train/ckpt/step_*.ckpt | head -1)
echo "   Checkpoint moi nhat: $CKPT"
/root/TuneX/.venv/bin/python scripts/export_checkpoint.py \
    --ckpt "$CKPT" \
    --out checkpoints/t2s_finetuned_epoch6.safetensors \
    --stage t2s

echo "=== 2. Sua config S2A: t2s_checkpoint -> finetuned epoch 6 ==="
sed -i "s|t2s_checkpoint: ./checkpoints/t2s_model.safetensors.*|t2s_checkpoint: ./checkpoints/t2s_finetuned_epoch6.safetensors   # frozen backbone FINETUNED (giu giong)|" \
    configs/train_s2a.yaml
grep "t2s_checkpoint" configs/train_s2a.yaml

echo "=== 3. Kill T2S training ==="
ps aux | grep -E "confuciustts.cli.train_t2s|train_with_push.*t2s" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
echo "   Da kill T2S"

echo "=== 4. Chay S2A training ==="
nohup /root/TuneX/.venv/bin/python scripts/train_with_push.py --stage s2a \
    > /tmp/s2a_train.log 2>&1 &
echo "   S2A PID: $!"
sleep 10
tail -5 /tmp/s2a_train.log
