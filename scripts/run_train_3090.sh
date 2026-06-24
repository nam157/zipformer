#!/usr/bin/env bash
# RTX 3090 24GB — đủ VRAM cho batch lớn, full multi-dataset stage B.
set -eou pipefail
EXP=exp/vi_zipformer_3090
mkdir -p $EXP

# Tối ưu cho 24GB:
#  - max-duration 500s (~50-60 utterance/batch)
#  - fp16 (Ampere không có bf16 native nhanh bằng fp16, dùng fp16 + GradScaler)
#  - không cần grad-checkpoint
#  - num-workers 8, prefetch 4
#  - TF32 cho matmul (Ampere)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# Stage B: full multi-corpus, oversample clean tier ×2
python zipformer/train.py \
  --manifest-dir data/manifests \
  --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --rare-tokens data/lang_bpe_500/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_full_fbank.jsonl.gz \
  --tier-weights "clean=2.0,medium=1.0,noisy=0.7" \
  --exp-dir $EXP \
  --pretrain-ckpt exp/vi_zipformer_stageA/epoch-15.pt \
  --init-modules encoder,decoder,joiner \
  --num-epochs 30 --start-epoch 1 \
  --max-duration 500 \
  --grad-accum-steps 1 \
  --num-workers 8 \
  --base-lr 0.035 --lr-epochs 6 --lr-batches 7500 \
  --prune-range 5 --simple-loss-scale 0.5 \
  --precision fp16 \
  --grad-checkpoint 0 \
  --grad-clip 5.0 \
  --log-interval 50 --save-every-n 4000 \
  --world-size 1 \
  --tf32 1
