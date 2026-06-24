#!/usr/bin/env bash
# RTX 4070 12GB — config tiết kiệm VRAM, dùng grad-accum để giả lập batch lớn.
set -eou pipefail
EXP=exp/vi_zipformer_4070
mkdir -p $EXP

# Tối ưu cho 12GB:
#  - max-duration 180s (~1.8x dùng được, batch còn ~16-24 utterance)
#  - grad-accum 4 -> effective max-duration 720s
#  - bf16 (RTX 40x hỗ trợ tốt hơn fp16, ổn định gradient hơn cho RNN-T)
#  - num-workers 6 (Ryzen/Intel mid-range), persistent + pin
#  - gradient checkpointing bật (giảm ~30% VRAM encoder)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

python zipformer/train.py \
  --manifest-dir data/manifests \
  --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --rare-tokens data/lang_bpe_500/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_clean_fbank.jsonl.gz \
  --tier-weights "clean=1.0" \
  --exp-dir $EXP \
  --pretrain-ckpt pretrain/zipformer_en_streaming.pt \
  --init-modules encoder \
  --num-epochs 30 --start-epoch 1 \
  --max-duration 180 \
  --grad-accum-steps 4 \
  --num-workers 6 \
  --base-lr 0.03 --lr-epochs 6 --lr-batches 6000 \
  --prune-range 5 --simple-loss-scale 0.5 \
  --precision bf16 \
  --grad-checkpoint 1 \
  --grad-clip 5.0 \
  --log-interval 50 --save-every-n 4000 \
  --world-size 1 \
  --tf32 1
