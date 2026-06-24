#!/usr/bin/env bash
set -eou pipefail

EXP=exp/vi_zipformer
mkdir -p $EXP

python zipformer/train.py \
  --manifest-dir data/manifests \
  --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --rare-tokens data/lang_bpe_500/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --exp-dir $EXP \
  --pretrain-ckpt pretrain/zipformer_en_streaming.pt \
  --init-modules encoder \
  --num-epochs 30 --start-epoch 1 \
  --max-duration 600 --num-workers 4 \
  --base-lr 0.035 --lr-epochs 6 --lr-batches 7500 \
  --prune-range 5 --simple-loss-scale 0.5 \
  --use-fp16 1 --grad-clip 5.0 \
  --log-interval 50 --save-every-n 4000 \
  --world-size 1
