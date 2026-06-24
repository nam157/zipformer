#!/usr/bin/env bash
# Train trên 1 dataset duy nhất (vd FOSD 25h hoặc ViVoice 1000h).
# Không có 2-stage, không source-weighting, ít epoch hơn.
#
# Usage: bash scripts/run_train_single.sh 3090   |   4070
set -eou pipefail
GPU=${1:-3090}

if [ "$GPU" = "4070" ]; then
  MAXDUR=180; ACC=4; PREC=bf16; CKPT=1; NW=6
else
  MAXDUR=500; ACC=1; PREC=fp16; CKPT=0; NW=8
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP=exp/vi_zipformer_single
mkdir -p $EXP

# Số epoch tuỳ size:
#   FOSD 25h  → 50 epoch (data nhỏ, cần lặp nhiều)
#   ViVoice 1000h → 25 epoch
# Tự chỉnh NUM_EPOCHS theo dataset bro chạy
NUM_EPOCHS=${NUM_EPOCHS:-30}

python zipformer/train.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --rare-tokens data/lang_bpe_2000/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_full_fbank.jsonl.gz \
  --exp-dir $EXP \
  --pretrain-ckpt pretrain/zipformer_en_streaming.pt --init-modules encoder \
  --num-epochs $NUM_EPOCHS --start-epoch 1 \
  --max-duration $MAXDUR --grad-accum-steps $ACC \
  --num-workers $NW --precision $PREC --grad-checkpoint $CKPT --tf32 1 \
  --base-lr 0.03 --lr-epochs 6 --lr-batches 5000 \
  --prune-range 5 --simple-loss-scale 0.5 --grad-clip 5.0 \
  --log-interval 50 --save-every-n 3000
