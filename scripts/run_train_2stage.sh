#!/usr/bin/env bash
# 2-stage curriculum: A) clean-only 15 epoch  ->  B) full mix 15 epoch
# Chạy được trên cả 4070 (chỉnh max-duration) hoặc 3090.
set -eou pipefail

GPU=${1:-3090}    # ./run_train_2stage.sh 4070  |  3090

if [ "$GPU" = "4070" ]; then
  MAXDUR=180; ACC=4; PREC=bf16; CKPT=1; NW=6
else
  MAXDUR=500; ACC=1; PREC=fp16; CKPT=0; NW=8
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP_A=exp/vi_zipformer_stageA
EXP_B=exp/vi_zipformer_stageB
mkdir -p $EXP_A $EXP_B

# ===== Stage A: clean tier =====
python zipformer/train.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --rare-tokens data/lang_bpe_500/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_clean_fbank.jsonl.gz \
  --tier-weights "clean=1.0" \
  --exp-dir $EXP_A \
  --pretrain-ckpt pretrain/zipformer_en_streaming.pt --init-modules encoder \
  --num-epochs 15 --max-duration $MAXDUR --grad-accum-steps $ACC \
  --num-workers $NW --precision $PREC --grad-checkpoint $CKPT \
  --base-lr 0.03 --lr-epochs 4 --tf32 1

# ===== Stage B: full mix, init từ stage A =====
python zipformer/train.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --rare-tokens data/lang_bpe_500/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_full_fbank.jsonl.gz \
  --tier-weights "clean=2.0,medium=1.0,noisy=0.7" \
  --exp-dir $EXP_B \
  --pretrain-ckpt $EXP_A/epoch-15.pt --init-modules encoder,decoder,joiner \
  --num-epochs 15 --start-epoch 1 --max-duration $MAXDUR --grad-accum-steps $ACC \
  --num-workers $NW --precision $PREC --grad-checkpoint $CKPT \
  --base-lr 0.02 --lr-epochs 4 --tf32 1
