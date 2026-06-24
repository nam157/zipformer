#!/usr/bin/env bash
# Train 6000h mix VI: VLSP{2020,2021,2023} + FPT + BUD500 + VietSpeech +
# FLEURS + VietMed + GigaSpeech2-Vi + ViVoice + PhoAudioBook
# Usage: bash scripts/run_train_6000h.sh 3090   |   4070
set -eou pipefail
GPU=${1:-3090}

if [ "$GPU" = "4070" ]; then
  MAXDUR=180; ACC=4; PREC=bf16; CKPT=1; NW=6
else
  MAXDUR=500; ACC=1; PREC=fp16; CKPT=0; NW=8
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_A=exp/vi_zipformer_6kh_stageA
EXP_B=exp/vi_zipformer_6kh_stageB
mkdir -p $EXP_A $EXP_B

# ========== Stage A — clean tier (~1250h), 15 epoch ==========
# Upsample domain hiếm (VietMed, FLEURS); cap ViVoice để tránh bias TTS prosody
python zipformer/train.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --rare-tokens data/lang_bpe_2000/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_clean_fbank.jsonl.gz \
  --source-weights "vietmed=3.0,fleurs_vi=2.5,vivoice=0.5,phoaudiobook=1.5,fosd=2.0" \
  --exp-dir $EXP_A \
  --pretrain-ckpt pretrain/zipformer_en_streaming.pt --init-modules encoder \
  --num-epochs 15 --start-epoch 1 \
  --max-duration $MAXDUR --grad-accum-steps $ACC \
  --num-workers $NW --precision $PREC --grad-checkpoint $CKPT --tf32 1 \
  --base-lr 0.03 --lr-epochs 4 --lr-batches 7500 \
  --prune-range 5 --simple-loss-scale 0.5 --grad-clip 5.0 \
  --log-interval 50 --save-every-n 5000

# ========== Stage B — full 6000h, cap dominant sources, 15 epoch ==========
# GigaSpeech2 (3000h, noisy) cap ×0.3 → ~900h hiệu dụng
# ViVoice (1000h, clean nhưng TTS) cap ×0.6 → ~600h hiệu dụng
# VLSP/VietSpeech medium ×1.5; clean domain ×2-3
python zipformer/train.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --rare-tokens data/lang_bpe_2000/rare_tokens.json \
  --musan-cuts data/manifests/musan_cuts.jsonl.gz \
  --train-manifest cuts_train_full_fbank.jsonl.gz \
  --source-weights "gigaspeech2_vi=0.3,bud500=0.5,vivoice=0.6,vlsp2020=1.5,vlsp2021=1.5,vlsp2023=1.5,vietspeech=1.5,fosd=2.5,fleurs_vi=2.5,vietmed=3.0,phoaudiobook=2.0" \
  --exp-dir $EXP_B \
  --pretrain-ckpt $EXP_A/epoch-15.pt --init-modules encoder,decoder,joiner \
  --num-epochs 15 --start-epoch 1 \
  --max-duration $MAXDUR --grad-accum-steps $ACC \
  --num-workers $NW --precision $PREC --grad-checkpoint $CKPT --tf32 1 \
  --base-lr 0.02 --lr-epochs 4 --lr-batches 10000 \
  --prune-range 5 --simple-loss-scale 0.5 --grad-clip 5.0 \
  --log-interval 50 --save-every-n 5000
