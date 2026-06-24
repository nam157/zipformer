#!/usr/bin/env bash
# Pipeline chuẩn bị dữ liệu finetune Vietnamese ZipFormer-RNNT
set -eou pipefail

stage=${stage:-0}
stop_stage=${stop_stage:-100}

DATA_DIR=data
RAW_DIR=$DATA_DIR/raw                 # audio gốc + transcript thô
MANIFEST_DIR=$DATA_DIR/manifests
FBANK_DIR=$DATA_DIR/fbank
LANG_DIR=$DATA_DIR/lang_bpe_500
VOCAB_SIZE=500

mkdir -p $MANIFEST_DIR $FBANK_DIR $LANG_DIR

# ---------- Stage 0: chuẩn hoá audio 16k mono ----------
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: normalize audio -> 16k mono wav"
  python local/normalize_audio.py \
    --src-dir $RAW_DIR/audio \
    --dst-dir $DATA_DIR/wav16k \
    --num-jobs 8
fi

# ---------- Stage 1: chuẩn hoá text + Whisper refine ----------
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: text normalization + Whisper refinement"
  python local/normalize_text.py \
    --in $RAW_DIR/transcript.tsv \
    --out $DATA_DIR/transcript.norm.tsv

  python local/whisper_refine.py \
    --tsv $DATA_DIR/transcript.norm.tsv \
    --wav-dir $DATA_DIR/wav16k \
    --out $DATA_DIR/transcript.refined.tsv \
    --whisper-model large-v3 \
    --max-wer 0.15
fi

# ---------- Stage 2: tạo Lhotse manifests ----------
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: build Lhotse manifests (train/dev/test)"
  python local/build_manifests.py \
    --tsv $DATA_DIR/transcript.refined.tsv \
    --wav-dir $DATA_DIR/wav16k \
    --out-dir $MANIFEST_DIR \
    --dev-ratio 0.01 --test-ratio 0.01 \
    --min-dur 1.0 --max-dur 20.0
fi

# ---------- Stage 3: compute fbank 80-dim ----------
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: compute fbank"
  python local/compute_fbank.py \
    --manifest-dir $MANIFEST_DIR \
    --fbank-dir $FBANK_DIR \
    --num-jobs 8
fi

# ---------- Stage 4: train BPE tokenizer ----------
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: train BPE ($VOCAB_SIZE)"
  cut -f2 $DATA_DIR/transcript.refined.tsv > $LANG_DIR/transcript.txt
  python local/train_bpe.py \
    --transcript $LANG_DIR/transcript.txt \
    --vocab-size $VOCAB_SIZE \
    --lang-dir $LANG_DIR
fi

# ---------- Stage 5: thống kê OOV để augment ----------
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: OOV / rare-token statistics"
  python local/oov_stats.py \
    --bpe-model $LANG_DIR/bpe.model \
    --transcript $LANG_DIR/transcript.txt \
    --out $LANG_DIR/rare_tokens.json \
    --min-count 5
fi

echo "Done. Manifests in $MANIFEST_DIR, fbank in $FBANK_DIR, lang in $LANG_DIR"
