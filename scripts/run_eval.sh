#!/usr/bin/env bash
set -eou pipefail
EXP=exp/vi_zipformer

# 1) WER/CER greedy
python zipformer/decode.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --exp-dir $EXP --epoch 30 --avg 10 \
  --method greedy_search --split test \
  --chunk-size 32 --left-context-frames 128

# 2) modified-beam-search
python zipformer/decode.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --exp-dir $EXP --epoch 30 --avg 10 \
  --method modified_beam_search --beam-size 4 \
  --split test --chunk-size 32 --left-context-frames 128

# 3) Full eval (3 chunk sizes + RTF CPU 1-thread)
python zipformer/eval.py \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --exp-dir $EXP --epoch 30 --avg 10 \
  --chunk-sizes 16,32,64 --bench-rtf 1

# 4) Export ONNX INT8 cho CPU deploy
python zipformer/export_onnx.py \
  --exp-dir $EXP --epoch 30 --avg 10 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --chunk-size 32 --left-context-frames 128 \
  --quantize 1
