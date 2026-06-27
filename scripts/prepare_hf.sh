#!/usr/bin/env bash
# Tải + chuẩn hoá multi-dataset VI từ HuggingFace, rồi nối tiếp pipeline cũ.
set -eou pipefail

DATA_DIR=data
WAV_DIR=$DATA_DIR/wav16k
TSV_RAW=$DATA_DIR/transcript.hf.tsv
TSV_NORM=$DATA_DIR/transcript.norm.tsv
TSV_REFINED=$DATA_DIR/transcript.refined.tsv

# Stage 0: tải HF datasets (streaming, lưu wav 16k mono)
# 11 nguồn ~6000h (VLSP local; bud500 + gigaspeech2 là gated repo → cần `hf auth login`
# sau khi được approve trên web HF: https://huggingface.co/datasets/{linhtran92/viet_bud500,speechcolab/gigaspeech2})
DATASETS=${DATASETS:-vlsp2020,vlsp2021,vlsp2023,fosd,bud500,vietspeech,fleurs_vi,vietmed,gigaspeech2_vi,vivoice,phoaudiobook}

python local/hf_datasets.py \
  --out-wav-dir $WAV_DIR \
  --out-tsv $TSV_RAW \
  --datasets "$DATASETS"

# Stage 1: chuẩn hoá text (giữ cột source/tier)
python -c "
import csv
from local.normalize_text import normalize
with open('$TSV_RAW') as fi, open('$TSV_NORM','w') as fo:
    r=csv.reader(fi, delimiter='\t'); w=csv.writer(fo, delimiter='\t')
    for row in r:
        if len(row)<5: continue
        t = normalize(row[1])
        if t: w.writerow([row[0], t, row[2], row[3], row[4]])
"

# Stage 2: Whisper refine chỉ áp cho tier noisy (tiết kiệm thời gian)
# --compute-type int8_float16: giảm VRAM ~75% so với float32, accuracy giảm nhẹ
# --beam-size 1: greedy decode, tiết kiệm thêm VRAM & nhanh hơn
python local/whisper_refine.py \
  --tsv $TSV_NORM --wav-dir $WAV_DIR \
  --out $TSV_REFINED --whisper-model large-v3 --max-wer 0.5 \
  --compute-type int8_float16 --beam-size 1

# Stage 3: build multi-source manifests (train_clean + train_full + dev/test per-source)
python local/build_manifests_multi.py \
  --tsv $TSV_REFINED --out-dir $DATA_DIR/manifests \
  --dev-per-source 500 --test-per-source 500

# Stage 4-5: fbank + BPE + OOV stats (giống prepare.sh stage 3-5)
python local/compute_fbank.py --manifest-dir $DATA_DIR/manifests \
  --fbank-dir $DATA_DIR/fbank --num-jobs 8

cut -f2 $TSV_REFINED > $DATA_DIR/lang_bpe_500/transcript.txt
python local/train_bpe.py --transcript $DATA_DIR/lang_bpe_500/transcript.txt \
  --vocab-size 2000 --lang-dir $DATA_DIR/lang_bpe_500
python local/oov_stats.py --bpe-model $DATA_DIR/lang_bpe_500/bpe.model \
  --transcript $DATA_DIR/lang_bpe_500/transcript.txt \
  --out $DATA_DIR/lang_bpe_500/rare_tokens.json --min-count 5
