# Vietnamese ZipFormer-RNNT Fine-tuning

Streaming ASR cho tiếng Việt dựa trên **ZipFormer + RNN-Transducer** (k2 + icefall), tối ưu cho **CPU inference**.

- Params: ~30M
- Loss: pruned RNN-T (k2)
- Chunk sizes: 16 / 32 / 64 frames (multi-chunk training)
- Train corpus: ~6000h từ 11 nguồn public Việt
- Target: WER < 10% test sạch, RTF < 0.3 trên CPU single-thread

---

## Mục lục

1. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
2. [Setup môi trường](#setup-môi-trường)
3. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Export ONNX cho CPU](#export-onnx-cho-cpu)
7. [Hyperparameter cheatsheet](#hyperparameter-cheatsheet)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Cấu trúc thư mục

```
vi_zipformer_ft/
├── README.md                         # file này
├── requirements.txt
├── configs/
│   └── sources_6000h.md              # phân tích corpus + chiến lược weighting
├── local/                            # script chuẩn bị data
│   ├── hf_datasets.py                # tải 11 dataset HF + VLSP local
│   ├── normalize_audio.py            # ffmpeg 16k mono + loudnorm
│   ├── normalize_text.py             # NFC, num→chữ, strip punct
│   ├── whisper_refine.py             # refine label bằng Whisper large-v3
│   ├── build_manifests.py            # single-source manifest
│   ├── build_manifests_multi.py      # multi-source w/ per-source dev/test
│   ├── compute_fbank.py              # 80-dim fbank
│   ├── train_bpe.py                  # SentencePiece BPE
│   └── oov_stats.py                  # token tail analysis
├── zipformer/                        # model + train + eval
│   ├── model.py                      # ZipFormer2 + Decoder + Joiner
│   ├── asr_datamodule.py             # dataloader + augment + tier/source weights
│   ├── train.py                      # finetune loop
│   ├── decode.py                     # greedy / modified-beam-search
│   ├── eval.py                       # WER/CER + RTF CPU benchmark
│   └── export_onnx.py                # export ONNX + INT8 quantize
├── scripts/                          # workflow đầu-cuối
│   ├── prepare.sh                    # pipeline data đơn giản
│   ├── prepare_hf.sh                 # pipeline 11 dataset HF (~6000h)
│   ├── run_train.sh                  # train default
│   ├── run_train_4070.sh             # RTX 4070 12GB preset
│   ├── run_train_3090.sh             # RTX 3090 24GB preset
│   ├── run_train_2stage.sh           # 2-stage clean→full (3 dataset)
│   ├── run_train_6000h.sh            # 2-stage 6000h mix
│   └── run_eval.sh                   # decode + eval + export
└── data/                             # tạo runtime
    ├── raw/                          # VLSP đặt vào đây (manual)
    ├── wav16k/                       # audio đã chuẩn hoá
    ├── manifests/                    # Lhotse cuts (jsonl.gz)
    ├── fbank/                        # features lilcom
    └── lang_bpe_2000/                # BPE model + vocab
```

---

## Setup môi trường

### Yêu cầu phần cứng
- **Training**: RTX 4070 (12GB) tối thiểu, RTX 3090/4090 (24GB) khuyên dùng
- **Disk**: ≥ 1TB SSD (corpus 6000h ~500-800GB sau resample)
- **RAM**: ≥ 32GB (lhotse load manifest)
- **CPU**: ≥ 8 cores cho data prep song song

### Install

```bash
# Python 3.10+
conda create -n vi_asr python=3.10 -y && conda activate vi_asr

# PyTorch (chọn theo CUDA version của bro)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# k2 (phải match torch + cuda version)
pip install k2==1.24.4.dev20240606+cuda12.1.torch2.3.1 -f https://k2-fsa.github.io/k2/cuda.html

# Pipeline deps
pip install -r requirements.txt

# icefall (cần PYTHONPATH, không pip install)
git clone https://github.com/k2-fsa/icefall ../icefall
cd ../icefall && pip install -r requirements.txt && cd -
export PYTHONPATH=$(pwd)/../icefall:$(pwd):$PYTHONPATH

# Verify
python -c "import torch, k2, lhotse; print(torch.__version__, k2.__version__, lhotse.__version__)"
```

Thêm vào `~/.bashrc`:
```bash
export PYTHONPATH=/path/to/icefall:/path/to/vi_zipformer_ft:$PYTHONPATH
```

### MUSAN noise (cho augmentation)

```bash
cd data
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz
python -c "
from lhotse.recipes import prepare_musan
from lhotse import CutSet
m = prepare_musan('data/musan', output_dir='data/manifests')
cuts = CutSet.from_manifests(recordings=m['music']['recordings']) + \
       CutSet.from_manifests(recordings=m['noise']['recordings'])
cuts.to_jsonl('data/manifests/musan_cuts.jsonl.gz')
"
```

### Pretrained checkpoint (English ZipFormer streaming)

```bash
mkdir -p pretrain
# Download từ HF
wget -O pretrain/zipformer_en_streaming.pt \
  https://huggingface.co/k2-fsa/icefall-asr-librispeech-zipformer-streaming-2023-05-17/resolve/main/exp/pretrained.pt
```

---

## Chuẩn bị dữ liệu

### Option A: Mix 11 dataset HuggingFace (~6000h, khuyên dùng)

**Bước 1 — tải VLSP về máy** (phải đăng ký tại https://vlsp.org.vn):
```
data/raw/vlsp2020/metadata.tsv      # format: utt_id\ttext\trelative_wav_path
data/raw/vlsp2020/audio/*.wav
data/raw/vlsp2021/metadata.tsv
data/raw/vlsp2021/audio/*.wav
data/raw/vlsp2023/metadata.tsv
data/raw/vlsp2023/audio/*.wav
```

**Bước 2 — pipeline đầu-cuối**:
```bash
bash scripts/prepare_hf.sh
```

Script này chạy 5 stage tuần tự (mất 1-3 ngày tuỳ băng thông + GPU):
1. Tải 11 dataset HF (streaming) → resample 16k mono → `data/wav16k/`
2. Chuẩn hoá text (NFC, số→chữ, lowercase, strip punct)
3. Whisper refine cho tier noisy (BUD500, GigaSpeech2-Vi)
4. Build Lhotse manifests: `cuts_train_clean.jsonl.gz`, `cuts_train_full.jsonl.gz`, dev/test **per-source**
5. Compute fbank 80-dim + BPE 2000 + OOV stats

**Chạy từng stage riêng** (nếu fail giữa chừng):
```bash
stage=0 stop_stage=0 bash scripts/prepare.sh   # chỉ stage 0
stage=2 stop_stage=4 bash scripts/prepare.sh   # stage 2 → 4
```

### Option B: Chỉ dùng data riêng / dataset nhỏ

Format `data/raw/transcript.tsv` (3 cột tab-separated):
```
utt_001  xin chào các bạn  data/raw/audio/utt_001.wav
utt_002  hôm nay trời đẹp  data/raw/audio/utt_002.wav
```

```bash
bash scripts/prepare.sh
```

### Kiểm tra data đã chuẩn bị xong

```bash
ls -lh data/manifests/
# Phải có:
#   cuts_train_clean_fbank.jsonl.gz
#   cuts_train_full_fbank.jsonl.gz
#   cuts_dev_fbank.jsonl.gz
#   cuts_test_fbank.jsonl.gz
#   cuts_dev_<source>.jsonl.gz   # per-source eval
#   cuts_test_<source>.jsonl.gz

# Đếm số utterance
python -c "
from lhotse import load_manifest_lazy
for s in ['train_clean', 'train_full', 'dev', 'test']:
    c = load_manifest_lazy(f'data/manifests/cuts_{s}_fbank.jsonl.gz')
    n = sum(1 for _ in c); hrs = sum(x.duration for x in load_manifest_lazy(f'data/manifests/cuts_{s}_fbank.jsonl.gz'))/3600
    print(f'{s}: {n} cuts, {hrs:.1f}h')
"
```

---

## Training

### Chọn script theo GPU

| GPU | Script | Effective batch | Precision | Thời gian (30 epoch, 6000h) |
|---|---|---|---|---|
| RTX 4070 12GB | `run_train_4070.sh` | ~720s | bf16 + grad-accum=4 | ~10-14 ngày |
| RTX 3090 24GB | `run_train_3090.sh` | ~500s | fp16 | ~5-7 ngày |
| RTX 4090 24GB | `run_train_3090.sh` | ~500s | bf16 (đổi `PREC=bf16`) | ~3-5 ngày |

### Quick start — 6000h, 2-stage

```bash
# RTX 3090
bash scripts/run_train_6000h.sh 3090

# RTX 4070
bash scripts/run_train_6000h.sh 4070
```

**Stage A** (15 epoch, ~1250h clean): init từ pretrain English, học âm vị Việt.
**Stage B** (15 epoch, ~3200h hiệu dụng): init từ Stage A, học robust trên mixed corpus.

Output ở `exp/vi_zipformer_6kh_stageB/epoch-{1..15}.pt`.

### Theo dõi training

```bash
# TensorBoard
tensorboard --logdir exp/vi_zipformer_6kh_stageB/tb --port 6006

# Log realtime
tail -f exp/vi_zipformer_6kh_stageB/log/log-train-*
```

Metric cần watch:
- `train/loss_per_frame`: giảm đều, không jump
- `dev/loss_per_frame`: stage A nên về ~0.05-0.08, stage B ~0.04-0.06
- `train/lr`: theo schedule Eden (warmup → decay)

### Resume training

```bash
python zipformer/train.py \
  --exp-dir exp/vi_zipformer_6kh_stageB \
  --start-epoch 8 \
  ... # các arg khác giữ nguyên
```

Script tự load `exp/.../epoch-7.pt` làm điểm bắt đầu.

### Multi-GPU (nếu có)

```bash
# Trong train.py args
--world-size 2    # 2 GPU
```
Script sẽ spawn DDP qua `torch.multiprocessing`.

---

## Evaluation

### Quick eval — WER/CER + RTF cho 3 chunk size

```bash
bash scripts/run_eval.sh
```

Output:
- `exp/.../hyp-test-greedy_search.txt`: hypothesis từng utterance
- `exp/.../eval-epoch30-avg10.json`: WER/CER/RTF cho chunk 16/32/64
- Stdout in: `WER=X.XX% CER=Y.YY%`

### Eval per-source (xem domain nào yếu)

```bash
for src in vlsp2020 fosd bud500 gigaspeech2_vi vivoice vietmed; do
  python zipformer/decode.py \
    --manifest-dir data/manifests --fbank-dir data/fbank \
    --bpe-model data/lang_bpe_2000/bpe.model \
    --exp-dir exp/vi_zipformer_6kh_stageB \
    --epoch 15 --avg 5 \
    --method modified_beam_search --beam-size 4 \
    --split test \
    --test-manifest cuts_test_${src}.jsonl.gz \
    --chunk-size 32 --left-context-frames 128
done
```

(thêm flag `--test-manifest` nếu chưa có — sửa `decode.py` để load đúng file)

### Average checkpoints (giảm WER ~5-10%)

```bash
# Trong decode.py / eval.py: --avg 10 sẽ average epoch-21..30
python zipformer/decode.py --epoch 30 --avg 10 ...
```

### Benchmark RTF CPU thật

```bash
# Cài đặt single-thread
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python zipformer/eval.py \
  --exp-dir exp/vi_zipformer_6kh_stageB --epoch 15 --avg 5 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --manifest-dir data/manifests --fbank-dir data/fbank \
  --chunk-sizes 16,32,64 --bench-rtf 1
```

RTF target:
- chunk 16: < 0.15 (low latency)
- chunk 32: < 0.10 (balanced)
- chunk 64: < 0.07 (throughput)

---

## Export ONNX cho CPU

```bash
python zipformer/export_onnx.py \
  --exp-dir exp/vi_zipformer_6kh_stageB \
  --epoch 15 --avg 5 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --chunk-size 32 --left-context-frames 128 \
  --quantize 1
```

Output `exp/.../onnx/`:
- `encoder.onnx`, `decoder.onnx`, `joiner.onnx` (FP32)
- `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx` (INT8 dynamic)

### Deploy với sherpa-onnx

```bash
pip install sherpa-onnx

sherpa-onnx \
  --encoder=exp/.../onnx/encoder.int8.onnx \
  --decoder=exp/.../onnx/decoder.int8.onnx \
  --joiner=exp/.../onnx/joiner.int8.onnx \
  --tokens=data/lang_bpe_2000/tokens.txt \
  --num-threads=1 \
  test_audio.wav
```

INT8 trên CPU thường nhanh hơn FP32 **2-3x** với mất ~0.5-1% WER.

---

## Hyperparameter cheatsheet

### LR scheduling (Eden)
- `--base-lr`: 0.035 (from scratch), 0.025-0.03 (finetune từ pretrain)
- `--lr-epochs`: 4-6 (decay epoch interval)
- `--lr-batches`: 7500-10000 (warmup steps)

### Loss
- `--prune-range`: 5 (k2 RNN-T pruning). Tăng → accuracy cao hơn, VRAM nhiều hơn
- `--simple-loss-scale`: 0.5 (đầu epoch 0 nên 1.0, về sau giảm về 0)
- `--am-scale 0.0 --lm-scale 0.0`: tắt label smoothing (default OK)

### Regularization
- SpecAugment: 2 freq mask × 27 dims, 2 time mask × 100 frames (đã hardcode)
- Speed perturb: 0.9/1.1, p=0.5
- MUSAN noise: SNR 10-20 dB, p=0.5
- `--grad-clip 5.0`

### Source weighting (mix 6000h)
Xem chi tiết ở [configs/sources_6000h.md](configs/sources_6000h.md).

Quick reference:
```
gigaspeech2_vi=0.3    # cap dominant noisy
bud500=0.5             # cap noisy
vivoice=0.6            # cap dominant clean (TTS bias)
vlsp*=1.5              # boost medium quality
fosd|fleurs_vi=2.5     # boost clean & rare
vietmed=3.0            # boost rare domain
phoaudiobook=2.0       # boost audiobook
```

---

## Troubleshooting

### OOM (CUDA out of memory)

| Triệu chứng | Fix |
|---|---|
| OOM ngay batch đầu | Giảm `--max-duration` về 1/2 |
| OOM sau N batch | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| OOM ở 4070 với 200s | Bật `--grad-checkpoint 1`, tăng `--grad-accum-steps` |
| OOM khi compute features | `--num-jobs 4` thay vì 8 |

### `k2.RnntLossError` / NaN loss

- Lower `--base-lr` xuống 0.02
- Đảm bảo `--prune-range >= 5`
- Check label: có cut nào text rỗng sau normalize không?
  ```python
  from lhotse import load_manifest_lazy
  c = load_manifest_lazy('data/manifests/cuts_train_full_fbank.jsonl.gz')
  for x in c:
      if not x.supervisions[0].text.strip():
          print(f"EMPTY: {x.id}")
  ```

### Whisper refine quá chậm

- Dùng `whisper-large-v3-turbo` thay `large-v3` (4x nhanh, accuracy gần bằng)
- Hoặc skip refine cho tier `clean` (đã skip mặc định)
- Batch nhiều file qua `faster-whisper` hoặc `whisperx`

### Train loss giảm nhưng dev WER không cải thiện

- Overfit corpus dominant (GigaSpeech2/ViVoice) → giảm `source_weights` của chúng
- Check per-source dev WER — nếu chỉ vài domain tệ thì cần thêm data domain đó
- Tăng SpecAugment: sửa `asr_datamodule.py` → `frames_mask_size=120, num_frame_masks=3`

### Lhotse load chậm / RAM hết

- Dùng `load_manifest_lazy` (đã dùng) thay `load_manifest`
- Manifest >50GB: split thành nhiều shard, dùng `CutSet.mux()`

### HuggingFace download lỗi

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer

# Resume
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

GigaSpeech2 nên dùng `streaming=True` (đã set) — không lưu cache toàn bộ.

---

## FAQ

**Q: Train 6000h trên 4070 12GB có khả thi không?**
A: Khả thi nhưng chậm (10-14 ngày). Đề xuất:
- Stage A trên 4070 (clean 1250h, ~3 ngày)
- Thuê 3090/4090 cloud (vast.ai ~$0.3/h) cho Stage B (~5 ngày × $0.3 × 24 = $36)

**Q: Cần Whisper refine cho ViVoice / PhoAudioBook không?**
A: Không. Đó là TTS-grade, label đã align tốt.

**Q: Vocab BPE 500 vs 2000?**
A: 500 cho corpus nhỏ (<500h), 2000 cho 6000h. Unigram 4000 thậm chí tốt hơn cho VI vì tách phụ âm/vần.

**Q: Có thể skip k2 không?**
A: Không nếu dùng pruned RNN-T loss. Có thể thay bằng `torchaudio.functional.rnnt_loss` (chậm 3-5x, không pruned).

**Q: Model 30M có chạy được trên mobile không?**
A: Có. Export ONNX INT8 → ~30MB. Test trên Pixel 6/iPhone 12: RTF ~0.3-0.5 chunk 32.

**Q: So với Whisper / wav2vec2 thì sao?**
A:
- Whisper-large: WER thấp hơn ~1-2%, RTF ~5x cao hơn, không streaming → khác use case
- wav2vec2-vi: WER tương đương, CTC loss nên không tốt cho streaming length-mismatch
- Mô hình này nhằm vào: streaming + CPU realtime, không phải SOTA WER tuyệt đối

**Q: Làm sao train nhanh hơn?**
A:
1. Dùng A100/H100 cloud (10-20x nhanh hơn 3090 cho fp16)
2. Giảm số epoch stage A (10 thay vì 15)
3. Tăng `--max-duration` đến giới hạn VRAM (test bằng `nvidia-smi`)
4. Pre-compute fbank trước, không augment on-the-fly khi prototype

**Q: Có cần language model (LM) shallow fusion không?**
A: Optional. Dùng kenlm 4-gram trên text crawl Việt:
```bash
# (chưa có script — sẽ thêm sau nếu cần)
```
Cải thiện WER ~0.5-2% trên domain hiếm.

---

## Citation / References

- ZipFormer: https://arxiv.org/abs/2310.11230
- icefall: https://github.com/k2-fsa/icefall
- k2: https://github.com/k2-fsa/k2
- Lhotse: https://github.com/lhotse-speech/lhotse
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx

## License

Code: MIT. Dữ liệu: theo license của từng nguồn (VLSP có ràng buộc thương mại).
