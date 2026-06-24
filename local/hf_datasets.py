#!/usr/bin/env python3
"""Tải + chuẩn hoá nhiều dataset VI từ HuggingFace, gộp về 1 TSV chung
   format: <utt_id>\t<text>\t<wav_path>\t<source>\t<tier>

tier: clean | medium | noisy  (ảnh hưởng cách Whisper-refine + weight sampling)
"""
import argparse, csv, hashlib, os, re
from pathlib import Path
from datasets import load_dataset, Audio
import soundfile as sf

# ----- Catalog -----
DATASETS = {
    "common_voice": dict(
        hf_id="mozilla-foundation/common_voice_17_0", config="vi",
        split="train+validation", text_col="sentence", audio_col="audio",
        tier="clean",
    ),
    "vivos": dict(
        hf_id="AILAB-VNUHCM/vivos", config=None,
        split="train", text_col="sentence", audio_col="audio",
        tier="clean",
    ),
    "vietbible": dict(
        hf_id="ntt123/viet-bible-vox", config=None,
        split="train", text_col="text", audio_col="audio",
        tier="clean",
    ),
    "fosd": dict(
        hf_id="doof-ferb/fpt_fosd", config=None,
        split="train", text_col="transcription", audio_col="audio",
        tier="clean",
    ),
    "vietmed": dict(
        hf_id="leduckhai/VietMed", config=None,
        split="train", text_col="transcript", audio_col="audio",
        tier="clean",
    ),
    "lsvsc": dict(
        hf_id="doof-ferb/LSVSC", config=None,
        split="train", text_col="transcription", audio_col="audio",
        tier="medium",
    ),
    "bud500": dict(
        hf_id="linhtran92/viet_bud500", config=None,
        split="train", text_col="transcription", audio_col="audio",
        tier="noisy",  # cần Whisper refine
    ),
    "yodas_vi": dict(
        hf_id="espnet/yodas", config="vi000",
        split="train", text_col="text", audio_col="audio",
        tier="noisy",
    ),
    # ===== Datasets bổ sung (tổng ~6000h) =====
    "fleurs_vi": dict(
        hf_id="google/fleurs", config="vi_vn",
        split="train", text_col="transcription", audio_col="audio",
        tier="clean",
    ),
    "vietspeech": dict(
        # Mirror HF của VietSpeech (NhutP) ~100h read/spontaneous
        hf_id="NhutP/VSV-1100", config=None,
        split="train", text_col="transcription", audio_col="audio",
        tier="medium",
    ),
    "gigaspeech2_vi": dict(
        # SpeechColab GigaSpeech2 — VI subset ~3000h, YouTube auto-transcribed
        hf_id="speechcolab/gigaspeech2", config="vi",
        split="train", text_col="text", audio_col="audio",
        tier="noisy",
    ),
    "vivoice": dict(
        # capleaf/viVoice ~1000h TTS-grade VI
        hf_id="capleaf/viVoice", config=None,
        split="train", text_col="text", audio_col="audio",
        tier="clean",
    ),
    "phoaudiobook": dict(
        # PhoAudioBook subset — audiobook đọc rõ, ~200h
        hf_id="vinai/PhoAudioBook", config=None,
        split="train", text_col="text", audio_col="audio",
        tier="clean",
    ),
    "vlsp2020": dict(
        # VLSP đòi đăng ký — đặt cờ local_path, KHÔNG tải qua HF
        hf_id=None, config=None, split=None,
        local_path="data/raw/vlsp2020",        # tự chỉnh path
        text_col="transcript", audio_col="audio",
        tier="medium",
    ),
    "vlsp2021": dict(
        hf_id=None, config=None, split=None,
        local_path="data/raw/vlsp2021",
        text_col="transcript", audio_col="audio",
        tier="medium",
    ),
    "vlsp2023": dict(
        hf_id=None, config=None, split=None,
        local_path="data/raw/vlsp2023",
        text_col="transcript", audio_col="audio",
        tier="medium",
    ),
}

# Các pattern bẩn hay gặp ở YouTube subtitles / VLSP
DIRTY_PATTERNS = [
    r"\[.*?\]", r"\(.*?\)", r"<.*?>",          # [Music], (laugh), <unk>
    r"♪.*?♪",
]
DIRTY_RE = re.compile("|".join(DIRTY_PATTERNS))

def clean_text(t: str) -> str:
    t = DIRTY_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def utt_id(source: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{source}-{idx}-{text}".encode("utf-8")).hexdigest()[:10]
    return f"{source}_{idx:08d}_{h}"

def dump_local_dataset(name: str, cfg: dict, out_wav_dir: Path, writer,
                       sample_rate: int = 16000):
    """Load VLSP-style local dataset: expect folder with audio + metadata.tsv
       hoặc cấu trúc <split>/<wav> + transcript.txt với format `<utt>\t<text>`."""
    root = Path(cfg["local_path"])
    if not root.exists():
        print(f"[{name}] SKIP — {root} không tồn tại (cần đăng ký VLSP)")
        return
    # Tìm metadata.tsv hoặc transcript.txt
    tsv_files = list(root.rglob("metadata.tsv")) + list(root.rglob("transcript.txt"))
    n_ok = 0
    for tsv in tsv_files:
        base = tsv.parent
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2: continue
                utt, text = parts[0], parts[1]
                # Audio có thể là cột 3 hoặc tự suy ra từ utt id
                wav = parts[2] if len(parts) >= 3 else f"{utt}.wav"
                wav_p = (base / wav) if not Path(wav).is_absolute() else Path(wav)
                if not wav_p.exists(): continue
                text = clean_text(text)
                if len(text) < 2: continue
                uid = utt_id(name, n_ok, text)
                writer.writerow([uid, text, str(wav_p), name, cfg["tier"]])
                n_ok += 1
    print(f"[{name}] local done: {n_ok}")

def dump_dataset(name: str, cfg: dict, out_wav_dir: Path, writer,
                 sample_rate: int = 16000, max_items: int = None):
    if cfg.get("local_path"):
        dump_local_dataset(name, cfg, out_wav_dir, writer, sample_rate)
        return
    print(f"[{name}] loading {cfg['hf_id']} ({cfg.get('config')})")
    ds = load_dataset(
        cfg["hf_id"], cfg.get("config"),
        split=cfg["split"], trust_remote_code=True, streaming=True,
    )
    ds = ds.cast_column(cfg["audio_col"], Audio(sampling_rate=sample_rate))

    out_dir = out_wav_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, ex in enumerate(ds):
        if max_items and i >= max_items: break
        text = clean_text(str(ex.get(cfg["text_col"]) or ""))
        if len(text) < 2: continue
        audio = ex[cfg["audio_col"]]
        arr, sr = audio["array"], audio["sampling_rate"]
        uid = utt_id(name, i, text)
        # Shard dir để tránh 1 folder triệu file
        shard = out_dir / f"{i // 5000:05d}"
        shard.mkdir(parents=True, exist_ok=True)
        wav_p = shard / f"{uid}.wav"
        if not wav_p.exists():
            sf.write(wav_p, arr, sr, subtype="PCM_16")
        writer.writerow([uid, text, str(wav_p), name, cfg["tier"]])
        n_ok += 1
        if n_ok % 1000 == 0: print(f"  {name}: {n_ok}")
    print(f"[{name}] done: {n_ok}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-wav-dir", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS.keys()),
                    help="comma list, vd: common_voice,vivos,fosd")
    ap.add_argument("--max-items-per-ds", type=int, default=None)
    args = ap.parse_args()

    Path(args.out_wav_dir).mkdir(parents=True, exist_ok=True)
    chosen = [d.strip() for d in args.datasets.split(",")]

    with open(args.out_tsv, "w", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        for name in chosen:
            if name not in DATASETS:
                print(f"skip unknown: {name}"); continue
            dump_dataset(name, DATASETS[name], Path(args.out_wav_dir), w,
                         max_items=args.max_items_per_ds)

if __name__ == "__main__":
    main()
