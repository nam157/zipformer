#!/usr/bin/env python3
"""Refine transcript bằng Whisper: nếu WER(ref, whisper) <= max-wer thì giữ ref;
nếu lệch lớn, có thể thay bằng whisper output hoặc drop.

Dùng faster-whisper (CTranslate2) để giảm VRAM so với openai-whisper gốc.
  - float16 (default): giảm ~50% so với float32
  - int8_float16: giảm ~75%, tốc độ nhanh hơn, accuracy giảm nhẹ
  - int8: dùng CPU hoặc GPU cũ không hỗ trợ float16
"""
import argparse, csv
from pathlib import Path
import jiwer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--replace-wer", type=float, default=0.4,
                    help="0.15..0.4: thay bằng whisper; >0.4: drop")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--compute-type", default="float16",
        choices=["float32", "float16", "int8_float16", "int8"],
        help="Quantization type. Dùng int8_float16 để tiết kiệm VRAM nhất trên GPU; "
             "int8 cho CPU. Mặc định: float16."
    )
    ap.add_argument(
        "--beam-size", type=int, default=5,
        help="Beam size cho decoding. Giảm xuống 1-2 để tiết kiệm thêm VRAM."
    )
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    device = args.device if args.device != "cuda" else "cuda"
    model = WhisperModel(
        args.whisper_model,
        device=device,
        compute_type=args.compute_type,
    )

    kept, replaced, dropped = 0, 0, 0
    with open(args.tsv, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        for row in csv.reader(fi, delimiter="\t"):
            if len(row) < 3:
                continue
            utt, ref, wav = row[0], row[1], row[2]
            extra = row[3:]  # giữ lại cột source, tier (nếu có)
            # TSV may store full relative paths (e.g. data/wav16k/fosd/...)
            # or just the basename — try as-is first, then join with wav-dir.
            wav_p = Path(wav) if Path(wav).is_absolute() else Path(wav)
            if not wav_p.exists():
                wav_p = Path(args.wav_dir) / wav
            if not wav_p.exists():
                continue
            segments, _info = model.transcribe(
                str(wav_p), language="vi", beam_size=args.beam_size
            )
            hyp = " ".join(seg.text for seg in segments).strip().lower()
            try:
                wer = jiwer.wer(ref, hyp)
            except Exception:
                wer = 1.0

            if wer <= args.max_wer:
                w.writerow([utt, ref, str(wav_p)] + extra); kept += 1
            elif wer <= args.replace_wer:
                w.writerow([utt, hyp, str(wav_p)] + extra); replaced += 1
            else:
                dropped += 1
    print(f"kept={kept} replaced={replaced} dropped={dropped}")

if __name__ == "__main__":
    main()
